"""Reproducible benchmark harness for Purce.

Runs the full extraction pipeline over benchmark corpora, compiles every
generated C file with strict gcc flags, and emits a deterministic report
(JSON + Markdown) that can be verified against a committed baseline.

Reproducibility guarantees:
  * Fixed random seed for any stochastic stage (default 42).
  * Content hash computed over sorted generated files with volatile
    timestamps normalized away, so the hash is stable across runs.
  * Environment fingerprint (python/numpy/gcc/git/platform) recorded in
    every report.
  * ``--verify`` mode fails when the report deviates from the baseline
    (counts, compile status, or content hash).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from purce import __version__

try:
    from purce.cli import _process_source
except Exception:  # pragma: no cover - cli always imports in repo tree
    _process_source = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 42
DEFAULT_CORPORA = ["benchmarks/corpus"]
GCC_FLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic", "-c"]

_VOLATILE_PATTERNS = [
    re.compile(r"^\s*\*\s*GENERATED AT:\s+.*$", re.MULTILINE),
    re.compile(r'"generated_at":\s*"[^"]*"'),
]


# ── Fingerprint ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnvironmentFingerprint:
    python_version: str
    numpy_version: str
    gcc_version: str
    platform: str
    cpu_count: int
    git_commit: str
    purce_version: str

    def to_dict(self) -> dict:
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "gcc_version": self.gcc_version,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "git_commit": self.git_commit,
            "purce_version": self.purce_version,
        }


def _gcc_version() -> str:
    gcc = shutil.which("gcc")
    if not gcc:
        return "not-found"
    try:
        result = subprocess.run(
            [gcc, "--version"], capture_output=True, text=True, timeout=15,
        )
        return result.stdout.splitlines()[0] if result.stdout else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def get_fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        gcc_version=_gcc_version(),
        platform=platform.platform(),
        cpu_count=os.cpu_count() or 1,
        git_commit=_git_commit(),
        purce_version=__version__,
    )


# ── Extraction ───────────────────────────────────────────────────────────────


@dataclass
class ExtractionStats:
    corpus_dir: str
    py_files: int
    c_files: int
    h_files: int
    prov_files: int
    error_directives: int
    unresolved_files: int
    total_c_lines: int


def run_extraction(corpus_dir: Path, out_dir: Path) -> ExtractionStats:
    """Run the purce extract pipeline into out_dir and collect stats."""
    cmd = [
        sys.executable, "-m", "purce.cli", "extract",
        str(corpus_dir), "-o", str(out_dir),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, cwd=str(_PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"purce extract failed for {corpus_dir}:\n{result.stdout}\n{result.stderr}"
        )

    c_files = sorted(out_dir.glob("*.c"))
    prov_files = sorted(out_dir.glob("*.prov.json"))
    h_files = sorted(out_dir.glob("*.h"))

    error_directives = 0
    unresolved_files = 0
    total_c_lines = 0
    for c in c_files:
        text = c.read_text(encoding="utf-8")
        if "#error" in text:
            error_directives += 1
        if "_unresolved_" in text:
            unresolved_files += 1
        total_c_lines += len([ln for ln in text.splitlines() if ln.strip()])

    return ExtractionStats(
        corpus_dir=str(corpus_dir),
        py_files=len(list(corpus_dir.glob("*.py"))),
        c_files=len(c_files),
        h_files=len(h_files),
        prov_files=len(prov_files),
        error_directives=error_directives,
        unresolved_files=unresolved_files,
        total_c_lines=total_c_lines,
    )


# ── Content hash ─────────────────────────────────────────────────────────────


def _normalize_content(text: str) -> str:
    """Strip volatile timestamps so hashing is stable across runs."""
    for pattern in _VOLATILE_PATTERNS:
        text = pattern.sub("", text)
    return text


def content_hash(out_dir: Path) -> str:
    """SHA-256 over sorted generated files with volatile fields normalized."""
    hasher = hashlib.sha256()
    c_files = sorted(out_dir.glob("*.c"))
    prov_files = sorted(out_dir.glob("*.prov.json"))
    for f in c_files + prov_files:
        rel = str(f.relative_to(out_dir))
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(_normalize_content(f.read_text(encoding="utf-8")).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


# ── Compile gate ─────────────────────────────────────────────────────────────


@dataclass
class CompileStats:
    total: int = 0
    passed: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    def all_passed(self) -> bool:
        return self.failed == 0


def _compile_one(args: tuple[str, str]) -> tuple[str, str, bool, str]:
    c_path, gcc = args
    out_path = c_path + ".o"
    try:
        result = subprocess.run(
            [gcc, *GCC_FLAGS, c_path, "-o", out_path, "-lm"],
            capture_output=True, text=True, timeout=120,
        )
        ok = result.returncode == 0
        first_error = ""
        if not ok:
            for line in result.stderr.splitlines():
                if "error:" in line:
                    first_error = line.strip()
                    break
            if not first_error:
                first_error = result.stderr.splitlines()[0] if result.stderr else "unknown"
        return c_path, gcc, ok, first_error
    except subprocess.TimeoutExpired:
        return c_path, gcc, False, "Compilation timed out"
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def run_compile_gate(out_dir: Path, jobs: int) -> CompileStats:
    gcc = shutil.which("gcc")
    if not gcc:
        raise RuntimeError("gcc not found; cannot run compile gate")
    c_files = sorted(out_dir.glob("*.c"))
    stats = CompileStats(total=len(c_files))
    if not c_files:
        return stats
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_compile_one, (str(f), gcc)) for f in c_files]
        for fut in concurrent.futures.as_completed(futures):
            _, _, ok, first_error = fut.result()
            if ok:
                stats.passed += 1
            else:
                stats.failed += 1
                # failures collected after in deterministic order
    if stats.failed:
        results = [f.result() for f in futures]
        results.sort(key=lambda r: r[0])
        stats.failures = [(Path(r[0]).name, r[3]) for r in results if not r[2]]
    return stats


# ── Report ───────────────────────────────────────────────────────────────────


@dataclass
class HarnessReport:
    seed: int
    fingerprint: EnvironmentFingerprint
    corpora: list[ExtractionStats]
    compile_total: int
    compile_passed: int
    compile_failed: int
    compile_failures: list[tuple[str, str]]
    content_sha256: str
    total_c_files: int
    total_py_files: int
    timestamp_utc: str

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "fingerprint": self.fingerprint.to_dict(),
            "corpora": [c.__dict__ for c in self.corpora],
            "compile": {
                "total": self.compile_total,
                "passed": self.compile_passed,
                "failed": self.compile_failed,
                "failures": [{"file": f, "error": e} for f, e in self.compile_failures],
            },
            "content_sha256": self.content_sha256,
            "totals": {
                "c_files": self.total_c_files,
                "py_files": self.total_py_files,
            },
            "timestamp_utc": self.timestamp_utc,
        }

    def as_markdown(self) -> str:
        lines = [
            "# Purce Reproducible Benchmark Report",
            "",
            f"- Seed: `{self.seed}`",
            f"- Content SHA-256: `{self.content_sha256}`",
            f"- Generated: {self.timestamp_utc} (UTC)",
            "",
            "## Environment",
            "",
        ]
        fp = self.fingerprint
        lines.append("| Component | Version |")
        lines.append("|-----------|---------|")
        lines.append(f"| Python | {fp.python_version} |")
        lines.append(f"| NumPy | {fp.numpy_version} |")
        lines.append(f"| GCC | {fp.gcc_version} |")
        lines.append(f"| Platform | {fp.platform} |")
        lines.append(f"| CPU cores | {fp.cpu_count} |")
        lines.append(f"| Git commit | {fp.git_commit} |")
        lines.append(f"| Purce | {fp.purce_version} |")
        lines.append("")
        lines.append("## Extraction")
        lines.append("")
        lines.append("| Corpus | Py files | C files | Prov | Hdr | #error | unresolved | C lines |")
        lines.append("|--------|----------|---------|------|-----|--------|------------|---------|")
        for c in self.corpora:
            lines.append(
                f"| {c.corpus_dir} | {c.py_files} | {c.c_files} | {c.prov_files} "
                f"| {c.h_files} | {c.error_directives} | {c.unresolved_files} | {c.total_c_lines} |"
            )
        lines.append("")
        lines.append("## Compile Gate (strict gcc)")
        lines.append("")
        lines.append(f"- Total files: `{self.compile_total}`")
        lines.append(f"- Passed: `{self.compile_passed}`")
        lines.append(f"- Failed: `{self.compile_failed}`")
        lines.append("")
        if self.compile_failures:
            lines.append("### Failures")
            lines.append("")
            for name, err in self.compile_failures:
                lines.append(f"- `{name}`: {err}")
            lines.append("")
        lines.append(f"**Gate: {'PASS' if self.compile_failed == 0 else 'FAIL'}**")
        lines.append("")
        return "\n".join(lines)


# ── Baseline ─────────────────────────────────────────────────────────────────


def load_baseline(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_against_baseline(report: HarnessReport, baseline: dict) -> list[str]:
    """Return a list of deviations; empty list means reproducible."""
    deviations: list[str] = []
    comp = baseline.get("compile", {})
    if comp.get("total") != report.compile_total:
        deviations.append(
            f"compile.total {baseline.get('compile', {}).get('total')} != {report.compile_total}"
        )
    if comp.get("failed") != report.compile_failed:
        deviations.append(
            f"compile.failed {comp.get('failed')} != {report.compile_failed}"
        )
    if baseline.get("content_sha256") != report.content_sha256:
        deviations.append(
            f"content_sha256 {baseline.get('content_sha256')} != {report.content_sha256}"
        )
    totals = baseline.get("totals", {})
    if totals.get("c_files") != report.total_c_files:
        deviations.append(
            f"totals.c_files {totals.get('c_files')} != {report.total_c_files}"
        )
    baseline_corpora = baseline.get("corpora", [])
    if len(baseline_corpora) != len(report.corpora):
        deviations.append("corpus count differs")
    else:
        for b, r in zip(baseline_corpora, report.corpora):
            if b.get("c_files") != r.c_files:
                deviations.append(f"corpus {r.corpus_dir}: c_files {b.get('c_files')} != {r.c_files}")
            if b.get("error_directives") != r.error_directives:
                deviations.append(
                    f"corpus {r.corpus_dir}: #error {b.get('error_directives')} != {r.error_directives}"
                )
    return deviations


# ── Orchestration ────────────────────────────────────────────────────────────


def run_harness(
    corpus_dirs: list[str],
    jobs: int,
    seed: int,
    baseline_path: Path | None = None,
    keep_output: bool = False,
) -> int:
    """Run the full harness. Returns process exit code (0 = reproducible/pass)."""
    import random
    random.seed(seed)

    fingerprint = get_fingerprint()
    output_root = _PROJECT_ROOT / ".benchmarks"
    output_root.mkdir(parents=True, exist_ok=True)

    corpora: list[ExtractionStats] = []
    total_c = 0
    total_py = 0
    combined_out: Path | None = None

    for corpus_dir in corpus_dirs:
        corpus_path = Path(corpus_dir)
        if not corpus_path.is_absolute():
            corpus_path = _PROJECT_ROOT / corpus_path
        if not corpus_path.is_dir():
            raise RuntimeError(f"corpus directory not found: {corpus_path}")

        out_dir = Path(tempfile.mkdtemp(prefix="purce_repro_"))
        try:
            stats = run_extraction(corpus_path, out_dir)
        except RuntimeError:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise
        corpora.append(stats)
        total_c += stats.c_files
        total_py += stats.py_files

        if combined_out is None:
            combined_out = out_dir
        else:
            for f in out_dir.glob("*.c"):
                shutil.copy2(f, combined_out)
            for f in out_dir.glob("*.prov.json"):
                shutil.copy2(f, combined_out)
            for f in out_dir.glob("*.h"):
                shutil.copy2(f, combined_out)
            shutil.rmtree(out_dir, ignore_errors=True)

    if combined_out is None:
        raise RuntimeError("no corpora to process")

    try:
        compile_stats = run_compile_gate(combined_out, jobs)
    except RuntimeError:
        if not keep_output:
            shutil.rmtree(combined_out, ignore_errors=True)
        raise

    digest = content_hash(combined_out)

    report = HarnessReport(
        seed=seed,
        fingerprint=fingerprint,
        corpora=corpora,
        compile_total=compile_stats.total,
        compile_passed=compile_stats.passed,
        compile_failed=compile_stats.failed,
        compile_failures=compile_stats.failures,
        content_sha256=digest,
        total_c_files=total_c,
        total_py_files=total_py,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8",
    )
    md_path = output_root / "report.md"
    md_path.write_text(report.as_markdown() + "\n", encoding="utf-8")
    print(f"report.json -> {report_path}")
    print(f"report.md   -> {md_path}")

    if keep_output:
        final_out = output_root / "out"
        if final_out.exists():
            shutil.rmtree(final_out, ignore_errors=True)
        shutil.copytree(combined_out, final_out)
        print(f"generated C  -> {final_out}")
    else:
        shutil.rmtree(combined_out, ignore_errors=True)

    if baseline_path is not None:
        baseline = load_baseline(baseline_path)
        deviations = verify_against_baseline(report, baseline)
        if deviations:
            print("\nBASELINE MISMATCH:")
            for d in deviations:
                print(f"  - {d}")
            print("Reproducibility check FAILED.")
            return 1
        print("\nReproducibility check PASSED (matches baseline).")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="purce-repro",
        description="Run the reproducible Purce benchmark harness.",
    )
    parser.add_argument(
        "--corpora", nargs="*", default=DEFAULT_CORPORA,
        help="Corpus directories (default: %(default)s)",
    )
    parser.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4),
                        help="Parallel gcc jobs")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Baseline JSON to verify against")
    parser.add_argument("--keep-output", action="store_true",
                        help="Keep generated C in .benchmarks/out")
    args = parser.parse_args(argv)
    return run_harness(
        corpus_dirs=args.corpora,
        jobs=args.jobs,
        seed=args.seed,
        baseline_path=args.baseline,
        keep_output=args.keep_output,
    )


if __name__ == "__main__":
    sys.exit(main())
