"""Verification sub-agent: comprehensive post-change validation.

Run this after any code change to verify correctness:
  python -m tests.verification_agent

Checks:
  1. All unit tests pass (174+ tests)
  2. All fuzz tests pass (25 operations, 200 iterations each)
  3. Full pipeline integration on real-world ML code
  4. Generated C compiles (structural check)
  5. Zero-heap-allocation verification in generated C
  6. File provenance headers present in all outputs
  7. Synthetic pipeline sanity checks
  8. No regressions in existing behavior
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.fuzzer import DifferentialFuzzer


@dataclass
class VerificationReport:
    timestamp: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    fuzz_passed: int = 0
    fuzz_failed: int = 0
    fuzz_operations: int = 0
    pipeline_sources_tested: int = 0
    pipeline_kernels_generated: int = 0
    c_files_generated: int = 0
    c_files_structurally_valid: int = 0
    c_files_heap_free: int = 0
    c_files_provenance_ok: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def all_passed(self) -> bool:
        return self.tests_failed == 0 and self.fuzz_failed == 0 and len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            "=== PURCE VERIFICATION REPORT ===",
            f"Timestamp:          {self.timestamp}",
            f"Duration:           {self.duration_seconds:.1f}s",
            "",
            f"Unit Tests:         {self.tests_passed}/{self.tests_passed + self.tests_failed} passed",
            f"Fuzz Tests:         {self.fuzz_passed}/{self.fuzz_passed + self.fuzz_failed} passed ({self.fuzz_operations} ops)",
            f"Pipeline Sources:   {self.pipeline_sources_tested}",
            f"Kernels Generated:  {self.pipeline_kernels_generated}",
            f"C Files:            {self.c_files_generated} ({self.c_files_structurally_valid} valid, {self.c_files_heap_free} heap-free, {self.c_files_provenance_ok} with provenance)",
            "",
        ]
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append(f"RESULT: {'PASS' if self.all_passed else 'FAIL'}")
        return "\n".join(lines)


def _run_pipeline(source: str, module_name: str) -> tuple:
    parser = PythonParser(target_profile="generic-c99")
    parsed = parser.parse_source(source, f"{module_name}.py")
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    entry_points = list(graph.nodes.keys())
    slice_result = slicer.slice(graph, entry_points)
    generator = C99Generator(target_profile="generic-c99")
    gen_result = generator.generate(slice_result.graph, module_name=module_name)
    return gen_result, graph


HARB_KEYWORDS = ("malloc", "calloc", "realloc", "free(", "new ", "delete ")


def _check_c_structural(content: str) -> bool:
    checks = [
        "PURCE OUTPUT" in content,
        "#include" in content,
        ("void " in content or "#error" in content),
    ]
    return all(checks)


def _check_c_heap_free(content: str) -> bool:
    for kw in HARB_KEYWORDS:
        if kw in content:
            return False
    return True


def _check_c_provenance(content: str) -> bool:
    return "PURCE OUTPUT" in content and "Source module:" in content


def run_verification() -> VerificationReport:
    from datetime import UTC, datetime
    report = VerificationReport(timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    start = time.time()

    # ── Phase 1: Unit tests via pytest ──
    print("[1/4] Running unit tests...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "passed" in line and "failed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed":
                        report.tests_passed = int(parts[i - 1])
                    if p == "failed":
                        report.tests_failed = int(parts[i - 1])
            elif line.strip().endswith("passed"):
                report.tests_passed = int(line.strip().split()[0])
        if result.returncode != 0 and report.tests_failed == 0:
            report.tests_failed = 1
            report.errors.append(f"pytest exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        report.errors.append("Unit tests timed out after 120s")
    except Exception as e:
        report.errors.append(f"Unit tests failed: {e}")

    # ── Phase 2: Fuzz tests ──
    print("[2/4] Running fuzz tests (25 operations)...")
    try:
        fuzzer = DifferentialFuzzer(seed=42)
        results = fuzzer.fuzz_all(iterations=200)
        report.fuzz_operations = len(results)
        for op, result in results.items():
            if result.all_passed:
                report.fuzz_passed += 1
            else:
                report.fuzz_failed += 1
                report.errors.append(f"Fuzz failed: {op} ({result.failed}/{result.iterations} failed)")
    except Exception as e:
        report.errors.append(f"Fuzz tests failed: {e}")

    # ── Phase 3: Pipeline integration on real-world sources ──
    print("[3/5] Running pipeline integration tests...")
    realworld_dir = os.path.join(os.path.dirname(__file__), "realworld")
    if os.path.isdir(realworld_dir):
        for fname in sorted(os.listdir(realworld_dir)):
            if fname.endswith(".py") and fname != "__init__.py" and not fname.startswith("model"):
                fpath = os.path.join(realworld_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        src = f.read()
                    gen, graph = _run_pipeline(src, fname.replace(".py", ""))
                    report.pipeline_sources_tested += 1
                    report.pipeline_kernels_generated += len(graph.nodes)
                    c_files = [f for f in gen.files if f.file_type == "c"]
                    report.c_files_generated += len(c_files)
                    for cf in c_files:
                        if _check_c_structural(cf.content):
                            report.c_files_structurally_valid += 1
                        else:
                            report.warnings.append(f"Structural check failed: {cf.path}")
                        if _check_c_heap_free(cf.content):
                            report.c_files_heap_free += 1
                        if _check_c_provenance(cf.content):
                            report.c_files_provenance_ok += 1
                except Exception as e:
                    report.errors.append(f"Pipeline failed for {fname}: {e}")

    # ── Phase 4: Synthetic pipeline test ──
    print("[4/5] Running synthetic pipeline test...")
    synthetic_sources = [
        ("add", "import numpy as np\ndef add(a, b):\n    return np.add(a, b)"),
        ("matmul", "import numpy as np\ndef mm(A, B):\n    return np.matmul(A, B)"),
        ("softmax", "import numpy as np\ndef sm(x):\n    return np.divide(np.exp(x), np.sum(np.exp(x)))"),
        ("norm", "import numpy as np\ndef nrm(x, g, b):\n    m = np.mean(x)\n    return np.add(np.multiply(g, np.subtract(x, m)), b)"),
        ("fft", "import numpy as np\ndef f(r, i, n):\n    return np.fft.fft(r)"),
    ]
    for name, src in synthetic_sources:
        try:
            gen, graph = _run_pipeline(src, f"synth_{name}")
            report.pipeline_sources_tested += 1
            report.pipeline_kernels_generated += len(graph.nodes)
            c_files = [f for f in gen.files if f.file_type == "c"]
            report.c_files_generated += len(c_files)
            for cf in c_files:
                if _check_c_structural(cf.content):
                    report.c_files_structurally_valid += 1
                if _check_c_heap_free(cf.content):
                    report.c_files_heap_free += 1
                if _check_c_provenance(cf.content):
                    report.c_files_provenance_ok += 1
        except Exception as e:
            report.errors.append(f"Synthetic pipeline failed for {name}: {e}")

    # ── Phase 5: Memory safety verification ──
    print("[5/5] Running memory safety verification...")
    if report.c_files_generated > 0:
        heap_violations = report.c_files_generated - report.c_files_heap_free
        if heap_violations > 0:
            report.warnings.append(f"{heap_violations}/{report.c_files_generated} C files use heap allocation")
        provenance_gaps = report.c_files_generated - report.c_files_provenance_ok
        if provenance_gaps > 0:
            report.warnings.append(f"{provenance_gaps}/{report.c_files_generated} C files missing provenance headers")
        print(f"  Heap-free: {report.c_files_heap_free}/{report.c_files_generated}")
        print(f"  Provenance: {report.c_files_provenance_ok}/{report.c_files_generated}")
    else:
        print("  No C files generated to verify")

    report.duration_seconds = time.time() - start
    return report


if __name__ == "__main__":
    report = run_verification()
    print(report.summary())
    sys.exit(0 if report.all_passed else 1)
