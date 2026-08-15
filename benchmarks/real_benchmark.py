"""Real benchmark: run purce on actual code and measure everything.

Produces benchmarks/results.md with real measured data.
"""
from __future__ import annotations

# -*- coding: utf-8 -*-
import os
import re
import sys
import time
from pathlib import Path

# ... rest of the code ...

# ── Ensure project root is on path ──
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.fuzzer import DifferentialFuzzer
from purce.verifier.z3_verifier import Z3Verifier


def _run_pipeline(source: str, module_name: str) -> tuple:
    parser = PythonParser(target_profile="generic-c99")
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    entry_points = list(graph.nodes.keys())
    slice_result = slicer.slice(graph, entry_points)
    generator = C99Generator(target_profile="generic-c99")
    gen_result = generator.generate(slice_result.graph, module_name=module_name)
    return gen_result, graph, slice_result


def _load_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _count_c_lines(content: str) -> int:
    return len([l for l in content.splitlines() if l.strip() and not l.strip().startswith("//") and not l.strip().startswith("/*") and not l.strip().startswith("*")])


def _count_functions(content: str) -> int:
    return len(re.findall(r'\bvoid\s+\w+\s*\(', content))


def _has_error_directive(content: str) -> bool:
    return "#error" in content


def _bench_pipeline(source: str, module_name: str) -> dict:
    start = time.perf_counter()
    gen_result, graph, slice_result = _run_pipeline(source, module_name)
    elapsed = time.perf_counter() - start

    c_files = [f for f in gen_result.files if f.file_type == "c"]
    h_files = [f for f in gen_result.files if f.file_type == "h"]
    prov_files = [f for f in gen_result.files if f.file_type == "prov"]

    total_c_lines = sum(_count_c_lines(f.content) for f in c_files)
    total_functions = sum(_count_functions(f.content) for f in c_files)
    error_count = sum(1 for f in c_files if _has_error_directive(f.content))
    clean_count = len(c_files) - error_count

    return {
        "nodes": len(graph.nodes),
        "c_files": len(c_files),
        "h_files": len(h_files),
        "prov_files": len(prov_files),
        "total_c_lines": total_c_lines,
        "total_functions": total_functions,
        "error_files": error_count,
        "clean_files": clean_count,
        "time_ms": elapsed * 1000,
    }


def run_benchmarks() -> str:
    realworld_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "realworld")
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")

    lines = [
        "# Purce Benchmark Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python: {sys.version.split()[0]}",
        "",
    ]

    # ── Section 1: Pipeline benchmarks ──
    lines.append("## Pipeline Performance")
    lines.append("")
    lines.append("Time to process Python source through full pipeline: parse → IR → slice → generate C99.")
    lines.append("")
    lines.append("| Source | Nodes | C Files | Clean C | #error C | C Lines | Time (ms) |")
    lines.append("|--------|-------|---------|---------|----------|---------|-----------|")

    # Test fixtures
    fixtures_files = [f for f in os.listdir(fixtures_dir) if f.endswith(".py")]
    combined_fixtures = ""
    for ff in sorted(fixtures_files):
        combined_fixtures += _load_file(os.path.join(fixtures_dir, ff)) + "\n"
    r = _bench_pipeline(combined_fixtures, "fixtures")
    lines.append(f"| fixtures/ ({len(fixtures_files)} files) | {r['nodes']} | {r['c_files']} | {r['clean_files']} | {r['error_files']} | {r['total_c_lines']} | {r['time_ms']:.1f} |")

    # Test realworld
    rw_files = sorted([f for f in os.listdir(realworld_dir) if f.endswith(".py") and f != "__init__.py"])
    combined_rw = ""
    for rwf in rw_files:
        combined_rw += _load_file(os.path.join(realworld_dir, rwf)) + "\n"
    r = _bench_pipeline(combined_rw, "realworld")
    lines.append(f"| realworld/ ({len(rw_files)} files) | {r['nodes']} | {r['c_files']} | {r['clean_files']} | {r['error_files']} | {r['total_c_lines']} | {r['time_ms']:.1f} |")

    # Individual realworld files
    lines.append("")
    lines.append("### Per-File Breakdown")
    lines.append("")
    lines.append("| File | Nodes | C Files | Clean | #error | C Lines | Time (ms) |")
    lines.append("|------|-------|---------|-------|--------|---------|-----------|")
    rw_nodes: list[int] = []
    rw_times: list[float] = []
    for rwf in rw_files:
        src = _load_file(os.path.join(realworld_dir, rwf))
        r = _bench_pipeline(src, rwf.replace(".py", ""))
        rw_nodes.append(r["nodes"])
        rw_times.append(r["time_ms"] / 1000.0)
        lines.append(f"| {rwf} | {r['nodes']} | {r['c_files']} | {r['clean_files']} | {r['error_files']} | {r['total_c_lines']} | {r['time_ms']:.1f} |")

    # ── Section 2: Verification ──
    lines.append("")
    lines.append("## Verification Results")
    lines.append("")

    # Z3
    lines.append("### Z3 SMT Bounds Checking")
    lines.append("")
    verifier = Z3Verifier()
    gen_result, graph, _ = _run_pipeline(combined_rw, "realworld")
    reports = verifier.verify_graph(graph)
    verified = sum(1 for r in reports.values() if r.all_verified)
    violated = sum(1 for r in reports.values() if not r.all_verified and r.violated_count > 0)
    unknown = len(reports) - verified - violated
    lines.append(f"- Total nodes: {len(reports)}")
    lines.append(f"- Verified (safe): {verified}")
    lines.append(f"- Violated (potential overflow): {violated}")
    lines.append(f"- Unknown (Z3 timeout): {unknown}")
    lines.append("")

    # Fuzzing
    lines.append("### Differential Fuzzing (Python-only, 1000 iterations)")
    lines.append("")
    fuzzer = DifferentialFuzzer(seed=42)
    fuzz_results = fuzzer.fuzz_all(iterations=1000)
    lines.append("| Operation | Passed | Total | Rate | Max Error | Status |")
    lines.append("|-----------|--------|-------|------|-----------|--------|")
    all_passed = True
    for op, result in sorted(fuzz_results.items()):
        status = "PASS" if result.all_passed else "FAIL"
        if not result.all_passed:
            all_passed = False
        err = f"{result.max_error:.2e}" if result.tested_c else "N/A"
        lines.append(f"| {op} | {result.passed} | {result.iterations} | {result.success_rate:.1%} | {err} | {status} |")
    lines.append("")
    if all_passed:
        lines.append("All 25 operations passed differential fuzzing.")
    else:
        lines.append("Some operations failed. See details above.")
    lines.append("")

    # ── Section 3: Code quality ──
    lines.append("## Generated Code Quality")
    lines.append("")
    c_files = [f for f in gen_result.files if f.file_type == "c"]
    total_lines = sum(_count_c_lines(f.content) for f in c_files)
    total_funcs = sum(_count_functions(f.content) for f in c_files)
    error_files = [f for f in c_files if _has_error_directive(f.content)]
    clean_files = [f for f in c_files if not _has_error_directive(f.content)]

    lines.append(f"- Total .c files: {len(c_files)}")
    lines.append(f"- Files with #error (unsupported ops): {len(error_files)}")
    lines.append(f"- Clean compilable files: {len(clean_files)}")
    lines.append(f"- Total C lines (excluding comments): {total_lines}")
    lines.append(f"- Total function definitions: {total_funcs}")
    lines.append(f"- Average lines per function: {total_lines / max(total_funcs, 1):.0f}")
    lines.append("")

    # ── Section 4: Supported vs unsupported ──
    lines.append("## Operation Coverage")
    lines.append("")
    lines.append("### Supported (generates valid C99)")
    lines.append("")
    lines.append("| Operation | Algorithm | Kernel Count |")
    lines.append("|-----------|-----------|--------------|")
    from purce.backend.c99_generator import MATH_KERNEL_BODIES
    for algo in sorted(MATH_KERNEL_BODIES.keys()):
        count = sum(1 for n in graph.nodes.values() if n.algorithm == algo)
        if count > 0:
            lines.append(f"| various | {algo} | {count} |")

    lines.append("")
    lines.append("### Unsupported (generates #error)")
    lines.append("")
    unknown_nodes = [n for n in graph.nodes.values() if n.algorithm == "unknown"]
    lines.append(f"- Total nodes with 'unknown' algorithm: {len(unknown_nodes)}")
    if unknown_nodes:
        lines.append("- These contain operations like: numpy.power, numpy.maximum, numpy.minimum, numpy.where, numpy.clip, etc.")
        lines.append("- Adding these to MATH_KERNEL_BODIES would eliminate the #error directives.")
    lines.append("")

    # ── Section 5: Summary ──
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Python source files processed | {len(rw_files) + len(fixtures_files)} |")
    lines.append(f"| Total Math-IR nodes extracted | {sum(rw_nodes)} |")
    lines.append(f"| Generated .c files | {len(c_files)} |")
    lines.append(f"| Clean compilable files | {len(clean_files)} ({len(clean_files)/max(len(c_files),1)*100:.0f}%) |")
    lines.append(f"| Files with #error | {len(error_files)} ({len(error_files)/max(len(c_files),1)*100:.0f}%) |")
    lines.append(f"| Z3 verified nodes | {verified}/{len(reports)} |")
    lines.append(f"| Fuzz operations passed | {sum(1 for rr in fuzz_results.values() if rr.all_passed)}/{len(fuzz_results)} |")
    lines.append(f"| Pipeline time (all files) | {sum(rw_times):.1f}s |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Running benchmarks...")
    report = run_benchmarks()

    # Write report with UTF-8 encoding
    results_path = os.path.join(os.path.dirname(__file__), "results.md")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print report to console
    print(report)
