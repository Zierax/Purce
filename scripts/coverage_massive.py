"""Massive-scale differential sweep: verify every reachable kernel across
thousands of randomized cases per algorithm and several independent seed
sequences, then write a consolidated JSON report to .benchmarks/."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from purce.verifier import coverage_sweep as cs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".benchmarks" / "coverage_massive.json"

ITERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
START_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

seeds = tuple(range(START_SEED, START_SEED + N_SECONDS))
t0 = time.perf_counter()
report = cs.run_sweep(iterations=ITERATIONS, seeds=seeds)
elapsed = time.perf_counter() - t0

rows = []
for o in report.outcomes:
    rows.append({
        "algorithm": o.algorithm,
        "mode": o.mode,
        "passed": o.passed,
        "failed": o.failed,
        "max_error": o.max_error,
        "errors": o.errors[:5],
    })

summary = {
    "algorithms": len(report.outcomes),
    "iterations_per_seed": ITERATIONS,
    "seeds": list(seeds),
    "cases_per_algorithm": ITERATIONS * len(seeds),
    "total_cases": len(report.outcomes) * ITERATIONS * len(seeds),
    "all_passed": report.all_passed,
    "elapsed_seconds": round(elapsed, 1),
    "results": rows,
}
OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(f"massive sweep: {sum(1 for r in rows if not r['failed'])}/{len(rows)} algorithms ok")
print(f"  iterations={ITERATIONS} seeds={seeds}")
print(f"  total_cases={summary['total_cases']} elapsed={elapsed:.1f}s")
print(f"  report -> {OUT.name}")

worst = sorted((r for r in rows if r["mode"] == "reference"),
               key=lambda r: r["max_error"], reverse=True)[:12]
print("  worst reference max_err:")
for r in worst:
    print(f"    {r['algorithm']:<22} {r['max_error']:.4g}")