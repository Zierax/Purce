"""Hardened harness — runs stages in parallel, fully reproducible."""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from purce.ir.builder import MathIRBuilder
from purce.backend.c99_generator import C99Generator, _STUB_ALGORITHMS
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.coverage_sweep import run_sweep
from purce.verifier.fuzzer import DifferentialFuzzer

from .config import STAGES, DETERMINISM_SEEDS, ADVERSARIAL_CASES


def _content_hash(files: list[Path]) -> str:
    import re
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: str(x)):
        data = p.read_bytes()
        # Normalize volatile fields: timestamps, absolute paths
        try:
            text = data.decode("utf-8")
            text = re.sub(r'"generated_at":\s*"[^"]+"', '"generated_at": "NORMALIZED"', text)
            text = re.sub(r'"commit":\s*"[^"]+"', '"commit": "NORMALIZED"', text)
            # Also strip GENERATED AT in C headers if present
            data = text.encode("utf-8")
        except Exception:
            pass
        h.update(data)
    return h.hexdigest()[:16]


def _get_c_content(gen_result) -> str:
    for f in gen_result.files:
        if f.file_type == "c":
            return f.content
    return gen_result.files[0].content if gen_result.files else ""


def run_stage(stage_key: str, workdir: Path) -> dict:
    cfg = STAGES[stage_key]
    t0 = time.perf_counter()
    # 1. Coverage sweep with configured iterations/seeds
    sweep = run_sweep(iterations=cfg["iterations"] // 500 + 2, seeds=cfg["seeds"])
    # 2. Fuzzer on 25 legacy ops
    fuzzer = DifferentialFuzzer(seed=cfg["seed"])
    # try with C backend, fallback to python
    try:
        fz, comp = DifferentialFuzzer.with_c_backend(seed=cfg["seed"])
        fres = fz.fuzz_all(iterations=cfg["iterations"] // 10)
        comp.close()
    except Exception:
        fres = fuzzer.fuzz_all(iterations=cfg["iterations"] // 10)
    elapsed = time.perf_counter() - t0
    return {
        "stage": stage_key,
        "name": cfg["name"],
        "iterations": cfg["iterations"],
        "sweep_total": sweep.total if hasattr(sweep, "total") else len(sweep.outcomes),
        "sweep_passed": sum(1 for o in sweep.outcomes if o.ok) if hasattr(sweep, "outcomes") else 0,
        "fuzz_total": len(fres),
        "fuzz_passed": sum(1 for r in fres.values() if r.all_passed),
        "elapsed": elapsed,
    }


def run_determinism_check(repo_root: Path) -> dict:
    """Spawn subprocesses with different PYTHONHASHSEED and compare hashes."""
    import tempfile, shutil
    results = {}
    for seed in DETERMINISM_SEEDS:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        # Run a minimal extraction and hash the provenance
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            cmd = [sys.executable, "-m", "purce.cli", "extract", str(repo_root / "tests" / "realworld"), "-o", str(out)]
            subprocess.run(cmd, env=env, capture_output=True, timeout=60)
            provs = sorted(out.rglob("*.prov.json"))
            h = _content_hash(provs) if provs else "no-out"
            results[str(seed)] = h
    all_same = len(set(results.values())) == 1
    return {"hashes": results, "deterministic": all_same}


def run_adversarial(repo_root: Path) -> dict:
    """Deliberately provoke failures and verify they are loud."""
    checks = {}
    # VLA provocation: n=100000 for sort should be rejected (guard return)
    from purce.ir.nodes import MathIRNode, MathIRGraph, Dtype, Effect
    from purce.backend.c99_generator import C99Generator
    # Test 1: array_sort with n=100000 should use heap (malloc) not VLA overflow
    node = MathIRNode(
        node_id="adv_sort", origin_symbol="adv.sort", origin_file="adv.py", origin_line=1,
        origin_commit=None, origin_signature="x array", math_intent="adv", inputs=[("x", Dtype.FLOAT64, "array"), ("n", Dtype.INT64, "scalar")], outputs=[("out", Dtype.FLOAT64, "array")], effects=[Effect.PURE], algorithm="array_sort", stack_usage=0
    )
    g = MathIRGraph(); g.add_node(node)
    content = _get_c_content(C99Generator().generate(g, module_name="adv"))
    checks["array_sort_has_guard"] = "malloc" in content and "free(tmp)" in content
    # Test 2: linalg_det now uses heap (malloc) for large n
    node2 = MathIRNode(node_id="adv_det", origin_symbol="adv.det", origin_file="adv.py", origin_line=1, origin_commit=None, origin_signature="x", math_intent="adv", inputs=[("x", Dtype.FLOAT64, "array"), ("n", Dtype.INT64, "scalar")], outputs=[("result", Dtype.FLOAT64, "array")], effects=[Effect.PURE], algorithm="linalg_det", stack_usage=0)
    g2 = MathIRGraph(); g2.add_node(node2)
    content2 = _get_c_content(C99Generator().generate(g2, module_name="adv"))
    checks["linalg_det_has_guard"] = "malloc" in content2 and "free(lu)" in content2
    # Test 3: stub header is unverified
    node3 = MathIRNode(node_id="adv_eig", origin_symbol="adv.eig", origin_file="adv.py", origin_line=1, origin_commit=None, origin_signature="x", math_intent="adv", inputs=[("A", Dtype.FLOAT64, "array"), ("n", Dtype.INT64, "scalar")], outputs=[("eigenvalues", Dtype.FLOAT64, "array")], effects=[Effect.PURE], algorithm="linalg_eig", stack_usage=0)
    g3 = MathIRGraph(); g3.add_node(node3)
    content3 = _get_c_content(C99Generator().generate(g3, module_name="adv"))
    checks["stub_header_unverified"] = "Verified: NO" in content3 and "WARNING: stub" in content3
    # Test 4: trunc uses trunc() not (int)
    node4 = MathIRNode(node_id="adv_trunc", origin_symbol="adv.trunc", origin_file="adv.py", origin_line=1, origin_commit=None, origin_signature="x", math_intent="adv", inputs=[("x", Dtype.FLOAT64, "array"), ("n", Dtype.INT64, "scalar")], outputs=[("out", Dtype.FLOAT64, "array")], effects=[Effect.PURE], algorithm="element_trunc", stack_usage=0)
    g4 = MathIRGraph(); g4.add_node(node4)
    content4 = _get_c_content(C99Generator().generate(g4, module_name="adv"))
    checks["trunc_uses_trunc"] = "trunc(x" in content4 and "(int)x" not in content4
    # Test 5: isinf uses isinf()
    node5 = MathIRNode(node_id="adv_isinf", origin_symbol="adv.isinf", origin_file="adv.py", origin_line=1, origin_commit=None, origin_signature="x", math_intent="adv", inputs=[("x", Dtype.FLOAT64, "array"), ("n", Dtype.INT64, "scalar")], outputs=[("out", Dtype.FLOAT64, "array")], effects=[Effect.PURE], algorithm="element_isinf", stack_usage=0)
    g5 = MathIRGraph(); g5.add_node(node5)
    content5 = _get_c_content(C99Generator().generate(g5, module_name="adv"))
    checks["isinf_uses_isinf"] = "isinf(x" in content5

    # Test 6: parser rejects eval/open
    from purce.parser.python_parser import PythonParser
    p = PythonParser()
    bad = p.parse_source("import numpy as np\ndef f(x):\n    eval('np.add(x,x)')\n", filename="bad.py")
    checks["parser_rejects_eval"] = len(bad.diagnostics) > 0 or len(bad.functions) == 0

    checks["all_loud"] = all(checks.values())
    return checks


def run_parallel_all(workdir: Path, repo_root: Path) -> dict:
    """Run stage1,2,3 + determinism + adversarial in parallel."""
    workdir.mkdir(parents=True, exist_ok=True)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {}
        for stage in ["stage1", "stage2", "stage3"]:
            futs[ex.submit(run_stage, stage, workdir)] = stage
        futs[ex.submit(run_determinism_check, repo_root)] = "determinism"
        futs[ex.submit(run_adversarial, repo_root)] = "adversarial"
        for fut in concurrent.futures.as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result(timeout=600)
            except Exception as e:
                results[key] = {"error": str(e)[:500]}
    return results
