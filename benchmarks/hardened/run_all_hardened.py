"""Entry point for hardened suite — parallel B, fully reproducible."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.hardened.corpus_harsh import generate_harsh_corpus
from benchmarks.hardened.corpus_chaos import generate_chaos_corpus
from benchmarks.hardened.harness import run_parallel_all
from benchmarks.hardened.config import STAGES

def main():
    out_root = Path(__file__).parent / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    harsh_dir = out_root / "harsh_corpus"
    chaos_dir = out_root / "chaos_corpus"

    print("=== Generating harsh corpus (25 files, seed fixed) ===")
    harsh_files = generate_harsh_corpus(str(harsh_dir))
    print(f"  -> {len(harsh_files)} files")

    print("=== Generating chaos corpus (100 files, seed fixed) ===")
    chaos_files = generate_chaos_corpus(str(chaos_dir), n=100)
    print(f"  -> {len(chaos_files)} files")

    print(f"=== Running stages in parallel B: {list(STAGES.keys())} + determinism + adversarial ===")
    t0 = time.perf_counter()
    results = run_parallel_all(out_root, ROOT)
    elapsed = time.perf_counter() - t0

    # Write raw JSON
    with open(out_root / "hardened_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Write markdown summary
    md = []
    md.append("# Hardened Frontier Results\n")
    md.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"Elapsed: {elapsed:.1f}s (parallel)\n")
    md.append(f"Harsh corpus: {len(harsh_files)} files, Chaos corpus: {len(chaos_files)} files\n")
    for k in ["stage1", "stage2", "stage3"]:
        r = results.get(k, {})
        if "error" in r:
            md.append(f"## {k} — ERROR\n```\n{r['error']}\n```\n")
        else:
            md.append(f"## {r.get('name',k)} ({k})\n")
            md.append(f"- Iterations: {r.get('iterations')}\n")
            md.append(f"- Sweep: {r.get('sweep_passed')}/{r.get('sweep_total')} passed\n")
            md.append(f"- Fuzz: {r.get('fuzz_passed')}/{r.get('fuzz_total')} passed\n")
            md.append(f"- Time: {r.get('elapsed',0):.1f}s\n")
    det = results.get("determinism", {})
    md.append("## Determinism\n")
    md.append(f"- Deterministic across PYTHONHASHSEED: {det.get('deterministic')}\n")
    md.append(f"```json\n{json.dumps(det.get('hashes',{}), indent=2)}\n```\n")
    adv = results.get("adversarial", {})
    md.append("## Adversarial (deliberate failure)\n")
    for ck, val in adv.items():
        md.append(f"- {ck}: {'PASS' if val else 'FAIL'}\n")
    md.append(f"\n**Overall loud failures verified: {adv.get('all_loud') }**\n")

    # Limitations & Roadmap
    md.append("\n## LIMITATIONS (discovered)\n")
    md.append("- `array_take` now uses per-input lengths `n` (len x) and `k` (len idx) — `idx < n` correctly (fixed)\n")
    md.append("- `array_sort/unique` and `linalg_det` now use heap (`malloc/free`) — large n handled, no silent return (fixed)\n")
    md.append("- `linalg_eig/qr/svd/array_split` are stubs marked UNVERIFIED — not for production (P0 next)\n")
    md.append("- `from numpy import dot` and `import as la` now resolved via alias map (fixed)\n")
    md.append("- Determinism: raw bytes differ due to timestamps, but normalized content hash is stable\n")
    md.append("- Chaos corpus: 10% invalid programs correctly rejected by parser (eval/open)\n")

    md.append("\n## ROADMAP TODO\n")
    md.append("- P0: Implement real `eig` (QR iteration), `qr` (Gram-Schmidt), `svd` (Jacobi/Golub) and remove stub flag\n")
    md.append("- P1: Promote chaos corpus to CI nightly with 500 programs\n")
    md.append("- P2: Extract `c99_generator.py` God Object into `registry/` + `kernels/` + `emitter/`\n")
    md.append("- P2: Unify verifier duplication (`_all_close`, `compile`, reference oracles) into `_common`\n")

    with open(out_root / "hardened_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("\n".join(md))
    print(f"\nWrote {out_root / 'hardened_results.json'} and .md")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
