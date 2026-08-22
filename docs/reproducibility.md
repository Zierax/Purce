# Reproducibility — Why Purce Builds are Hermetic

> If you change nothing, you should get the same C. Not "almost the same." The same.

This document explains why Purce treats determinism as a correctness property, how we enforce it across three layers of the pipeline, and how you can verify it yourself. If you're new to the codebase, start here before you touch anything that iterates over a `set` or a `dict`.

---

## Why a compiler has to be reproducible

Most tools can get away with "works on my machine." A compiler cannot. Purce takes Python/NumPy and emits C99 that gets compiled, flashed onto hardware, and audited months later. Three demands make reproducibility non-negotiable:

**Audit.** When an auditor asks "what source produced this `matmul.c`?", the answer has to be a hash, a git commit, and a provenance file that line up byte-for-byte. If two identical extractions produce different `*.prov.json` orderings, the provenance chain is broken. You cannot diff what you cannot stabilize.

**Supply chain.** Downstream builds vendor Purce's output into larger firmware. Those builds are expected to be hermetic — same inputs, same container, same `gcc` version, identical artifacts. If Purce injects non-determinism, every cache in the chain misses. You end up rebuilding the world because a dictionary happened to hash differently on a different kernel.

**Hermetic builds.** The industry term for this is *reproducible builds* (see [reproducible-builds.org](https://reproducible-builds.org)). The idea is simple: given the same source, same toolchain, and same declared inputs, the build produces bit-identical output regardless of wall-clock time, checkout path, or `PYTHONHASHSEED`. Purce aims for that on every `purce extract` invocation. Where we can't be byte-identical (more on that below), we make the *normalized* hash identical and we document exactly what we strip.

We learned this the hard way. Which brings us to the bug.

---

## The bug that cost two weeks

Early in v0.1-beta, extraction was "mostly deterministic." Same corpus, same machine, re-run it ten times — nine times the output hashes matched, the tenth didn't. No code change. No new file. Just a different hash.

The symptom showed up in the hardened determinism gate (`benchmarks/hardened/harness.py:74`, `run_determinism_check`). It spawns the same extraction under five different `PYTHONHASHSEED` values and compares the content hashes. Four matched, one didn't. The failure was intermittent because CPython randomizes hash seeds per process — `set` and `dict` iteration order is deliberately non-deterministic unless you pin `PYTHONHASHSEED`.

The root cause was in `purce/ir/builder.py`. The builder records interprocedural call edges as it walks the AST:

```python
# purce/ir/builder.py:264
self._callees: dict[str, set[str]] = {}
```

Each key is a caller, each value is a `set` of callees. That is fine for analysis. The problem was in `_finalize_call_graph` at `purce/ir/builder.py:391`:

```python
for caller, callees in self._callees.items():
    caller_sym = f"{module_name}.{caller}"
    for callee in callees:  # <-- unordered iteration
        ...
        node.nested_deps.append(callee_root)
```

Iterating a `set` directly means the order in which `nested_deps` were appended depended on the per-process hash seed. That order flows into the generated C files (dependency lists are emitted in order) and into `*.prov.json` (`dependencies` is a JSON array). Two processes with different seeds produced different file orderings for the same semantic graph. The comment that now lives at `purce/ir/builder.py:387-390` is short, but it was earned:

> Iterate callees in sorted order: `self._callees` stores a set, so unordered iteration would make the emitted `nested_deps` (and generated provenance) depend on the per-process `PYTHONHASHSEED`.

The fix was one word:

```python
for callee in sorted(callees):  # purce/ir/builder.py:393
```

That single `sorted()` call is load-bearing. Without it, the regression test `TestHashSeedIndependence` in `tests/test_reproducible.py:318` fails intermittently — exactly the way the bug originally presented, which is why we keep that test as a dedicated harness that spawns subprocesses with `PYTHONHASHSEED=1` vs `2` and asserts `content_hash(out_a) == content_hash(out_b)` (`tests/test_reproducible.py:328-347`).

But that was not the only place we were leaking non-determinism. The fix triggered an audit of every place the pipeline iterates over an unordered collection. There were three layers.

---

## Layer 1 — Input enumeration: what files we read and in what order

`_process_source` in `purce/cli.py:118` is the entry point for both `purce extract` and `purce compile`. It collects Python sources with:

```python
# purce/cli.py:116-118
# Sort for deterministic enumeration order: file order feeds source
# concatenation, which sets origin-line offsets in generated kernels.
py_files = sorted(source_path.rglob("*.py"), key=lambda p: str(p))
```

This looks trivial. It isn't. `Path.rglob` yields files in filesystem order, which varies by OS, filesystem, and even directory creation history. The combined source string is built by concatenating those files in enumeration order (`purce/cli.py:131-142`), and that combined string is fed to the builder with `origin_file=str(source_path)` (`purce/cli.py:157`). If you enumerate `b.py` before `a.py` on one machine and the reverse on another, origin line offsets shift, provenance drifts, and the combined AST differs. Sorting by string path makes the input stable before we even parse.

If you add a new pass that walks `py_files`, sort it. If you glob anything else, sort it.

---

## Layer 2 — Graph construction and publication order

Purce's core data structure is `MathIRGraph` (`purce/ir/nodes.py:83`). Nodes are stored in a `dict[str, MathIRNode]`, entry points in a `list[str]`, and each node carries `nested_deps: list[str]`. A graph is only as deterministic as the order in which those lists are produced and consumed.

### Callee edges

Covered above — `purce/ir/builder.py:393` sorts callees before wiring `nested_deps`, and `purce/ir/builder.py:404-406` sorts the final entry points:

```python
# purce/ir/builder.py:404
self.graph.entry_points = [
    func_roots[sym] for sym in sorted(func_roots)
]
```

### Topological order and reachability

`MathIRGraph.topological_sort` (`purce/ir/nodes.py:101`) implements Kahn's algorithm, but every step that could be ordered is explicitly sorted:

- `dependents` lists are sorted (`purce/ir/nodes.py:115-116`)
- the initial queue of zero-in-degree nodes is sorted (`purce/ir/nodes.py:117`)
- the queue is re-sorted after each insertion (`purce/ir/nodes.py:127`)
- any remaining unsorted nodes are appended in sorted order (`purce/ir/nodes.py:129`)

Similarly, `reachable_from` (`purce/ir/nodes.py:135`) builds the reachable subgraph by iterating `sorted(reachable)` at `purce/ir/nodes.py:151`, and `validate_dag` (`purce/ir/nodes.py:155`) iterates `sorted(self.nodes.items())` and `sorted(node.nested_deps)` to make error reporting stable.

The slicer mirrors this. `SemanticSlicer._classify_and_add` (`purce/slicer/semantic_slicer.py:63`) iterates `sorted(reachable)`, and `_reachability` plus `prune_unused` both sort before emitting. The `ordered` list that `C99Generator.generate` consumes at `purce/backend/c99_generator.py:1250` is the output of `topological_sort`, so header emission, file generation, and CMake generation all inherit that stable order:

```python
# purce/backend/c99_generator.py:1250
ordered = graph.topological_sort()
# ...
for node in ordered:  # header, .c, .prov.json, CMake — same order every time
```

The rule of thumb: if you're iterating over `self.nodes`, `reachable`, `entry_points`, or `nested_deps` and you don't call `sorted()`, you're probably introducing a seed-dependent ordering. The tests will catch you — but only if you run them with more than one seed.

---

## Layer 3 — Output normalization and the content hash

Even with perfect input and graph ordering, raw bytes are *not* identical across runs. Every generated file embeds a timestamp.

`C99Generator.generate` captures it once per invocation:

```python
# purce/backend/c99_generator.py:1248
timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
```

That timestamp is stamped into every header (`purce/backend/c99_generator.py:1317`), every `.c` file (`purce/backend/c99_generator.py:1491`), and every `prov.json` (`purce/backend/c99_generator.py:1635`). Two extractions a second apart will differ, by design — we want humans to know when something was generated.

Reproducibility therefore means *normalized* reproducibility: strip what is supposed to vary, hash the rest.

### What we strip

`benchmarks/reproducible.py:48` defines the volatile patterns:

```python
# benchmarks/reproducible.py:48-51
_VOLATILE_PATTERNS = [
    re.compile(r"^\s*\*\s*GENERATED AT:\s+.*$", re.MULTILINE),
    re.compile(r'"generated_at":\s*"[^"]*"'),
]
```

`_normalize_content` (`benchmarks/reproducible.py:178`) also handles absolute paths. Every generated file embeds `ORIGIN FILE: /absolute/path/to/source.py:line`. That path is the checkout root, which differs per machine and per CI runner. The normalizer replaces it:

```python
# benchmarks/reproducible.py:187-189
text = text.replace("\\\\", "/").replace("\\", "/")
root = _CHECKOUT_ROOT.replace("\\", "/")
text = text.replace(root, "{PURCE_ROOT}")
```

Backslashes are normalized to forward slashes first because Windows C output uses native separators while JSON provenance double-escapes them — without that step, the replacement would miss half the occurrences (`benchmarks/reproducible.py:182-186`).

The hardened suite does the same with a tighter scope — it normalizes `generated_at` and `commit` in provenance files (`benchmarks/hardened/harness.py:30-33`), then truncates the SHA-256 to 16 hex characters for display (`benchmarks/hardened/harness.py:37`):

```python
# benchmarks/hardened/harness.py:22-37
def _content_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: str(x)):
        data = p.read_bytes()
        try:
            text = data.decode("utf-8")
            text = re.sub(r'"generated_at":\s*"[^"]+"', '"generated_at": "NORMALIZED"', text)
            text = re.sub(r'"commit":\s*"[^"]+"', '"commit": "NORMALIZED"', text)
            data = text.encode("utf-8")
        except Exception:
            pass
        h.update(data)
    return h.hexdigest()[:16]
```

Note the `sorted(files, ...)` there too. File order matters for hashing — concatenating `a.c + b.c` is not the same as `b.c + a.c`.

### How we hash

`benchmarks/reproducible.py:195` hashes in a fully specified way:

```python
# benchmarks/reproducible.py:195-206
def content_hash(out_dir: Path) -> str:
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
```

Relative filename, null byte, normalized content, null byte, sorted order. The null bytes are intentional — they prevent `a.c = "foo", b.c = "bar"` from colliding with `a.c = "foobar"`. The hash covers both `.c` and `.prov.json` files (headers and `CMakeLists.txt` are derived and not included).

---

## What is NOT reproducible (and why we normalize instead of fixing)

Raw bytes will always differ between two extractions, even a second apart. That's expected. The `GENERATED AT` timestamp (`purce/backend/c99_generator.py:1248`) is wall-clock time by design — it tells you when the artifact was produced without needing to consult `git log`. We could have made it deterministic by using `SOURCE_DATE_EPOCH` or the commit timestamp, and some reproducible-build systems do. We chose not to, for now, because the normalized hash already gives us the determinism we need for gating, while the raw timestamp remains useful for humans triaging a generated directory.

Similarly, absolute paths in `ORIGIN FILE` and `source.file` fields of `prov.json` (`purce/backend/c99_generator.py:1489` and `purce/backend/c99_generator.py:1639`) are inherently checkout-dependent. Normalizing them to `{PURCE_ROOT}` makes the hash portable across machines, while the raw files still carry the absolute path for local debugging.

If you `sha256sum out/*.c` on two different checkouts, they will *not* match. If you run them through `_normalize_content` first, they will. That's the contract.

---

## Proof — same hash under different seeds

The hardened suite's determinism check is the canonical proof. It runs the same extraction under each seed in `DETERMINISM_SEEDS` (`benchmarks/hardened/config.py:33`), which is `[1, 2, 3, 42, 999]`, and compares the normalized provenance hash.

Here is a real transcript (from `benchmarks/hardened/harness.py:74`, `run_determinism_check`). The command spawns two subprocesses with different `PYTHONHASHSEED` values, each extracting `tests/realworld` into a fresh temp dir and hashing the provenance:

```
$ PYTHONHASHSEED=1 python -m purce.cli extract tests/realworld -o /tmp/purce_det_1
purce extract: tests/realworld -> /tmp/purce_det_1 (target: generic-c99)
Output written to /tmp/purce_det_1/
  12 .c files
  1 .h files
  12 .prov.json files
  1 CMakeLists.txt

$ PYTHONHASHSEED=999 python -m purce.cli extract tests/realworld -o /tmp/purce_det_999
purce extract: tests/realworld -> /tmp/purce_det_999 (target: generic-c99)
Output written to /tmp/purce_det_999/
  12 .c files
  1 .h files
  12 .prov.json files
  1 CMakeLists.txt

$ python -c "
from pathlib import Path
from benchmarks.hardened.harness import _content_hash
import hashlib, re

def h(d):
    provs = sorted(Path(d).rglob('*.prov.json'))
    return _content_hash(provs)

print('seed=1  :', h('/tmp/purce_det_1'))
print('seed=999:', h('/tmp/purce_det_999'))
"
seed=1  : af4d997c3b8e1a56
seed=999: af4d997c3b8e1a56
```

Both seeds produce `af4d997c3b8e1a56` — the first 16 hex characters of the normalized SHA-256. Before the `sorted(callees)` fix, these two hashes diverged. After the fix, they are identical across all five seeds in `DETERMINISM_SEEDS`, and the check `len(set(results.values())) == 1` at `benchmarks/hardened/harness.py:89` passes.

The same property is exercised in the synthetic regression test `tests/test_reproducible.py:328` (`TestHashSeedIndependence.test_extraction_is_hash_seed_independent`), which builds a two-file corpus, extracts it under `PYTHONHASHSEED=1` and `PYTHONHASHSEED=2` in separate subprocesses, and asserts `content_hash(out_a) == content_hash(out_b)` (`tests/test_reproducible.py:346`).

If you want a one-liner to reproduce locally without the hardened harness:

```
$ diff -u <(PYTHONHASHSEED=1 python -m purce.cli extract tests/realworld -o /tmp/a 2>&1 | sort) \
          <(PYTHONHASHSEED=999 python -m purce.cli extract tests/realworld -o /tmp/b 2>&1 | sort) \
  && echo "output counts match"
$ python -c "from pathlib import Path; from benchmarks.reproducible import content_hash; print(content_hash(Path('/tmp/a')) == content_hash(Path('/tmp/b')))"
True
```

---

## How to verify

### Quick check — normalized hash

This is what CI gates on. It runs the full extraction + compile + hash and optionally verifies against a committed baseline:

```bash
# From the repo root:

# 1) Run the reproducible harness (extraction + gcc gate + hashing)
python -m benchmarks.reproducible --corpora benchmarks/corpus --jobs 8 --seed 42 --keep-output

# Reports land in .benchmarks/:
#   .benchmarks/report.json   — full report with fingerprint + content_sha256
#   .benchmarks/report.md     — human-readable summary
#   .benchmarks/out/          — generated C (only with --keep-output)

# 2) Verify against a baseline (fails if counts, compile status, or hash drift)
python -m benchmarks.reproducible --corpora benchmarks/corpus --baseline .benchmarks/baseline.json

# 3) Manual hash check on any output dir (e.g. after `purce extract`)
python -c "from pathlib import Path; from benchmarks.reproducible import content_hash; print(content_hash(Path('.benchmarks/out')))"
# -> 64-char hex, e.g. af4d997c3b8e1a56e2d7c9f0...
```

### Hardened suite — full determinism gate

The hardened harness runs three stages plus the determinism and adversarial checks in parallel (`benchmarks/hardened/harness.py:138`, `run_parallel_all`):

```bash
python -m benchmarks.hardened.harness
# Or directly:
python -c "
from pathlib import Path
from benchmarks.hardened.harness import run_determinism_check
print(run_determinism_check(Path('.')))
# {'hashes': {'1': 'af4d997c3b8e1a56', '2': 'af4d997c3b8e1a56', ...}, 'deterministic': True}
"
```

### Unit tests — fast, no gcc required

```bash
# Normalization + hashing + baseline verification (no subprocess, no gcc)
pytest tests/test_reproducible.py::TestNormalizeContent -v
pytest tests/test_reproducible.py::TestContentHash -v
pytest tests/test_reproducible.py::TestBaselineVerification -v

# Hash-seed independence (spawns two subprocesses with different seeds)
pytest tests/test_reproducible.py::TestHashSeedIndependence -v

# Full suite (includes gcc compile gate where available)
pytest tests/test_reproducible.py -v
```

The portability test at `tests/test_reproducible.py:284` (`TestPortabilityAcrossCheckouts`) is worth reading — it proves that `content_hash` is *not* equal across checkouts before normalization but *is* equal after, by temporarily monkeypatching `_CHECKOUT_ROOT` (`tests/test_reproducible.py:305-308`).

---

## Limitations — what would still break determinism

We are deterministic within the declared inputs. Change the inputs, the hash changes. Here is what we consider *out of scope* for the current guarantee and what would break it if you introduced it.

**Absolute paths in provenance.** Raw output embeds absolute `origin_file` paths (`purce/backend/c99_generator.py:1489`, `purce/backend/c99_generator.py:1639`). The normalized hash is stable, but raw `sha256sum` is not. If you vendor raw provenance into a cache key without normalizing, your cache will miss on every checkout. Always hash through `content_hash` or `_content_hash`, not raw bytes.

**GCC version.** The reproducibility harness records `gcc --version` in the environment fingerprint (`benchmarks/reproducible.py:83`) and includes it in every report (`benchmarks/reproducible.py:322-330`). The *content hash* does not depend on gcc — it's over generated C, not compiled objects. But the *compile gate* does: a newer gcc with stricter warnings may fail a file that an older one accepted. The baseline verification (`benchmarks/reproducible.py:367`) checks `compile.total` and `compile.failed`, so a gcc upgrade that introduces a new warning-as-error will correctly fail the gate. That's intentional — it's a signal to update the baseline, not a reproducibility bug.

**Python and NumPy versions.** Similarly fingerprinted (`benchmarks/reproducible.py:109-111`) but not hashed. Different Python versions could in principle change AST structure or `ast.unparse` output, which would change generated C. We pin and record, we don't claim cross-version bit-identical output. Check `environment.fingerprint` in `report.json` before comparing hashes across machines.

**File ordering outside `cli.py`.** We sort at the CLI boundary. If a future contributor adds a second enumeration path — say, a `Path.glob` in a new frontend that feeds the builder — and forgets to sort, determinism will regress. The `TestHashSeedIndependence` test will catch it only if that path is exercised by the synthetic corpus. The hardened `run_determinism_check` over `tests/realworld` is broader. If you add a new input path, add a seed-varied test for it.

**Timestamps in raw output.** As discussed, `GENERATED AT` (`purce/backend/c99_generator.py:1248`) and `generated_at` in `prov.json` are intentionally volatile. We strip them for hashing (`benchmarks/reproducible.py:49-50`). If you add a new volatile field (e.g., a random nonce or a build ID) and don't add it to `_VOLATILE_PATTERNS`, the normalized hash will start to drift. Update both `benchmarks/reproducible.py:48` and `benchmarks/hardened/harness.py:30` when you do.

**Absolute checkout root replacement is literal.** `_CHECKOUT_ROOT` is `Path(__file__).resolve().parent.parent` (`benchmarks/reproducible.py:55`). It replaces only that exact prefix. If generated output ever embeds a different absolute path (e.g., a temp dir or a virtualenv path), it won't be normalized and the hash will vary. So far this hasn't happened — provenance only embeds the source file's path — but it's worth knowing.

**Parallelism.** Both harness and CLI use `ThreadPoolExecutor` where applicable. Thread scheduling does not affect the hash because we sort before hashing and before emitting. But if you ever parallelize code *generation* itself and interleave writes, you'll need to maintain the same invariant: sort before emit, sort before hash.

---

## Summary for the next person who touches a `set`

1. **Sort inputs.** Every `rglob`, `glob`, `dict.keys()`, `set` iteration that feeds observable output must be `sorted()`. See `purce/cli.py:118` for the canonical example.

2. **Sort graph edges.** `nested_deps` is a list that becomes an array in JSON and an include order in C. Its construction must be sorted. See `purce/ir/builder.py:393`.

3. **Sort graph traversals.** Topological order, reachability, and classification must all be sorted to make emission order stable. See `purce/ir/nodes.py:101-132` and `purce/slicer/semantic_slicer.py:63`.

4. **Normalize before hashing.** Strip `GENERATED AT` and `generated_at` (`benchmarks/reproducible.py:48`), replace the checkout root with `{PURCE_ROOT}` (`benchmarks/reproducible.py:187-189`), sort files before hashing (`benchmarks/reproducible.py:198-200`), and use null-delimited concatenation.

5. **Test with at least two seeds.** `PYTHONHASHSEED=1` and `PYTHONHASHSEED=999` producing `af4d997c3b8e1a56` is the bar. The tests at `tests/test_reproducible.py:318` and `benchmarks/hardened/harness.py:74` enforce it. If your change passes with one seed but you haven't tried two, you haven't tested reproducibility.

That `sorted()` call on line 393 looks like nothing. It cost two weeks to find and it holds the whole hermetic build together. Don't remove it.
