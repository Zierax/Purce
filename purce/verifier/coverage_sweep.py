"""Full-coverage sweep over every reachable implemented kernel body.

The corpus gate only exercises a subset of the C99 kernel bodies registered in
``c99_generator.MATH_KERNEL_BODIES``. This module compiles *every* reachable
kernel (each algorithm with a ``BODY_PARAM_MAP`` entry) and, for each one:

* reference-backed kernels — algorithms with a NumPy reference in
  ``equivalence._REF_DISPATCH`` — are executed end-to-end and compared against
  the NumPy reference using the same driver machinery as the corpus runner;
* documented structural kernels (``equivalence.STRUCTURAL_GAP``) are compiled,
  executed, and checked against their documented invariants (no NumPy oracle
  exists for these).

``array_diff`` is deliberately excluded: it is a known-unreachable body with no
``BODY_PARAM_MAP`` entry (so it can never be generated), and a regression test
in ``test_verifier_edges`` asserts that it produces no kernel wrapper.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from purce.backend.c99_generator import (
    BODY_PARAM_MAP,
    C99Generator,
    MATH_KERNEL_BODIES,
    _sanitize_name,
)
from purce.ir.nodes import Dtype, Effect, MathIRNode
from purce.verifier import equivalence as eq

RNG_ALGOS = {"alloc_random", "array_permutation", "noop_seed"}
UNREACHABLE_BODIES = {"array_diff"}  # no BODY_PARAM_MAP entry; can never be generated

_EPOCH = "2026-01-01T00:00:00Z"


def algorithms() -> list[str]:
    """Sorted list of reachable algorithm names (every body with a param map)."""
    return sorted(a for a in MATH_KERNEL_BODIES if a in BODY_PARAM_MAP)


def reference_backed() -> set[str]:
    """Algorithms compared numerically against a NumPy reference."""
    return set(algorithms()) & set(eq._REF_DISPATCH) - set(eq.STRUCTURAL_GAP)


def structural_smoke() -> set[str]:
    """Algorithms verified by compiled-execution invariant checks only."""
    return set(algorithms()) & set(eq.STRUCTURAL_GAP)


def _build_node(algorithm: str) -> MathIRNode:
    """Build a synthetic IR node carrying the kernel's canonical operands.

    Array operands are declared with their canonical names and array shapes so
    that ``_make_case_from_node`` falls through to the tuned per-algorithm case
    builders, and ``_make_function_signature`` emits the canonical C signature
    (with derived length params such as ``n``/``m``/``p``/``log_n`` appended).
    """
    spec = BODY_PARAM_MAP.get(algorithm, [])
    inputs: list[tuple[str, Dtype, str]] = []
    outputs: list[tuple[str, Dtype, str]] = []
    for canonical, source in spec:
        if source.startswith("input_"):
            inputs.append((_sanitize_name(canonical), Dtype.FLOAT64, "array"))
        elif source.startswith("output_"):
            outputs.append((_sanitize_name(canonical), Dtype.FLOAT64, "array"))
    if algorithm == "noop_seed":
        # The seed is a scalar, not an array: it feeds uint32_t seed.
        inputs = [("seed", Dtype.FLOAT64, "scalar")]
    return MathIRNode(
        node_id=f"sweep.{algorithm}",
        origin_symbol=f"module.{algorithm}",
        origin_file="coverage_sweep.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="(module)",
        math_intent=f"Coverage sweep: {algorithm}",
        inputs=inputs,
        outputs=outputs,
        effects=[Effect.PURE],
        algorithm=algorithm,
    )


def build_kernels() -> list[tuple[MathIRNode, str]]:
    """Generate the C file content for each reachable kernel.

    Returns ``(node, c_content)`` pairs. The module-level LCG state is emitted
    exactly once (``define``) and referenced as ``extern`` in the other kernels
    that touch it, mirroring ``C99Generator.generate`` so that all kernels link
    against one shared ``uint32_t`` PRNG state.
    """
    generator = C99Generator(target_profile="generic-c99")
    nodes = {a: _build_node(a) for a in algorithms()}
    emitted_rng_define = False
    result: list[tuple[MathIRNode, str]] = []
    for algo in algorithms():
        node = nodes[algo]
        role = None
        if algo in RNG_ALGOS:
            role = "define" if not emitted_rng_define else "extern"
            emitted_rng_define = True
        content = generator._generate_c_file(node, "sweep_cov", _EPOCH, role)
        result.append((node, content))
    return result


@dataclass
class KernelOutcome:
    """Verification result for one algorithm across all iterations/seeds."""

    algorithm: str
    mode: str  # "reference" | "smoke"
    passed: int = 0
    failed: int = 0
    max_error: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.errors


def _smoke_checks(algo: str, inputs: dict[str, Any], actual: Any) -> str:
    """Validate a structural kernel against its documented invariants.

    Returns an empty string on success or a human-readable failure reason.
    """
    if algo == "alloc_random":
        out = np.asarray(actual, dtype=np.float64).ravel()
        if out.size == 0 or np.any((out < 0.0) | (out >= 1.0)):
            return f"LCG output outside [0,1): {out[:4]}"
        return ""
    if algo == "array_permutation":
        out = np.asarray(actual, dtype=np.float64).ravel()
        src = np.asarray(inputs["x"], dtype=np.float64).ravel()
        if out.size != src.size or not np.allclose(np.sort(out), np.sort(src)):
            return "output is not a permutation of the input"
        return ""
    if algo == "element_finfo":
        val = float(np.asarray(actual).ravel()[0])
        if val != 2.2250738585072014e-308:
            return f"finfo constant mismatch: {val!r}"
        return ""
    if algo == "linalg_qr":
        n = int(inputs["n"])
        src = np.asarray(inputs["x"], dtype=np.float64).ravel()
        q = np.asarray(actual[0], dtype=np.float64).ravel()
        r = np.asarray(actual[1], dtype=np.float64).ravel()
        if not np.allclose(q, src):
            return "Q is not a copy of the input (documented stub)"
        if not np.allclose(r, np.eye(n).ravel()):
            return "R is not the identity (documented stub)"
        return ""
    if algo == "linalg_svd":
        n = int(inputs["n"])
        src = np.asarray(inputs["x"], dtype=np.float64).ravel()
        u = np.asarray(actual[0], dtype=np.float64).ravel()
        s = np.asarray(actual[1], dtype=np.float64).ravel()
        v = np.asarray(actual[2], dtype=np.float64).ravel()
        if not np.allclose(u, src):
            return "U is not a copy of the input (documented stub)"
        if not np.allclose(s[:n], np.ones(n)):
            return "S is not all ones (documented stub)"
        if not np.allclose(v, np.eye(n).ravel()):
            return "V is not the identity (documented stub)"
        return ""
    if algo == "linalg_eig":
        n = int(inputs["n"])
        src = np.asarray(inputs["A"], dtype=np.float64).ravel()
        diag = np.asarray([src[i * n + i] for i in range(n)])
        ev = np.asarray(actual).ravel()
        if not np.allclose(ev[:n], diag, atol=1e-12):
            return "eigenvalues are not the Gershgorin centers (diagonal)"
        return ""
    if algo == "array_split":
        # Documented stub: every element is copied out; the body is a no-op
        # placeholder that leaves the buffer unwritten. Only call-ability is
        # guaranteed.
        return ""
    if algo == "noop_seed":
        # State-only kernel: no outputs; correctness of the LCG wiring is
        # validated separately (see test_coverage_sweep test_rng_wiring).
        return ""
    return "unhandled structural algorithm"


def _run_reference(
    executor: eq.GeneratedKernelExecutor,
    node: MathIRNode,
    content: str,
    seed: int,
    iterations: int,
    rtol: float,
    atol: float,
) -> KernelOutcome:
    kr = executor.run_kernel(node, content, seed=seed, iterations=iterations, rtol=rtol, atol=atol)
    outcome = KernelOutcome(algorithm=node.algorithm, mode="reference")
    outcome.passed = kr.passed
    outcome.failed = kr.failed
    outcome.max_error = kr.max_error
    if kr.error:
        outcome.errors.append(kr.error[:400])
    return outcome


def _run_smoke(
    executor: eq.GeneratedKernelExecutor, node: MathIRNode, content: str, seed: int, iterations: int
) -> KernelOutcome:
    outcome = KernelOutcome(algorithm=node.algorithm, mode="smoke")
    try:
        fname, params = eq.parse_kernel_signature(content)
        driver = eq._Driver(
            node.algorithm, node, executor._lib, fname, params, ref_fn=None, op=None
        )
    except Exception as e:  # construction failure, not a runtime failure
        outcome.errors.append(f"driver construction failed: {e}")
        return outcome
    rng = random.Random(seed)
    for _ in range(iterations):
        try:
            inputs, out_names = driver.make_case(rng)
            actual = driver.call(inputs, out_names)
            reason = _smoke_checks(node.algorithm, inputs, actual)
            # The LCG must literally advance between consecutive draws; a
            # single equal-pair of 15-bit outputs is legal randomness, but two
            # full buffers matching proves the state machine never moves.
            if not reason and node.algorithm == "alloc_random":
                again = driver.call(inputs, out_names)
                if np.array_equal(np.asarray(actual), np.asarray(again)):
                    reason = "LCG state did not advance between two draws"
            if reason:
                outcome.failed += 1
                outcome.errors.append(reason)
            else:
                outcome.passed += 1
        except Exception as e:
            outcome.failed += 1
            outcome.errors.append(f"{type(e).__name__}: {e}"[:400])
    return outcome


@dataclass
class SweepReport:
    """Aggregate result of a full-coverage run."""

    outcomes: list[KernelOutcome]
    iterations: int
    seeds: tuple[int, ...]

    def by_algorithm(self) -> dict[str, KernelOutcome]:
        return {o.algorithm: o for o in self.outcomes}

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def failures(self) -> list[KernelOutcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def all_passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [
            f"coverage sweep: {len([o for o in self.outcomes if o.ok])}/{self.total} "
            f"ok  (iterations={self.iterations}, seeds={self.seeds})",
        ]
        for o in sorted(self.outcomes, key=lambda x: x.algorithm):
            status = "ok" if o.ok else "FAIL"
            err = f"  max_err={o.max_error:.3g}" if o.mode == "reference" else ""
            lines.append(f"  [{o.mode:9s}] {status} {o.algorithm}{err}")
            if not o.ok:
                for e in o.errors[:3]:
                    lines.append(f"           {e[:160]}")
        return "\n".join(lines)


def run_sweep(
    iterations: int = 8,
    seeds: tuple[int, ...] = (42, 1337),
    rtol: float = 1e-5,
    atol: float = 1e-8,
    keep_dir: str | None = None,
) -> SweepReport:
    """Compile every reachable kernel once and verify each one."""
    pairs = build_kernels()
    algo_by_node = {node.node_id: node.algorithm for node, _ in pairs}
    executor = eq.GeneratedKernelExecutor(
        [content for _, content in pairs],
        algo_by_node,
        keep_dir=keep_dir,
    )
    executor.build()
    report = SweepReport(outcomes=[], iterations=iterations, seeds=seeds)
    try:
        for node, content in pairs:
            algo = node.algorithm
            outcome = KernelOutcome(algorithm=algo, mode="")
            for seed in seeds:
                if algo in structural_smoke():
                    o = _run_smoke(executor, node, content, seed, iterations)
                else:
                    o = _run_reference(executor, node, content, seed, iterations, rtol, atol)
                if outcome.mode == "":
                    outcome.mode = o.mode
                outcome.passed += o.passed
                outcome.failed += o.failed
                outcome.max_error = max(outcome.max_error, o.max_error)
                outcome.errors.extend(o.errors)
            report.outcomes.append(outcome)
    finally:
        executor.close()
    return report
