"""Full-coverage sweep tests: every reachable implemented kernel body is
verified end-to-end (reference-backed kernels against NumPy, documented
structural kernels against their invariants)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from purce.backend.c99_generator import BODY_PARAM_MAP, C99Generator, MATH_KERNEL_BODIES
from purce.ir.nodes import Dtype, Effect, MathIRNode
from purce.verifier import coverage_sweep as cs
from purce.verifier import ctypes_bridge as _ct
from purce.verifier import equivalence as eq

_HAS_GCC = shutil.which("gcc") is not None or shutil.which("cc") is not None
RequiresGcc = pytest.mark.skipif(not _HAS_GCC, reason="gcc not available")

_EPOCH = "2026-01-01T00:00:00Z"


def _make_node(algorithm: str) -> MathIRNode:
    spec = BODY_PARAM_MAP.get(algorithm, [])
    inputs: list[tuple[str, Dtype, str]] = []
    outputs: list[tuple[str, Dtype, str]] = []
    for canonical, source in spec:
        if source.startswith("input_"):
            inputs.append((canonical, Dtype.FLOAT64, "array"))
        elif source.startswith("output_"):
            outputs.append((canonical, Dtype.FLOAT64, "array"))
    if algorithm == "noop_seed":
        inputs = [("seed", Dtype.FLOAT64, "scalar")]
    return MathIRNode(
        node_id=f"rng_{algorithm}",
        origin_symbol=f"module.{algorithm}",
        origin_file="test_coverage_sweep.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="(module)",
        math_intent=f"RNG wiring: {algorithm}",
        inputs=inputs,
        outputs=outputs,
        effects=[Effect.PURE],
        algorithm=algorithm,
    )


class TestCoverageCompleteness:
    def test_reachable_bodies_matched_to_paramspec(self) -> None:
        reachable = cs.algorithms()
        assert set(reachable) == set(MATH_KERNEL_BODIES) - cs.UNREACHABLE_BODIES

    def test_array_diff_remains_unreachable_and_wrapperless(self) -> None:
        assert cs.UNREACHABLE_BODIES == {"array_diff"}
        assert "array_diff" not in cs.algorithms()
        assert _ct._make_kernel_wrapper("array_diff") == ""

    def test_reference_backed_and_structural_are_disjoint_and_total(self) -> None:
        rb = cs.reference_backed()
        sm = cs.structural_smoke()
        assert rb.isdisjoint(sm)
        assert rb | sm == set(cs.algorithms())

    def test_every_reference_backed_kernel_has_a_numpy_reference(self) -> None:
        assert cs.reference_backed() <= set(eq._REF_DISPATCH)

    def test_structural_gap_is_fully_documented(self) -> None:
        assert cs.structural_smoke() == set(eq.STRUCTURAL_GAP) & set(cs.algorithms())

    def test_newly_wired_loop_concat_is_reference_backed(self) -> None:
        assert "loop_concat" in cs.reference_backed()


@pytest.fixture(scope="module")
def sweep_report():
    if not _HAS_GCC:
        pytest.skip("gcc not available")
    return cs.run_sweep(iterations=4, seeds=(42, 1337))


@RequiresGcc
class TestCoverageSweepRuntime:
    def test_sweep_covers_every_reachable_body(self, sweep_report) -> None:
        assert sweep_report.total == len(cs.algorithms())
        assert set(sweep_report.by_algorithm()) == set(cs.algorithms())

    def test_all_reference_backed_kernels_match_numpy(self, sweep_report) -> None:
        failures = [
            o.algorithm for o in sweep_report.outcomes
            if not o.ok and o.mode == "reference"
        ]
        assert failures == []

    def test_all_structural_kernels_satisfy_documented_invariants(
        self, sweep_report,
    ) -> None:
        failures = [
            o.algorithm for o in sweep_report.outcomes
            if not o.ok and o.mode == "smoke"
        ]
        assert failures == []

    def test_sweep_fully_passes(self, sweep_report) -> None:
        assert sweep_report.all_passed, sweep_report.summary()


@RequiresGcc
class TestRngWiring:
    """End-to-end proof that noop_seed actually rewires the shared LCG stream."""

    @staticmethod
    def _build(nodes: list[MathIRNode]):
        generator = C99Generator(target_profile="generic-c99")
        contents: list[str] = []
        algo_by_node: dict[str, str] = {}
        role_used = False
        for node in nodes:
            if node.algorithm in cs.RNG_ALGOS:
                role = "define" if not role_used else "extern"
                role_used = True
            else:
                role = None
            contents.append(generator._generate_c_file(
                node, "rng_wire", _EPOCH, role,
            ))
            algo_by_node[node.node_id] = node.algorithm
        executor = eq.GeneratedKernelExecutor(contents, algo_by_node)
        executor.build()
        return executor

    @staticmethod
    def _call(executor, node: MathIRNode, inputs: dict) -> np.ndarray:
        generator = C99Generator(target_profile="generic-c99")
        c = generator._generate_c_file(node, "rng_wire", _EPOCH, "extern")
        fname, params = eq.parse_kernel_signature(c)
        driver = eq._Driver(node.algorithm, node, executor._lib, fname, params,
                            ref_fn=None, op=None)
        out_names = [p.name for p in params if driver._is_output(p.name)]
        return driver.call(inputs, out_names)

    def test_noop_seed_reproduces_default_stream(self) -> None:
        alloc = _make_node("alloc_random")
        seed = _make_node("noop_seed")

        lib_default = self._build([alloc])
        r_default = self._call(lib_default, alloc, {"n": 8.0})

        lib_seeded = self._build([seed, alloc])
        self._call(lib_seeded, seed, {"seed": 12345.0})
        r_seeded = self._call(lib_seeded, alloc, {"n": 8.0})

        assert np.array_equal(r_default, r_seeded), "seed(12345) != default stream"

    def test_noop_seed_alters_the_stream(self) -> None:
        alloc = _make_node("alloc_random")
        seed = _make_node("noop_seed")

        lib_default = self._build([alloc])
        r_default = self._call(lib_default, alloc, {"n": 8.0})

        lib_other = self._build([seed, alloc])
        self._call(lib_other, seed, {"seed": 999.0})
        r_other = self._call(lib_other, alloc, {"n": 8.0})

        assert not np.array_equal(r_default, r_other), "different seed, same stream"


@RequiresGcc
class TestSpecificRegressions:
    """Focused checks for defects the sweep exposed (before this change)."""

    def test_array_unique_matches_numpy(self) -> None:
        node, content = next(
            (n, c) for n, c in cs.build_kernels() if n.algorithm == "array_unique"
        )
        executor = eq.GeneratedKernelExecutor([content], {node.node_id: node.algorithm})
        executor.build()
        try:
            kr = executor.run_kernel(node, content, seed=7, iterations=5)
            assert kr.failed == 0 and not kr.error, kr.error
        finally:
            executor.close()

    def test_fft_receives_correct_log_n(self) -> None:
        # Regression: the equivalence case builder previously omitted log_n,
        # so the C bit-reversal stage never ran.
        import random
        node = _make_node("fft")
        case = eq._make_case("fft", node, random.Random(42))
        assert "log_n" in case
        n = int(case["n"])
        assert 2 ** int(case["log_n"]) == n

    def test_searchsorted_is_sorted_and_scalar_output(self) -> None:
        import random
        node = _make_node("array_searchsorted")
        case = eq._make_case("array_searchsorted", node, random.Random(42))
        assert np.all(np.diff(case["x"]) >= 0.0) or case["x"].size == 1
        assert "v" in case and "n" in case

    def test_element_ceil_negative_values(self) -> None:
        # The old body computed ceil((-2.5)) == -1 via truncation.
        body = MATH_KERNEL_BODIES["element_ceil"]
        assert "ceil(x[i])" in body
        assert "(int)x[i]" not in body