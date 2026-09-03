"""Edge-coverage tests for CLI, parser, and C99 backend.

Targets lines the regular suites do not reach:
  * purce/cli.py: 33, 82-84, 87-88, 101-105, 228, 238, 244-245, 249
  * purce/parser/python_parser.py: 42, 166, 207, 209-210, 219, 230, 250,
    272, 281, 290, 297, 330, 332, 334
  * purce/backend/c99_generator.py: 976, 978, 980, 1002, 1026, 1292, 1435, 1437

Also verifies the deterministic-enumeration regression from commit
c7faa6b (sorted rglob discovery / reproducible origin lines) and
compiles a generated corpus with strict gcc.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

import purce.backend.c99_generator as cg
import purce.cli as cli_mod
from purce import __version__
from purce.backend.c99_generator import (
    BODY_PARAM_MAP,
    C99Generator,
    _build_body_param_mapping,
    _sanitize_name,
    _substitute_body_params,
)
from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import Dtype, Effect, MathIRGraph, MathIRNode
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer, SliceResult
from purce.verifier.fuzzer import FuzzResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fixtures_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures")


def _mk_node(
    algorithm: str,
    inputs: list[tuple[str, object]],
    outputs: list[tuple[str, object]],
    node_id: str = "m.mod_x1",
) -> MathIRNode:
    """Build a concrete MathIRNode without going through the parser."""
    return MathIRNode(
        node_id=node_id,
        origin_symbol="m.f",
        origin_file="t.py",
        origin_line=3,
        origin_commit=None,
        origin_signature="double -> double",
        math_intent=f"kernel {algorithm}",
        inputs=[(name, Dtype.FLOAT64, shape) for name, shape in inputs],
        outputs=[(name, Dtype.FLOAT64, shape) for name, shape in outputs],
        effects=[Effect.PURE],
        algorithm=algorithm,
        stack_usage=32,
    )


def _graph(*nodes: MathIRNode) -> MathIRGraph:
    g = MathIRGraph()
    for n in nodes:
        g.add_node(n)
        g.entry_points.append(n.node_id)
    return g


def _render_c(graph: MathIRGraph) -> list[str]:
    res = C99Generator().generate(graph, "mymod")
    return [f.content for f in res.files if f.file_type == "c"]


def _run_pipeline(source: str, module_name: str = "test_mod"):
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slice_result = SemanticSlicer().slice(graph, list(graph.nodes.keys()))
    gen_result = C99Generator(target_profile="generic-c99").generate(
        slice_result.graph, module_name=module_name
    )
    return gen_result, graph, slice_result


def _reset_root_logging() -> None:
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)


def _snapshot(directory: str) -> dict[str, str]:
    """Sorted relpath -> normalized content snapshot of an output tree."""
    out: dict[str, str] = {}
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        text = re.sub(r"\* GENERATED AT:.*", "", text)
        text = re.sub(r'"generated_at":\s*"[^"]*"', '"generated_at": ""', text)
        out[str(p.relative_to(directory))] = text
    return out


# ── CLI: logging levels and diagnostics ──────────────────────────────────────


class TestCLILogging:
    def test_quiet_suppresses_info(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        _reset_root_logging()
        try:
            out_dir = str(tmp_path / "out")
            result = runner.invoke(cli_mod.main, ["-q", "extract", fixtures_dir, "-o", out_dir])
            assert result.exit_code == 0
            assert "Found 3 Python files" not in result.output
        finally:
            _reset_root_logging()

    def test_verbose_prints_diagnostics(self, runner: CliRunner, tmp_path: object) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "good.py").write_text(
            "import numpy as np\n\ndef add(x):\n    return np.add(x, 1.0)\n", encoding="utf-8"
        )
        (src / "bad.py").write_text(
            "import numpy as np\n\ndef dyn(x):\n    y = np.add(x, 1.0)\n    return eval('y')\n",
            encoding="utf-8",
        )
        _reset_root_logging()
        try:
            out_dir = str(tmp_path / "out")
            result = runner.invoke(cli_mod.main, ["extract", str(src), "-o", out_dir, "--verbose"])
            assert result.exit_code == 0
            assert "Diagnostics (1)" in result.output
            assert "[ERROR]" in result.output
            assert "Dynamic dispatch 'eval()'" in result.output
        finally:
            _reset_root_logging()

    def test_no_kernels_warns_and_exits_zero(self, runner: CliRunner, tmp_path: object) -> None:
        src = tmp_path / "plain"
        src.mkdir()
        (src / "pure.py").write_text("def p(x):\n    return x * 2 + 1\n", encoding="utf-8")
        result = runner.invoke(cli_mod.main, ["extract", str(src), "-o", str(tmp_path / "out")])
        assert result.exit_code == 0
        assert "Warning: No math kernels found" in result.output


class TestCLIDataAssetsAndVerify:
    def test_data_asset_without_embed_flag_exits(
        self,
        runner: CliRunner,
        fixtures_dir: str,
        tmp_path: object,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_slice(self, graph, entry_points) -> SliceResult:
            return SliceResult(graph=graph, data_assets=["module.asset"])

        monkeypatch.setattr(cli_mod.SemanticSlicer, "slice", fake_slice)
        result = runner.invoke(cli_mod.main, ["extract", fixtures_dir, "-o", str(tmp_path / "out")])
        assert result.exit_code == 1
        assert "data assets found but --embed-assets not set" in result.output

    def test_verify_z3_not_available(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli_mod.Z3Verifier, "available", False)

        def fake_fuzz_all(self, iterations: int = 1000) -> dict:
            return {
                "matmul": FuzzResult(
                    operation="matmul",
                    iterations=iterations,
                    passed=iterations,
                    failed=0,
                    tested_c=True,
                )
            }

        monkeypatch.setattr(cli_mod.DifferentialFuzzer, "fuzz_all", fake_fuzz_all)
        monkeypatch.setattr(
            cli_mod, "_try_c_backend", lambda: (cli_mod.DifferentialFuzzer(), mock.Mock())
        )
        result = runner.invoke(cli_mod.main, ["verify", "--iterations", "3"])
        # Without Z3 the verification suite cannot run at all, so the command
        # must fail loudly rather than silently skipping the Z3 phase.
        assert result.exit_code == 1
        assert (
            "Z3 solver: not available (install z3-solver); verification cannot run" in result.output
        )

    def test_verify_failure_exit_code(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_fuzz_all(self, iterations: int = 1000) -> dict:
            return {
                "matmul": FuzzResult(
                    operation="matmul",
                    iterations=iterations,
                    passed=iterations - 1,
                    failed=1,
                    tested_c=True,
                )
            }

        monkeypatch.setattr(cli_mod.DifferentialFuzzer, "fuzz_all", fake_fuzz_all)
        monkeypatch.setattr(
            cli_mod, "_try_c_backend", lambda: (cli_mod.DifferentialFuzzer(), mock.Mock())
        )
        result = runner.invoke(cli_mod.main, ["verify", "--iterations", "3"])
        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "Some operations failed" in result.output


class TestCLIModuleEntry:
    def test_main_guard_runs_module(self) -> None:
        import contextlib
        import io
        import runpy
        import warnings

        buf = io.StringIO()
        prev_argv = sys.argv
        sys.argv = ["purce", "--version"]
        try:
            with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*found in sys.modules.*",
                    category=RuntimeWarning,
                )
                with pytest.raises(SystemExit) as excinfo:
                    runpy.run_module("purce.cli", run_name="__main__")
            assert excinfo.value.code == 0
            assert __version__ in buf.getvalue()
        finally:
            sys.argv = prev_argv


# ── Parser: unsupported syntax / NumPy-call edges ─────────────────────────────


class TestParserDiagnostics:
    def test_diagnostic_to_dict(self) -> None:
        p = PythonParser()
        res = p.parse_source("def broken(:\n    pass\n", "x.py")
        assert len(res.diagnostics) == 1
        d = res.diagnostics[0].to_dict()
        assert d["severity"] == "ERROR"
        assert d["construct"] == "syntax"
        assert d["line"] == "1"

    def test_eval_is_unsupported_and_dropped(self) -> None:
        p = PythonParser()
        source = "import numpy as np\ndef dyn(x):\n    y = np.add(x, 1.0)\n    return eval('y')\n"
        res = p.parse_source(source, "e.py")
        assert res.functions == []
        assert [d.construct for d in res.diagnostics] == ["eval"]

    def test_open_is_unsupported_and_dropped(self) -> None:
        p = PythonParser()
        source = (
            "import numpy as np\n"
            "def io(x):\n"
            "    y = np.zeros(8)\n"
            "    data = open('f').read()\n"
            "    return y\n"
        )
        res = p.parse_source(source, "o.py")
        assert res.functions == []
        assert [d.construct for d in res.diagnostics] == ["open"]

    def test_non_call_func_target_skipped(self) -> None:
        p = PythonParser()
        source = "import numpy as np\ndef h(a, b):\n    y = np.add(a, b)\n    return y[0](0.5)\n"
        res = p.parse_source(source, "t.py")
        assert len(res.functions) == 1
        assert res.functions[0].algorithm == "element_add"

    def test_bare_numpy_name_call_is_ignored(self) -> None:
        p = PythonParser()
        res = p.parse_source("import numpy as np\ndef k(x):\n    return numpy(x)\n", "k.py")
        assert res.functions == []

    def test_self_arg_skipped_in_inputs(self) -> None:
        p = PythonParser()
        source = "import numpy as np\ndef member(self, x):\n    return np.add(x, 1.0)\n"
        res = p.parse_source(source, "s.py")
        assert len(res.functions) == 1
        names, dtypes, _ = zip(*res.functions[0].inputs)
        assert list(names) == ["x"]
        assert list(dtypes) == [Dtype.FLOAT64]

    def test_attribute_type_hints(self) -> None:
        p = PythonParser()
        source = (
            "import numpy as np\n"
            "def typed(x: np.int32, y: np.int64) -> np.float64:\n"
            "    return np.add(x, y)\n"
        )
        res = p.parse_source(source, "ty.py")
        assert len(res.functions) == 1
        fn = res.functions[0]
        assert fn.inputs == [("x", Dtype.INT32, "array"), ("y", Dtype.INT64, "array")]
        assert fn.outputs[0][1] == Dtype.FLOAT64

    def test_extract_segment_beyond_source_returns_empty(self) -> None:
        tree = ast.parse("def f(x):\n    return np.add(x, 1)\n")
        func = tree.body[0]
        p = PythonParser()
        p._source_lines = ["def f(x):"]
        assert p._extract_source_segment(func) == ""

    def test_extract_segment_without_lineno_returns_empty(self) -> None:
        p = PythonParser()
        p._source_lines = ["line"]
        assert p._extract_source_segment(types.SimpleNamespace()) == ""

    def test_extract_segment_normal(self) -> None:
        tree = ast.parse("def f(x):\n    return np.add(x, 1)\n")
        func = tree.body[0]
        p = PythonParser()
        p._source_lines = ["def f(x):", "    return np.add(x, 1)"]
        assert p._extract_source_segment(func) == "def f(x):\n    return np.add(x, 1)"


class TestParserNormalizeDeadGuard:
    def test_bare_numpy_call_never_reaches_bare_split_guard(self) -> None:
        # A call target can never reduce to the bare name "numpy" (line 249
        # pre-filter), so the `len(parts) < 2` guard in _normalize_numpy_target
        # is unreachable in practice.
        p = PythonParser()
        res = p.parse_source("import numpy as np\ndef q(x):\n    return numpy(x)\n", "q.py")
        assert res.functions == []


# ── Backend: name sanitization / string builders ──────────────────────────────


class TestSanitizeName:
    def test_trailing_underscore_stripped(self) -> None:
        assert _sanitize_name("vec_") == "vec"

    def test_only_underscores_becomes_unnamed(self) -> None:
        assert _sanitize_name("___") == "_unnamed"

    def test_leading_digit_prefixed(self) -> None:
        assert _sanitize_name("1vec") == "_1vec"

    def test_separators_normalized(self) -> None:
        assert _sanitize_name("a.b/c\\d") == "a_b_c_d"


class TestBodyParamMapping:
    def test_unknown_spec_source_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patched = dict(BODY_PARAM_MAP)
        patched["patched_algo"] = [("Z", "mystery_source")]
        monkeypatch.setattr(cg, "BODY_PARAM_MAP", patched)
        node = _mk_node("patched_algo", [("a", "array")], [("b", "array")])
        mapping = _build_body_param_mapping(node)
        assert mapping["Z"] == "Z"

    def test_empty_mapping_is_passthrough(self) -> None:
        body = "for (int i = 0; i < n; i++) { out[i] = x[i]; }"
        assert _substitute_body_params(body, {}) == body


class TestUnknownAlgorithm:
    def test_missing_impl_emits_guard_error(self) -> None:
        node = _mk_node("unknown_kernel", [("a", "array")], [("b", "array")])
        contents = _render_c(_graph(node))
        assert len(contents) == 1
        assert "No C implementation for algorithm 'unknown_kernel'" in contents[0]
        assert "#error" in contents[0]
        assert re.search(
            r"void \w+\(const double \* restrict a, double \* restrict b\)", contents[0]
        )

    def test_output_name_collision_is_deduped(self) -> None:
        node = _mk_node(
            "element_add",
            [("A", "array"), ("B", "array")],
            [("C", "array"), ("C", "array")],
        )
        contents = _render_c(_graph(node))
        sig = re.search(r"void \w+\(.*?\)", contents[0], re.DOTALL).group(0)
        assert sig.count("C") == 1


class TestScalarParamCorner:
    """Scalar-shaped inputs become bare `double B` params and are de-subscripted.

    The scalar predicate is `s != "array"` and not a str starting with "(".
    A *tuple* shape ("m",) therefore also classifies as a scalar param with the
    current code (see finding note).
    """

    def _pair(self, shape: object, node_id: str) -> str:
        node = _mk_node(
            "element_mul",
            [("A", "array"), ("B", shape)],
            [("C", "array")],
            node_id=node_id,
        )
        return _render_c(_graph(node))[0]

    def test_none_shape_is_scalar(self) -> None:
        content = self._pair(None, "m.mod_x_none")
        assert re.search(r"\bdouble B\b", content)
        assert re.search(r"C\[i\] = A\[i\] \* B;", content)
        assert "B[i]" not in content

    def test_loop_var_shape_is_scalar(self) -> None:
        content = self._pair("loop_var", "m.mod_x_loop")
        assert re.search(r"\bdouble B\b", content)
        assert "B[i]" not in content

    def test_tuple_shape_is_treated_as_scalar(self) -> None:
        content = self._pair(("m",), "m.mod_x_tuple")
        assert re.search(r"\bdouble B\b", content)
        assert "const double * restrict B" not in content
        assert re.search(r"C\[i\] = A\[i\] \* B;", content)
        assert "B[i]" not in content

    def test_two_kernel_corpus_signatures_and_bodies(self) -> None:
        node1 = _mk_node(
            "element_mul", [("A", "array"), ("B", None)], [("C", "array")], node_id="m.mod_x_k1"
        )
        node2 = _mk_node(
            "element_mul", [("A", "array"), ("B", ("m",))], [("C", "array")], node_id="m.mod_x_k2"
        )
        contents = _render_c(_graph(node1, node2))
        assert len(contents) == 2
        for content in contents:
            assert re.search(r"\bdouble B\b", content)
            assert "B[i]" not in content
            assert "A[i] * B" in content

    def test_regular_array_input_stays_pointer(self) -> None:
        node = _mk_node(
            "element_mul", [("A", "array"), ("B", "array")], [("C", "array")], node_id="m.mod_x_ptr"
        )
        content = _render_c(_graph(node))[0]
        assert "const double * restrict B" in content
        assert "B[i]" in content


# ── Deterministic enumeration (regression from c7faa6b) ──────────────────────


class TestDeterministicOrdering:
    def test_repeat_extract_produces_identical_snapshots(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")
        r1 = runner.invoke(cli_mod.main, ["extract", fixtures_dir, "-o", out1])
        r2 = runner.invoke(cli_mod.main, ["extract", fixtures_dir, "-o", out2])
        assert r1.exit_code == 0
        assert r2.exit_code == 0

        snap1 = _snapshot(out1)
        snap2 = _snapshot(out2)
        assert list(snap1.keys()) == list(snap2.keys())
        for rel, content in snap1.items():
            assert content == snap2[rel], f"difference in {rel}"

        c1 = sorted(p for p in snap1 if p.endswith(".c"))
        c2 = sorted(p for p in snap2 if p.endswith(".c"))
        assert c1 == c2
        assert len(c1) >= 1
        for cf in c1:
            header1 = re.search(r"^ \* ORIGIN FILE:.*$", snap1[cf], re.MULTILINE)
            header2 = re.search(r"^ \* ORIGIN FILE:.*$", snap2[cf], re.MULTILINE)
            assert header1 and header2
            assert header1.group(0) == header2.group(0)

    def test_ascii_sorted_file_discovery_drives_origin_lines(
        self, runner: CliRunner, tmp_path: object
    ) -> None:
        src = tmp_path / "corpus"
        src.mkdir()
        (src / "b_zz.py").write_text(
            "import numpy as np\n\ndef kernel_b(x):\n    return np.multiply(x, 2.0)\n",
            encoding="utf-8",
        )
        (src / "a_aa.py").write_text(
            "import numpy as np\n\ndef kernel_a(x):\n    return np.add(x, 1.0)\n",
            encoding="utf-8",
        )
        out = str(tmp_path / "out")
        result = runner.invoke(cli_mod.main, ["extract", str(src), "-o", out])
        assert result.exit_code == 0

        prov_files = sorted(Path(out).rglob("*.prov.json"))
        assert len(prov_files) == 2
        entries = []
        for pf in prov_files:
            prov = json.loads(pf.read_text(encoding="utf-8"))
            entries.append((prov["source"]["symbol"], prov["source"]["line"]))
        entries.sort(key=lambda t: t[1])
        assert entries[0][0].endswith("kernel_a")
        assert entries[1][0].endswith("kernel_b")
        assert entries[0][1] < entries[1][1]

        c_files = sorted(Path(out).rglob("*.c"))
        lines = []
        for cf in c_files:
            m = re.search(
                r"^ \* ORIGIN FILE:.*:(\d+)$", cf.read_text(encoding="utf-8"), re.MULTILINE
            )
            lines.append((cf.name, int(m.group(1)) if m else -1))
        lines.sort(key=lambda t: t[1])
        assert lines[0][1] < lines[1][1]
        assert "kernel_a" in lines[0][0]
        assert "kernel_b" in lines[1][0]


# ── Generated corpus compiles under strict gcc ──────────────────────────────


RequiresGcc = pytest.mark.skipif(
    shutil.which("gcc") is None, reason="gcc not available on this host"
)


@RequiresGcc
class TestGeneratedCCompiles:
    CORPUS = (
        "import numpy as np\n"
        "\n"
        "def dot_kernel(a, b):\n"
        "    return np.dot(a, b)\n"
        "\n"
        "def add_kernel(a, b):\n"
        "    return np.add(a, b)\n"
        "\n"
        "def cos_kernel(x):\n"
        "    return np.cos(x)\n"
        "\n"
        "def sum_kernel(x):\n"
        "    return np.sum(x)\n"
    )

    def test_compiles_under_strict_gcc(self, tmp_path: object) -> None:
        gen_result, _, _ = _run_pipeline(self.CORPUS)
        out = str(tmp_path / "gen")
        gen_result.write_all(out)

        gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        assert gcc.returncode == 0

        c_files = sorted(Path(out).rglob("*.c"))
        assert len(c_files) >= 3
        for cf in c_files:
            proc = subprocess.run(
                ["gcc", "-std=c99", "-Wall", "-Wextra", "-fsyntax-only", str(cf)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert proc.returncode == 0, f"{cf}:\n{proc.stderr}"
