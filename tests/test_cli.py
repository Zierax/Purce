import json
import os
import shutil

import pytest
from click.testing import CliRunner

from purce.cli import main

_HAS_GCC = shutil.which("gcc") is not None or shutil.which("cc") is not None
RequiresGcc = pytest.mark.skipif(not _HAS_GCC, reason="gcc not available")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fixtures_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures")


class TestCLIExtract:
    def test_extract_basic(self, runner: CliRunner, fixtures_dir: str, tmp_path: object) -> None:
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(main, ["extract", fixtures_dir, "-o", out_dir])
        assert result.exit_code == 0
        assert "purce extract" in result.output
        assert "Done" in result.output

    def test_extract_with_verbose(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(main, ["extract", fixtures_dir, "-o", out_dir, "--verbose"])
        assert result.exit_code == 0

    def test_extract_empty_dir(self, runner: CliRunner, tmp_path: object) -> None:
        empty_dir = str(tmp_path) + "/empty"
        os.makedirs(empty_dir)
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(main, ["extract", empty_dir, "-o", out_dir])
        assert result.exit_code == 1
        assert "No Python files" in result.output

    def test_extract_provenance_only(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(main, ["extract", fixtures_dir, "-o", out_dir, "--provenance-only"])
        assert result.exit_code == 0

    def test_extract_fixed_point(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(
            main, ["extract", fixtures_dir, "-o", out_dir, "--target", "bare-arm-q31"]
        )
        assert result.exit_code == 0


class TestCLICompile:
    def test_compile_basic(self, runner: CliRunner, fixtures_dir: str, tmp_path: object) -> None:
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(main, ["compile", fixtures_dir, "-o", out_dir])
        assert result.exit_code == 0
        assert "purce compile" in result.output
        assert "Done" in result.output

    def test_compile_with_verify(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        result = runner.invoke(main, ["compile", fixtures_dir, "-o", out_dir, "--verify"])
        assert result.exit_code == 0
        assert "Running verification" in result.output
        assert "Done" in result.output


class TestCLIVerify:
    @RequiresGcc
    def test_verify_basic(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["verify", "--iterations", "50"])
        assert result.exit_code == 0
        assert "purce verify" in result.output
        assert "PASS" in result.output


class TestCLIVersion:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __import__("purce").__version__ in result.output


class TestCLIOutputStructure:
    def test_output_has_all_files(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        runner.invoke(main, ["extract", fixtures_dir, "-o", out_dir])

        files = []
        for root, dirs, fnames in os.walk(out_dir):
            for f in fnames:
                files.append(os.path.join(root, f))

        c_files = [f for f in files if f.endswith(".c")]
        h_files = [f for f in files if f.endswith(".h")]
        prov_files = [f for f in files if f.endswith(".prov.json")]
        cmake_files = [f for f in files if f.endswith("CMakeLists.txt")]

        assert len(c_files) >= 1
        assert len(h_files) == 1
        assert len(prov_files) >= 1
        assert len(cmake_files) == 1

    def test_provenance_json_valid(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        runner.invoke(main, ["extract", fixtures_dir, "-o", out_dir])

        for root, dirs, fnames in os.walk(out_dir):
            for f in fnames:
                if f.endswith(".prov.json"):
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as fh:
                        prov = json.load(fh)
                    assert "source" in prov
                    assert "ir_node" in prov
                    assert "memory" in prov

    def test_c99_files_have_standard_headers(
        self, runner: CliRunner, fixtures_dir: str, tmp_path: object
    ) -> None:
        out_dir = str(tmp_path) + "/out"
        runner.invoke(main, ["extract", fixtures_dir, "-o", out_dir])

        for root, dirs, fnames in os.walk(out_dir):
            for f in fnames:
                if f.endswith(".c"):
                    path = os.path.join(root, f)
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                    assert "PURCE OUTPUT" in content
                    assert "#include <stdint.h>" in content
                    assert "#include <math.h>" in content
