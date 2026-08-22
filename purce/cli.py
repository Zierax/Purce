"""Purce CLI - Pure-C Semantic Compiler

Converts Python/NumPy code to clean, self-contained C99 codebases.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from purce import __version__
from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import Dtype, Effect, MathIRGraph, MathIRNode
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.fuzzer import DifferentialFuzzer
from purce.verifier.z3_verifier import Z3Verifier

logger = logging.getLogger("purce")

# Exit codes, documented in README:
#   0 success; 1 runtime/verification failure; 2 usage error (Click default);
#   3 verification could not be completed (e.g. no C compiler available).
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_UNVERIFIED = 3


def _resolve_entry_points(graph, entry_names: list[str] | None = None,
                          module: str | None = None) -> list[str]:
    """Resolve the slicer entry points.

    * If ``--entry`` names are given, each must be a top-level function name;
      its root node id is used as an entry point.  Everything unreachable from
      those roots is pruned by the slicer (real dead-code elimination).
    * Otherwise, fall back to the builder's call-graph roots (one per top-level
      function).  If the graph carries none, every node is used (preserving
      historical behavior for hand-built graphs).
    """
    if entry_names:
        resolved: list[str] = []
        missing: list[str] = []
        for name in entry_names:
            # The builder's entry_points are root node ids; match the
            # origin_symbol "<module>.<name>" to resolve a function name to its
            # root node.
            prefix = f"{module}." if module else ""
            matched = [nid for nid in graph.entry_points
                       if graph.nodes.get(nid) is not None
                       and graph.nodes[nid].origin_symbol == f"{prefix}{name}"]
            if matched:
                resolved.extend(matched)
            else:
                missing.append(name)
        if missing:
            raise click.ClickException(
                "Unknown entry function(s): " + ", ".join(missing)
            )
        return resolved

    if graph.entry_points:
        return list(graph.entry_points)
    return list(graph.nodes.keys())


def _try_c_backend():
    """Build a DifferentialFuzzer with a compiled C backend when possible.

    Returns (fuzzer, compiled_or_None).  When no compiler is found, returns a
    python-only fuzzer AND prints an explicit warning so the operator knows the
    differential results are self-comparison, not a real C test.
    """
    try:
        fuzzer, compiled = DifferentialFuzzer.with_c_backend()
        return fuzzer, compiled
    except Exception as e:  # noqa: BLE001 - compiler may raise TimeoutExpired, FileNotFoundError, etc.
        click.echo(f"  Warning: C backend unavailable ({e}); fuzzing will NOT test compiled C.", err=True)
        return DifferentialFuzzer(), None


def _configure_logging(verbose: bool, quiet: bool) -> None:
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def _info(quiet: bool, message: str) -> None:
    """Echo an informational line to stdout, suppressed by --quiet."""
    if not quiet:
        click.echo(message)


def _process_source(
    source_dir: str, target: str, output: str, embed_assets: bool,
    amalgamate: bool, verbose: bool, provenance_only: bool = False,
    verify: bool = False, quiet: bool = False, command_name: str = "purce",
    entry_names: list[str] | None = None,
) -> None:
    """Shared logic for extract and compile commands."""
    _configure_logging(verbose, quiet)
    _info(quiet, f"{command_name}: {source_dir} -> {output} (target: {target})")

    source_path = Path(source_dir)
    # Sort for deterministic enumeration order: file order feeds source
    # concatenation, which sets origin-line offsets in generated kernels.
    py_files = sorted(source_path.rglob("*.py"), key=lambda p: str(p))

    if not py_files:
        click.echo("Error: No Python files found in source directory", err=True)
        sys.exit(EXIT_FAILURE)

    logger.info("Found %d Python files", len(py_files))

    parser = PythonParser(target_profile=target)
    all_diagnostics = []
    total_functions = 0

    combined_source = ""
    for py_file in py_files:
        result = parser.parse_file(str(py_file))
        all_diagnostics.extend(result.diagnostics)
        total_functions += len(result.functions)

        if verbose and result.functions:
            for fn in result.functions:
                logger.debug("  Extracted: %s (%s)", fn.name, fn.algorithm)

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                combined_source += f.read() + "\n"
        except (OSError, UnicodeDecodeError) as e:
            click.echo(f"Warning: could not read {py_file}: {e}", err=True)
            continue

    if all_diagnostics and verbose:
        logger.debug("\nDiagnostics (%d):", len(all_diagnostics))
        for d in all_diagnostics:
            logger.debug("  [%s] %s:%s - %s", d.severity, d.file, d.line, d.reason)

    if total_functions == 0:
        click.echo("Warning: No math kernels found in source files")
        sys.exit(0)
    logger.info("Found %d math kernel functions", total_functions)

    builder = MathIRBuilder(origin_file=str(source_path))
    graph = builder.build_from_source(combined_source, module=source_path.name)

    entry_points = _resolve_entry_points(graph, entry_names, module=source_path.name)
    if entry_names:
        _info(quiet, f"Entry points: {len(entry_points)} node(s) from --entry")

    slicer = SemanticSlicer(embed_assets=embed_assets)
    slice_result = slicer.slice(graph, entry_points)

    if slice_result.pal_stubs:
        _info(
            quiet,
            f"Warning: {len(slice_result.pal_stubs)} reachable node(s) routed to "
            "platform stubs (pal_stubs); ensure the target platform provides them. "
            f"{', '.join(slice_result.pal_stubs)}",
        )

    if slice_result.data_assets and not embed_assets:
        click.echo(
            f"Error: {len(slice_result.data_assets)} data assets found but --embed-assets not set",
            err=True,
        )
        sys.exit(EXIT_FAILURE)

    fixed_point = target in ("bare-arm-q31", "bare-arm-q15")
    generator = C99Generator(
        target_profile=target,
        fixed_point=fixed_point,
        amalgamate=amalgamate,
    )
    gen_result = generator.generate(slice_result.graph, module_name=source_path.name)

    if not provenance_only:
        gen_result.write_all(output)
    else:
        os.makedirs(output, exist_ok=True)

    c_count = sum(1 for f in gen_result.files if f.file_type == "c")
    h_count = sum(1 for f in gen_result.files if f.file_type == "h")
    prov_count = sum(1 for f in gen_result.files if f.file_type == "prov")

    if provenance_only:
        # Only provenance metadata is produced; do not claim code files exist.
        _info(quiet, f"Manifest written to {output}/")
        _info(quiet, f"  {prov_count} .prov.json files (provenance only; no code generated)")
    else:
        _info(quiet, f"Output written to {output}/")
        _info(quiet, f"  {c_count} .c files")
        _info(quiet, f"  {h_count} .h files")
        _info(quiet, f"  {prov_count} .prov.json files")
        _info(quiet, "  1 CMakeLists.txt")

    if verify:
        _info(quiet, "\nRunning verification...")
        verifier = Z3Verifier()
        reports = verifier.verify_graph(slice_result.graph)
        verified = sum(1 for r in reports.values() if r.all_verified)
        _info(quiet, f"  Z3 verification: {verified}/{len(reports)} nodes verified")
        total_violations = 0
        for nid, report in reports.items():
            for cond in report.conditions:
                if cond.result == "SAT":
                    total_violations += 1
                    click.echo(
                        f"    VIOLATION: {nid} [{cond.name}] {cond.formula}"
                        + (f" (counterexample: {cond.counterexample})" if cond.counterexample else "")
                    )
                elif cond.result == "UNKNOWN":
                    _info(quiet, f"    UNKNOWN:   {nid} [{cond.name}] {cond.formula}")

        # Only a proven counterexample (SAT) fails the build.  UNKNOWN is
        # advisory: the property could not be decided statically (e.g. a
        # runtime guard in the generated C is invisible to the SMT solver),
        # and is covered by differential fuzzing instead.
        if total_violations:
            click.echo(f"  Z3 verification FAILED: {total_violations} condition(s) violated.", err=True)
            sys.exit(EXIT_FAILURE)

        fuzzer, compiled = _try_c_backend()
        try:
            fuzz_results = fuzzer.fuzz_all(iterations=200)
            for op, result in fuzz_results.items():
                status = "PASS" if result.all_passed else "FAIL"
                tested = "C" if result.tested_c else "python-only"
                _info(quiet, f"  Fuzz {op}: {status} ({result.passed}/{result.iterations}, {tested})")
            if compiled is None and not any(r.tested_c for r in fuzz_results.values()):
                click.echo(
                    "\nWarning: no C compiler available; differential results were "
                    "python self-comparison and do NOT verify generated C.",
                    err=True,
                )
                sys.exit(EXIT_UNVERIFIED)
            failed_ops = [op for op, r in fuzz_results.items() if not r.all_passed]
            if failed_ops:
                click.echo(
                    f"  Differential fuzzing FAILED: {', '.join(failed_ops)}",
                    err=True,
                )
                sys.exit(EXIT_FAILURE)
        finally:
            if compiled is not None:
                compiled.close()

    _info(quiet, "\nDone.")


@click.group()
@click.version_option(version=__version__, prog_name="purce")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress non-error output")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed diagnostics")
@click.pass_context
def main(ctx: click.Context, quiet: bool, verbose: bool) -> None:
    """Purce - Pure-C Semantic Compiler

    Converts Python/NumPy code to clean, self-contained C99 codebases.
    """
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--target", type=click.Choice(["generic-c99", "bare-arm-q31", "bare-arm-q15"]),
              default="generic-c99", help="Target profile for generated code")
@click.option("-o", "--output", type=click.Path(), default="out",
              help="Output directory")
@click.option("--embed-assets", is_flag=True, default=False,
              help="Embed data assets instead of erroring")
@click.option("--amalgamate", is_flag=True, default=False,
              help="Generate single .c/.h output")
@click.option("--verbose", is_flag=True, default=False,
              help="Show detailed diagnostics")
@click.option("--provenance-only", is_flag=True, default=False,
              help="Generate manifest without code")
@click.option("--entry", "entry", multiple=True, metavar="NAME",
              help="Only emit this top-level function (repeatable); unreachable code is pruned")
@click.pass_context
def extract(ctx: click.Context, source_dir: str, target: str, output: str, embed_assets: bool,
            amalgamate: bool, verbose: bool, provenance_only: bool, entry: tuple) -> None:
    """Library extraction mode: extract math kernels from a directory."""
    _process_source(
        source_dir=source_dir, target=target, output=output,
        embed_assets=embed_assets, amalgamate=amalgamate,
        verbose=verbose or ctx.obj["verbose"], provenance_only=provenance_only,
        quiet=ctx.obj["quiet"], command_name="purce extract",
        entry_names=list(entry),
    )


@main.command()
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--target", type=click.Choice(["generic-c99", "bare-arm-q31", "bare-arm-q15"]),
              default="generic-c99", help="Target profile for generated code")
@click.option("-o", "--output", type=click.Path(), default="out",
              help="Output directory")
@click.option("--embed-assets", is_flag=True, default=False,
              help="Embed data assets instead of erroring")
@click.option("--amalgamate", is_flag=True, default=False,
              help="Generate single .c/.h output")
@click.option("--verbose", is_flag=True, default=False,
              help="Show detailed diagnostics")
@click.option("--verify", is_flag=True, default=False,
              help="Run Z3 verification and differential fuzzing")
@click.option("--entry", "entry", multiple=True, metavar="NAME",
              help="Only emit this top-level function (repeatable); unreachable code is pruned")
@click.pass_context
def compile(ctx: click.Context, source_dir: str, target: str, output: str, embed_assets: bool,
            amalgamate: bool, verbose: bool, verify: bool, entry: tuple) -> None:
    """User code compilation mode: compile a Python project to C99."""
    _process_source(
        source_dir=source_dir, target=target, output=output,
        embed_assets=embed_assets, amalgamate=amalgamate,
        verbose=verbose or ctx.obj["verbose"], verify=verify,
        quiet=ctx.obj["quiet"], command_name="purce compile",
        entry_names=list(entry),
    )


@main.command()
@click.option("--iterations", type=int, default=10000,
              help="Number of fuzzing iterations per operation")
@click.option("--seed", type=int, default=None,
              help="Random seed for reproducible fuzzing")
@click.pass_context
def verify(ctx: click.Context, iterations: int, seed: int | None) -> None:
    """Run verification suite: Z3 bounds checking + differential fuzzing.

    Z3 is run over representative nodes for every algorithm family the
    compiler can emit.  Differential fuzzing compiles the C99 kernels and
    compares against the Python reference when a C compiler is available;
    otherwise it reports that C was not tested and exits with code 3.
    """
    _configure_logging(ctx.obj["verbose"], ctx.obj["quiet"])
    click.echo(f"purce verify: running {iterations} iterations per operation")

    verifier = Z3Verifier()
    if not verifier.available:
        click.echo("Z3 solver: not available (install z3-solver); verification cannot run.", err=True)
        sys.exit(EXIT_FAILURE)

    click.echo("Z3 solver: available")
    graph = _representative_verification_graph()
    reports = verifier.verify_graph(graph)

    z3_ok = True
    for nid, report in reports.items():
        for cond in report.conditions:
            marker = {
                "SAT": "VIOLATION",
                "UNSAT": "verified",
                "UNKNOWN": "unknown",
            }.get(cond.result, cond.result)
            if cond.result == "SAT":
                z3_ok = False
            click.echo(
                f"  {nid:22s} {marker:9s} {cond.name}: {cond.formula}"
                + (f" (counterexample: {cond.counterexample})" if cond.counterexample else "")
            )

    if not z3_ok:
        click.echo("\nZ3 verification FAILED: violations found.", err=True)
        sys.exit(EXIT_FAILURE)
    click.echo(f"\nZ3 verification OK ({len(reports)} nodes, all conditions UNSAT).")

    fuzzer, compiled = _try_c_backend()
    try:
        click.echo(f"\nRunning differential fuzzing ({iterations} iterations)...")
        results = fuzzer.fuzz_all(iterations=iterations)

        all_passed = True
        tested_c = False
        for op, result in results.items():
            status = "PASS" if result.all_passed else "FAIL"
            if not result.all_passed:
                all_passed = False
            tested_c = tested_c or result.tested_c
            click.echo(f"  {op:20s} {status} ({result.passed}/{result.iterations} passed)")

        if not tested_c:
            click.echo(
                "\nWarning: no C compiler available; differential results were "
                "python self-comparison and do NOT verify generated C.",
                err=True,
            )
            sys.exit(EXIT_UNVERIFIED)

        if all_passed:
            click.echo("\nAll operations passed differential fuzzing against compiled C.")
        else:
            click.echo("\nSome operations failed. Review output above.", err=True)
            sys.exit(EXIT_FAILURE)
    finally:
        if compiled is not None:
            compiled.close()


def _representative_verification_graph() -> MathIRGraph:
    """Build a representative graph covering every algorithm family the Z3
    verifier knows how to check, for the standalone `purce verify` command.
    """
    graph = MathIRGraph()

    def node(algorithm: str, inputs=None, outputs=None) -> MathIRNode:
        return MathIRNode(
            node_id=f"verify_{algorithm}",
            origin_symbol=f"verify.{algorithm}",
            origin_file="verify.py",
            origin_line=1,
            origin_commit=None,
            origin_signature="n/a",
            math_intent=f"Representative {algorithm}",
            inputs=inputs or [("x", Dtype.FLOAT64, "array")],
            outputs=outputs or [("result", Dtype.FLOAT64, "array")],
            effects=[Effect.PURE],
            algorithm=algorithm,
            stack_usage=64,
        )

    graph.add_node(node("element_add"))
    graph.add_node(node("element_sub"))
    graph.add_node(node("element_mul"))
    graph.add_node(node("element_div"))
    graph.add_node(node("matmul",
                        inputs=[("A", Dtype.FLOAT64, "(m,k)"), ("B", Dtype.FLOAT64, "(k,n)")],
                        outputs=[("C", Dtype.FLOAT64, "(m,n)")]))
    graph.add_node(node("reduce_sum"))
    graph.add_node(node("reduce_mean"))
    graph.add_node(node("reduce_max"))
    graph.add_node(node("reduce_min"))
    graph.add_node(node("linalg_solve",
                        inputs=[("A", Dtype.FLOAT64, "(n,n)"), ("b", Dtype.FLOAT64, "(n,)")],
                        outputs=[("x", Dtype.FLOAT64, "(n,)")]))
    graph.add_node(node("linalg_inv",
                        inputs=[("A", Dtype.FLOAT64, "(n,n)")],
                        outputs=[("inv", Dtype.FLOAT64, "(n,n)")]))
    graph.add_node(node("fft",
                        inputs=[("real", Dtype.FLOAT64, "array"), ("imag", Dtype.FLOAT64, "array")],
                        outputs=[("out_real", Dtype.FLOAT64, "array"), ("out_imag", Dtype.FLOAT64, "array")]))
    graph.add_node(node("ifft",
                        inputs=[("real", Dtype.FLOAT64, "array"), ("imag", Dtype.FLOAT64, "array")],
                        outputs=[("out_real", Dtype.FLOAT64, "array"), ("out_imag", Dtype.FLOAT64, "array")]))
    return graph


if __name__ == "__main__":
    main()
