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
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer
from purce.verifier.fuzzer import DifferentialFuzzer
from purce.verifier.z3_verifier import Z3Verifier

logger = logging.getLogger("purce")


def _resolve_entry_points(graph) -> list[str]:
    """Collect all node IDs from the graph as entry points."""
    return list(graph.nodes.keys())


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


def _process_source(
    source_dir: str, target: str, output: str, embed_assets: bool,
    amalgamate: bool, verbose: bool, provenance_only: bool = False,
    verify: bool = False, quiet: bool = False, command_name: str = "purce",
) -> None:
    """Shared logic for extract and compile commands."""
    _configure_logging(verbose, quiet)
    click.echo(f"{command_name}: {source_dir} -> {output} (target: {target})")

    source_path = Path(source_dir)
    # Sort for deterministic enumeration order: file order feeds source
    # concatenation, which sets origin-line offsets in generated kernels.
    py_files = sorted(source_path.rglob("*.py"), key=lambda p: str(p))

    if not py_files:
        click.echo("Error: No Python files found in source directory", err=True)
        sys.exit(1)

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

        with open(py_file, "r", encoding="utf-8") as f:
            combined_source += f.read() + "\n"

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

    entry_points = _resolve_entry_points(graph)

    slicer = SemanticSlicer(embed_assets=embed_assets)
    slice_result = slicer.slice(graph, entry_points)

    if slice_result.data_assets and not embed_assets:
        click.echo(
            f"Error: {len(slice_result.data_assets)} data assets found but --embed-assets not set",
            err=True,
        )
        sys.exit(1)

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

    click.echo(f"Output written to {output}/")
    click.echo(f"  {c_count} .c files")
    click.echo(f"  {h_count} .h files")
    click.echo(f"  {prov_count} .prov.json files")
    click.echo("  1 CMakeLists.txt")

    if verify:
        click.echo("\nRunning verification...")
        verifier = Z3Verifier()
        reports = verifier.verify_graph(slice_result.graph)
        verified = sum(1 for r in reports.values() if r.all_verified)
        click.echo(f"  Z3 verification: {verified}/{len(reports)} nodes verified")

        fuzzer = DifferentialFuzzer()
        fuzz_results = fuzzer.fuzz_all(iterations=200)
        for op, result in fuzz_results.items():
            status = "PASS" if result.all_passed else "FAIL"
            click.echo(f"  Fuzz {op}: {status} ({result.passed}/{result.iterations})")

    click.echo("\nDone.")


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
@click.pass_context
def extract(ctx: click.Context, source_dir: str, target: str, output: str, embed_assets: bool,
            amalgamate: bool, verbose: bool, provenance_only: bool) -> None:
    """Library extraction mode: extract math kernels from a directory."""
    _process_source(
        source_dir=source_dir, target=target, output=output,
        embed_assets=embed_assets, amalgamate=amalgamate,
        verbose=verbose or ctx.obj["verbose"], provenance_only=provenance_only,
        quiet=ctx.obj["quiet"], command_name="purce extract",
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
@click.pass_context
def compile(ctx: click.Context, source_dir: str, target: str, output: str, embed_assets: bool,
            amalgamate: bool, verbose: bool, verify: bool) -> None:
    """User code compilation mode: compile a Python project to C99."""
    _process_source(
        source_dir=source_dir, target=target, output=output,
        embed_assets=embed_assets, amalgamate=amalgamate,
        verbose=verbose or ctx.obj["verbose"], verify=verify,
        quiet=ctx.obj["quiet"], command_name="purce compile",
    )


@main.command()
@click.option("--iterations", type=int, default=10000,
              help="Number of fuzzing iterations per operation")
@click.option("--seed", type=int, default=None,
              help="Random seed for reproducible fuzzing")
@click.pass_context
def verify(ctx: click.Context, iterations: int, seed: int | None) -> None:
    """Run verification suite: Z3 bounds checking + differential fuzzing."""
    _configure_logging(ctx.obj["verbose"], ctx.obj["quiet"])
    click.echo(f"purce verify: running {iterations} iterations per operation")

    verifier = Z3Verifier()
    if verifier.available:
        click.echo("Z3 solver: available")
    else:
        click.echo("Z3 solver: not available (install z3-solver)")

    fuzzer = DifferentialFuzzer(seed=seed)
    click.echo(f"\nRunning differential fuzzing ({iterations} iterations)...")
    results = fuzzer.fuzz_all(iterations=iterations)

    all_passed = True
    for op, result in results.items():
        status = "PASS" if result.all_passed else "FAIL"
        if not result.all_passed:
            all_passed = False
        click.echo(f"  {op:20s} {status} ({result.passed}/{result.iterations} passed)")

    if all_passed:
        click.echo("\nAll operations passed differential fuzzing.")
    else:
        click.echo("\nSome operations failed. Review output above.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
