"""Shared pytest fixtures for Purce test suite."""

from __future__ import annotations

import os
import tempfile

import pytest

from purce.backend.c99_generator import C99Generator
from purce.ir.builder import MathIRBuilder
from purce.ir.nodes import Dtype, Effect, MathIRGraph, MathIRNode
from purce.parser.python_parser import PythonParser
from purce.slicer.semantic_slicer import SemanticSlicer


@pytest.fixture
def parser():
    return PythonParser(target_profile="generic-c99")


@pytest.fixture
def builder():
    return MathIRBuilder(origin_file="test_module")


@pytest.fixture
def slicer():
    return SemanticSlicer()


@pytest.fixture
def generator():
    return C99Generator(target_profile="generic-c99")


@pytest.fixture
def tmp_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def make_node(node_id: str, algorithm: str = "element_add") -> MathIRNode:
    return MathIRNode(
        node_id=node_id,
        origin_symbol=f"test_mod.{node_id}",
        origin_file="test.py",
        origin_line=1,
        origin_commit=None,
        origin_signature="(double*, double*, double*, int) -> void",
        math_intent=f"test kernel {algorithm}",
        inputs=[
            ("input_0", Dtype.FLOAT64, "array"),
            ("input_1", Dtype.FLOAT64, "array"),
        ],
        outputs=[("output_0", Dtype.FLOAT64, "array")],
        effects=[Effect.PURE],
        algorithm=algorithm,
        stack_usage=256,
    )


def run_pipeline(source: str, module_name: str = "test_mod"):
    builder = MathIRBuilder(origin_file=module_name)
    graph = builder.build_from_source(source, module=module_name)
    slicer = SemanticSlicer()
    entry_points = list(graph.nodes.keys())
    slice_result = slicer.slice(graph, entry_points)
    generator = C99Generator(target_profile="generic-c99")
    gen_result = generator.generate(slice_result.graph, module_name=module_name)
    return gen_result, graph, slice_result
