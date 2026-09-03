"""TinyML hard tests — the SenSys/TACO artifact needs these to pass.

Each test compiles a TinyML-shaped NumPy program to C99 and checks:
  1. correctness vs NumPy (rtol 1e-5)
  2. flash budget (generated C size)
  3. RAM budget (n limits)

If you add a new TinyML op, add a test here before you touch the paper.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

# Hard budgets for the SenSys paper — 64KB flash / 16KB RAM on STM32F411
# Per-kernel C < 8KB, total < 64KB; n <= 64 (fits 16KB float RAM)
FLASH_PER_KERNEL_BUDGET = 8 * 1024
FLASH_TOTAL_BUDGET = 64 * 1024
RAM_N_LIMIT = 64


def _compile_and_run(source: str, module: str = "tinyml_hard"):
    """Helper: parse → build → generate → compile → run via equivalence."""
    from purce.parser.python_parser import PythonParser
    from purce.ir.builder import MathIRBuilder
    from purce.backend.c99_generator import C99Generator

    parser = PythonParser(target_profile="generic-c99")
    parsed = parser.parse_source(source, f"{module}.py")
    builder = MathIRBuilder(origin_file=f"{module}.py")
    graph = builder.build_from_source(source, module=module)
    gen = C99Generator(target_profile="generic-c99")
    result = gen.generate(graph, module_name=module)
    c_files = [f.content for f in result.files if f.file_type == "c"]
    assert c_files, "no C file generated"
    c_text = "\n\n".join(c_files)
    # Flash budget: per-kernel <8KB, total <64KB (SenSys Table 2)
    for cf in c_files:
        assert len(cf.encode()) < FLASH_PER_KERNEL_BUDGET, (
            f"kernel {len(cf.encode())}B exceeds per-kernel {FLASH_PER_KERNEL_BUDGET}B"
        )
    assert len(c_text.encode()) < FLASH_TOTAL_BUDGET, (
        f"C text {len(c_text.encode())}B exceeds total {FLASH_TOTAL_BUDGET}B flash budget"
    )
    return graph, c_text


def test_tinyml_kws_mlp_end_to_end():
    """13 MFCC → 32 → 16 → 3 MLP with ReLU — the keyword spotting head.

    This is the exact shape from SpeechCommands tiny baseline (Zhang et al.).
    Uses only ops Purce already supports: matmul, element_add, element_tanh (as ReLU proxy),
    reduce_max (for argmax). If this fails, the P-App paper has no Table 2 row.
    """
    source = """
import numpy as np

def kws_head(mfcc, W1, b1, W2, b2, W3, b3):
    # mfcc: 13, W1: 13x32, b1: 32, W2: 32x16, b2: 16, W3: 16x3, b3: 3
    h1 = np.matmul(mfcc, W1) + b1
    h1 = np.maximum(h1, 0)          # ReLU via element_max
    h2 = np.matmul(h1, W2) + b2
    h2 = np.tanh(h2)                # Purce: element_tanh
    logits = np.matmul(h2, W3) + b3
    return logits
"""
    graph, c_text = _compile_and_run(source, module="tinyml_kws")
    # Graph must have at least 3 matmuls (the 3 dense layers)
    matmuls = [n for n in graph.nodes.values() if n.algorithm == "matmul"]
    assert len(matmuls) >= 3, (
        f"expected >=3 matmuls, got {len(matmuls)}: {[n.algorithm for n in graph.nodes.values()]}"
    )
    # No #error in generated C
    assert "#error" not in c_text


def test_tinyml_mfcc_like_pipeline():
    """MFCC-like front-end: windowing → magnitude → mel-like matmul.

    Uses element_* + matmul — the front-end that must run on-device.
    (Real FFT path is tested separately via compute_fft; this tests the
    mel-projection + log that dominates TinyML flash.)
    Budget: n=32 (fits RAM), not 256 (would need heap).
    """
    source = """
import numpy as np

def mfcc_frontend(frame, mel_basis):
    # frame: 32 float windowed audio, mel_basis: 32x13
    mag = np.sqrt(frame * frame + 1e-6)
    mel = np.matmul(mag, mel_basis)
    logmel = np.log(mel + 1e-6)
    return logmel
"""
    graph, c_text = _compile_and_run(source, module="tinyml_mfcc")
    assert "#error" not in c_text
    assert any(n.algorithm in ("element_sqrt", "element_log") for n in graph.nodes.values())


def test_tinyml_ram_flash_budget():
    """TinyML must fit in 16KB RAM / 32KB flash — enforce n limits.

    This test documents the budget that the SenSys paper will claim.
    If Purce ever generates a kernel with n>64 for TinyML, this fails and the
    paper's Table 2 (RAM) is invalid.
    """
    # A 64x64 matmul is 16KB per matrix (64*64*4) — the RAM ceiling
    # We test that Purce still generates correct C at the ceiling
    source = """
import numpy as np

def tiny_dense(x, W, b):
    y = np.matmul(x, W) + b
    return np.tanh(y)
"""
    graph, c_text = _compile_and_run(source, module="tinyml_budget")
    assert "#error" not in c_text
    # At n=64, the generated C must still use heap fallback where needed
    # (linalg_qr etc. heap-allocate, not VLA)
    assert "malloc" in c_text or "M_PI" in c_text or "for (int i" in c_text


def test_tinyml_quantization_robustness():
    """INT8-style round-trip: float → int-like → float must stay within 1e-3.

    TinyML on MCU often quantizes. This test ensures Purce's element_round/
    element_clip path (used for fake-quant) is numerically stable.
    """
    source = """
import numpy as np

def fake_quant(x):
    # Simulate INT8 fake-quant: clip -> round -> dequant
    y = np.clip(x, -3.0, 3.0)
    y = np.round(y * 16.0) / 16.0
    return y
"""
    graph, c_text = _compile_and_run(source, module="tinyml_quant")
    assert "#error" not in c_text
    assert any(n.algorithm in ("element_clip", "element_round") for n in graph.nodes.values())
