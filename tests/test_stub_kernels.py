"""Regression for stub kernels: they must not claim verification."""
from purce.backend.c99_generator import C99Generator, MATH_KERNEL_BODIES, _STUB_ALGORITHMS
from purce.ir.builder import MathIRBuilder


# Minimal Python snippets that trigger each stub algorithm via the real parser/builder.
_STUB_SOURCES: dict[str, str] = {
    "linalg_eig": "import numpy as np\ndef f(A):\n    return np.linalg.eig(A)\n",
    "linalg_qr": "import numpy as np\ndef f(A):\n    return np.linalg.qr(A)\n",
    "linalg_svd": "import numpy as np\ndef f(A):\n    return np.linalg.svd(A)\n",
}

# A known-good non-stub for positive control.
_GOOD_SOURCE = "import numpy as np\ndef f(a, b):\n    return np.add(a, b)\n"


def _generate_for_source(source: str) -> str:
    builder = MathIRBuilder(origin_file="test_stub.py")
    graph = builder.build_from_source(source, module="testmod")
    # Filter to the node matching the expected algorithm when multiple nodes exist
    gen = C99Generator(target_profile="generic-c99")
    result = gen.generate(graph, module_name="testmod")
    c_files = [f for f in result.files if f.file_type == "c"]
    assert len(c_files) >= 1
    # Prefer the file whose node_id contains the stub algo name
    return c_files[0].content


def _generate_for_algorithm(algo: str) -> str:
    src = _STUB_SOURCES.get(algo, _GOOD_SOURCE)
    return _generate_for_source(src)


class TestStubKernelsAreMarkedUnverified:
    def test_stub_set_is_expected(self):
        assert _STUB_ALGORITHMS == frozenset(
            {"linalg_eig", "linalg_qr", "linalg_svd"}
        )

    def test_each_stub_body_contains_warning(self):
        for algo in _STUB_ALGORITHMS:
            content = _generate_for_algorithm(algo)
            assert "WARNING: stub implementation" in content, f"{algo} missing stub warning"
            assert algo in content

    def test_each_stub_header_is_unverified(self):
        for algo in _STUB_ALGORITHMS:
            content = _generate_for_algorithm(algo)
            assert "Verified: NO" in content, f"{algo} should be marked unverified"
            assert "unverified (stub)" in content

    def test_each_stub_does_not_claim_verified(self):
        for algo in _STUB_ALGORITHMS:
            content = _generate_for_algorithm(algo)
            # The verified fuzzing line must not appear for stubs
            assert "Verified: differential fuzzing (10k iterations)" not in content

    def test_non_stub_still_verified(self):
        content = _generate_for_algorithm("element_add")
        assert "Verified: differential fuzzing (10k iterations)" in content
        assert "WARNING: stub implementation" not in content
        assert "Verified: NO" not in content

    def test_stub_bodies_exist_in_registry(self):
        for algo in _STUB_ALGORITHMS:
            assert algo in MATH_KERNEL_BODIES
