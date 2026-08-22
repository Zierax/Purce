"""Hardened suite configuration — 3 stages, fully reproducible."""

STAGES = {
    "stage1": {
        "name": "Hardened Correctness",
        "iterations": 1000,
        "seed": 42,
        "seeds": (42, 1337),
        "sizes": [16, 64, 256, 1024, 4096],
        "description": "92 kernels x 1000 iter, sizes up to 4096, IEEE edge + chaos 50",
    },
    "stage2": {
        "name": "Brutal Scale",
        "iterations": 5000,
        "seed": 1337,
        "seeds": (42, 1337, 999),
        "sizes": [16, 64, 256, 1024, 4096, 16384],
        "description": "92 kernels x 5000 iter, sizes up to 16384, harsh corpus 25 + chaos 50",
    },
    "stage3": {
        "name": "Frontier Stress",
        "iterations": 10000,
        "seed": 999,
        "seeds": (1, 42, 1337, 999, 2024),
        "sizes": [8, 32, 128, 512, 2048, 8192, 16384],
        "description": "92 kernels x 10k iter, adversarial, VLA provocation, determinism x5, chaos 100",
    },
}

# Fixed seeds for corpus generation — ensures byte-reproducible output.
HARSH_CORPUS_SEED = 20260822
CHAOS_CORPUS_SEED = 20260823
DETERMINISM_SEEDS = [1, 2, 3, 42, 999]
ADVERSARIAL_CASES = [
    ("n_zero", {"n": 0}),
    ("n_one", {"n": 1}),
    ("n_over_64", {"n": 65}),
    ("n_huge_vla", {"n": 100000}),
    ("n_negative", {"n": -1}),
]
