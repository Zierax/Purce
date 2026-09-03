"""Tier-R runtime assembly: concatenate parts P1..P7 into one C99 source.

The runtime is a single translation unit compiled with
``gcc -std=c99 -Wall -Werror`` (see test_runtime_compile.py). The emitter
(purce/runtime/emitter.py) generates C that calls exactly the functions
defined here.
"""

from purce.runtime.c_parts import (
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
)

RUNTIME_TAIL = """
/* end of Tier-R runtime */
"""

RUNTIME_C_SOURCE = f"{P1}\n{P2}\n{P3}\n{P4}\n{P5}\n{P6}\n{P7}\n{RUNTIME_TAIL}".rstrip() + "\n"
