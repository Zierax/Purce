"""Tier-R M2: smoke compile of the runtime C source.

Skipped automatically when no C compiler is available (e.g. gcc missing).
Compiles with -std=c99 -Wall -Werror; that is the correctness gate for the
runtime text itself, independent of the emitter.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from purce.runtime.runtime_c import RUNTIME_C_SOURCE


def find_cc():
    for cc in ("gcc", "clang", "cc"):
        if shutil.which(cc):
            return cc
    return None


class TestRuntimeSmokeCompile(unittest.TestCase):
    def test_runtime_compiles(self):
        cc = find_cc()
        if cc is None:
            self.skipTest("no C compiler available")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "runtime.c"
            obj = Path(d) / "runtime.o"
            src.write_text(RUNTIME_C_SOURCE, encoding="utf-8")
            proc = subprocess.run(
                [cc, "-std=c99", "-Wall", "-Werror", "-O2", "-c",
                 str(src), "-o", str(obj)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(
                proc.returncode, 0,
                msg=f"runtime C source failed to compile:\n{proc.stdout}\n{proc.stderr}",
            )


if __name__ == "__main__":
    unittest.main()