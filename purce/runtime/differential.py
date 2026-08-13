"""Tier-R differential driver (M4/M5): emitted-C output vs CPython.

Runs inside WSL (or any Linux with gcc + python3 + ctypes):

    python3 -m purce.runtime.differential          # run the full corpus
    python3 -m purce.runtime.differential -k name  # one program

Protocol per program (a Python source string that uses print() as the
only observable):
  C side:    Emitter -> C99 -> gcc -shared -> ctypes -> purce_tir_run_list
             captures stdout from the C module (p_bi_print writes fd 1)
             and the harness error line (#ERROR:kind:msg) when the module
             fails.
  Python side: exec the same source in a fresh namespace, capturing
             stdout; exceptions are recorded as an error.
  Match: stdout must be identical; both sides must agree on
  success/error.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from purce.runtime.emitter import UnsupportedError, compile_source

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "runtime" / "build"
BUILD.mkdir(exist_ok=True)

HAVE_GCC = bool(subprocess.run(
    ["gcc", "--version"], capture_output=True, check=False
).returncode == 0)

# ── corpus: (name, source) — deterministic stdout only ────────────────
CORPUS: list[tuple[str, str]] = [
    ("literals", """
print(None, True, False, 0, -0, 42, -42)
print(3.5, -2.25, 0.1, 1e3, 2.0, 1e999)
print('', 'hello', "world", 'it\\'s', 'tab\\there', 'nl\\nnew')
"""),
    ("ints", """
print(17 + 25, 100 - 3, 7 * 8, 2 ** 10, 2 ** 100)
print(10 // 3, 10 % 3, -10 // 3, -10 % 3, 7 // -2, 7 % -2)
print(123456789012345678901234567890 + 1)
print(123456789012345678901234567890 * 999999999999)
print(10 / 4, 1 / 2, 7 / 2)
print(-(-(5)), +(+5), -(-5))
"""),
    ("bignum", """
print(2 ** 200)
print(2 ** 200 + 2 ** 199)
print(2 ** 200 * 3)
print(10 ** 60 // 7, 10 ** 60 % 7)
print(12345678901234567890 - 9876543210987654321)
print(-(2 ** 100))
print(5 ** 35)
"""),
    ("floats", """
print(0.1 + 0.2)
print(1.0 / 3.0)
print(2.5 * 2, 10.0 / 4, 5.5 % 2.0)
print(-0.0, 1e308 * 10.0, -1e308 * 10.0)
print(1.5e-5, 1.25e300)
"""),
    ("strings", """
s = 'hello'
print(len(s), s[0], s[4], s[-1], s[-5])
print('abc' + 'def', s * 2, s * 0)
print('a' in 'cat', 'z' in 'cat', 'z' not in 'cat')
print(str(123), str(-7), str(3.5), str(True), str(None))
print('x' == 'x', 'a' < 'b', 'b' < 'a', 'abc' <= 'abc')
t = 'héllo wörld ✓'
print(t, len(t))
"""),
    ("lists", """
a = [1, 2, 3]
print(a, len(a), a[0], a[-1])
b = a + [4, 5]
print(b, len(b))
print(2 in a, 9 in a)
a[1] = 99
print(a)
a = a + a
print(a)
print([] == [], [1] == [2], [1, 2] == [1, 2])
print([1] < [2], [1, 2] < [1, 3])
print([[1, [2, 3]], []], len([[1, [2, 3]], []]))
"""),
    ("tuples", """
t = (1, 2, 3)
print(t, len(t), t[1])
print((), (1,), (1, 2))
print(t == (1, 2, 3), t != (1, 2, 4))
print((1, 2) < (1, 3))
"""),
    ("dict", """
d = {'a': 1, 'b': 2}
print(d, len(d))
print(d['a'], d['b'])
d['c'] = 3
print(d)
print('a' in d, 'z' in d, 'z' not in d)
d = {}
print(d, len(d))
d[1] = 'one'
d[2] = 'two'
print(d)
print({} == {}, {'a': 1} == {'a': 1}, {'a': 1} == {'a': 2})
"""),
    ("compare", """
print(1 < 2 < 3, 1 < 2 < 2, 3 > 2 > 1, 3 > 2 > 3)
print(1 == 1, 1 != 2, 2 <= 2, 3 >= 4, 5 > 4)
print('a' < 'b' < 'c', 'a' < 'b' < 'a')
print(1 < 2 == 2, 1 < 2 > 1)
"""),
    ("identity", """
x = None
print(x is None, 1 is None, None is None)
print(1 is not None, None is not None)
"""),
    ("boolop", """
print(True and False, False and True, True or False, False or False)
print(0 and 1, 1 and 2, 0 or 2, 3 or 4)
print(1 and 2 and 3, 0 and 2 and 3, 0 or 2 or 3, 4 or 0 or 5)
print(1/0 if False else 42, 1 if True else 1/0)
print(False and 1/0, True or 1/0)
print(not 0, not 1, not '', not [], not None)
"""),
    ("if", """
def f(x):
    if x < 0:
        return 'neg'
    elif x == 0:
        return 'zero'
    elif x < 10:
        return 'small'
    else:
        return 'big'
print(f(-5), f(0), f(5), f(42))
print(1 if True else 2, 2 if False else 3)
"""),
    ("while", """
i = 0
s = 0
while i < 10:
    s = s + i
    i = i + 1
print(s)
i = 0
while True:
    i = i + 1
    if i > 3:
        break
print(i)
i = 0
n = 0
while i < 100:
    i = i + 1
    if i % 2 == 0:
        continue
    n = n + i
print(n)
"""),
    ("for", """
s = 0
for i in range(5):
    s = s + i
print(s)
for i in range(2, 6):
    s = s + i
print(s)
for i in range(0, 20, 3):
    print(i)
for c in 'abc':
    print(c)
for x in [1, 2, 3]:
    print(x * 2)
d = {'k1': 1, 'k2': 2}
for k in d:
    print(k)
s = 0
for i in range(10):
    if i == 4:
        break
    s = s + i
print(s)
s = 0
for i in range(10):
    if i == 2:
        continue
    s = s + i
print(s)
"""),
    ("range", """
r = range(5)
print(len(r), 2 in r, 9 in r, r[2], r[-1])
print(list(range(3)), list(range(2, 5)), list(range(0, 10, 3)))
print(len(range(2, 5)), len(range(0, 10, 3)))
"""),
    ("functions", """
def add(a, b):
    return a + b
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n - 1)
print(add(2, 3), fact(5), fact(20))
def even(n):
    if n == 0:
        return True
    return odd(n - 1)
def odd(n):
    if n == 0:
        return False
    return even(n - 1)
print(even(10), odd(10), even(7), odd(7))
def outer(x):
    y = x * 2
    def inner(z):
        return y + z
    return inner
f = outer(10)
print(f(1), f(2), outer(100)(5))
"""),
    ("closures", """
def counter():
    n = 0
    def inc():
        n = n + 1
        return n
    def get():
        return n
    return get
c = counter()
print(c())
"""),
    ("builtins", """
print(len([1, 2, 3]), len('abc'), len((1, 2)), len({'a': 1}))
print(abs(-5), abs(5), abs(-3.5))
print(max(1, 5, 2), min(1, 5, 2), max([3, 1, 2]), min([3, 1, 2]))
print(sum([]), sum([1, 2, 3]), sum(range(4)))
print(bool(0), bool(1), bool(''), bool('x'), bool([]), bool([0]))
print(int('42'), int('-17'), int(3.7), int(-3.7), int(True))
print(float('3.5'), float(7), float(-2), float(True))
print(str(1), str(1.5), str('x'), str(True), str(None))
print(list(range(4)), list('ab'), list([1, 2]))
print(range(3), len(range(3)))
"""),
    ("assert_", """
assert 1 == 1
assert True
x = 5
assert x > 3
print('passed')
"""),
    ("nested", """
s = 0
for i in range(3):
    for j in range(4):
        s = s + i * j
print(s)
def f(n):
    return n * n
t = 0
for i in range(5):
    if f(i) > 10:
        t = t + 1
    else:
        t = t + f(i)
print(t)
"""),
    ("strings_ops", """
s = 'hello world'
print(s[0:5] if False else s[6:])
"""),
    ("errors", """
try:
    pass
except Exception:
    pass
print(1 / 0)
"""),
    ("errors2", """
a = [1, 2]
print(a[5])
"""),
    ("errors3", """
d = {'a': 1}
print(d['z'])
"""),
    ("errors4", """
print(len(1))
"""),
    ("errors5", """
def f():
    return f()
f()
"""),
]

_ERROR_KINDS = {"zero", "index", "key", "type", "value", "assert"}


def run_python(program: str) -> tuple[str, str | None]:
    """Run program under CPython; return (stdout, error_kind or None)."""
    import io
    import contextlib
    buf = io.StringIO()
    kind: str | None = None
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(program, "<tir>", "exec"), {})
    except ZeroDivisionError:
        kind = "zero"
    except (IndexError, KeyError) as e:
        kind = "index" if isinstance(e, IndexError) else "key"
    except TypeError:
        kind = "type"
    except ValueError:
        kind = "value"
    except RecursionError:
        kind = "recur"
    except AssertionError:
        kind = "assert"
    except Exception:
        kind = "runtime"
    return buf.getvalue(), kind


def compile_so(program: str, tag: str) -> str | None:
    """Emit + gcc the program; return .so path or None on failure."""
    src = compile_source(program)
    so = BUILD / f"tir_{tag}.so"
    cfile = BUILD / f"tir_{tag}.c"
    cfile.write_text(src, encoding="utf-8")
    proc = subprocess.run(
        ["gcc", "-std=c99", "-Wall", "-Werror", "-O2", "-shared", "-fPIC",
         str(cfile), "-o", str(so)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"gcc failed for {tag}:\n{proc.stderr}\n")
        return None
    return str(so)


def run_so(so: str) -> tuple[str, str | None]:
    """Run the compiled module; return (stdout, error_kind or None)."""
    lib = ctypes.CDLL(so)
    lib.purce_tir_state_new.restype = ctypes.c_void_p
    lib.purce_tir_entry.restype = ctypes.c_void_p
    lib.purce_tir_run_list.restype = ctypes.c_int
    lib.purce_tir_run_list.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.c_char_p, ctypes.c_size_t]
    lib.purce_tir_state_free.argtypes = [ctypes.c_void_p]
    S = lib.purce_tir_state_new()
    ent = lib.purce_tir_entry()
    out = ctypes.create_string_buffer(1 << 16)
    line = b""
    rc = lib.purce_tir_run_list(S, ent, line, 0, out, len(out))
    lib.purce_tir_state_free(S)
    text = out.value.decode("utf-8", "replace")
    if text.startswith("#ERROR:"):
        parts = text[len("#ERROR:"):].split(":", 1)
        return "", parts[0].strip()
    return "", None


def diff_program(program: str, tag: str) -> tuple[bool, str]:
    """Full differential check of one program."""
    py_out, py_err = run_python(program)
    so = compile_so(program, tag)
    if so is None:
        return False, "gcc build failed"
    c_out, c_err = run_so(so)
    if (c_err is None) != (py_err is None):
        return False, f"error mismatch: C={c_err!r} Python={py_err!r}"
    if c_err is not None:
        return True, f"both error ({c_err})"
    if c_out != py_out:
        return False, f"stdout mismatch:\nC:      {c_out!r}\nPython: {py_out!r}"
    return True, "ok"


def main(argv: list[str]) -> int:
    if not HAVE_GCC:
        print("SKIP: no gcc (run inside WSL or install gcc)")
        return 0
    sel = argv[1:] if len(argv) > 1 else []
    failed = 0
    total = 0
    for name, program in CORPUS:
        if sel and name not in sel:
            continue
        total += 1
        ok, detail = diff_program(program, name)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}: {detail}")
    print(f"--- {total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
