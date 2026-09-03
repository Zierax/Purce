# reallyhardones

Journal / conference-grade hard tests. Not for `pytest -q` on every commit — for
`hardened` + paper artifact.

- **TinyML** is first. Each test compiles a full TinyML pipeline (MFCC → dense → softmax)
  to **self-contained C99** via `purce` and runs it via the `hardened` harness.
- Budgets are real: `flash < 32KB`, `RAM < 16KB`, `latency < 20ms` on `cortex-m4` (simulated by `n` limits).
- If a test here fails, `L8` in `docs/limitations.md` grows. If it passes, a table in the paper gets a row.

Run:

```powershell
python -m pytest tests/reallyhardones -q              # all hard tests
python -m pytest tests/reallyhardones/test_tinyml_hard.py -q -k kws
python -m benchmarks.hardened.run_all_hardened          # still the gate (91/91)
```
