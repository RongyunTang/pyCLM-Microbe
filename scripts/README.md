# scripts/

Entry point for running pyCLM-Microbe. Invoke from the project root.

| Script | Purpose |
|--------|---------|
| `runner_forward.py` | Forward simulation with fixed parameters |

## runner_forward.py

```bash
python scripts/runner_forward.py --sites MT35 T.09.C
python scripts/runner_forward.py --sites all --quiet
```

Flags: `--scenario` (output label), `--config`, `--output`, `--quiet`, `--verbose`.

Outputs (under `simulations/<scenario>/`): `<site>_results.csv`.
