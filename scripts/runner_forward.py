"""
runner_forward.py — Forward (no-calibration) runner for CLM-Microbe.

Runs the ``CLMMicrobe`` model forward with fixed parameters for one or more
sites and writes the full state-variable time series to CSV. Use this for
baseline simulations, hypothesis testing, and sensitivity exploration.

Pipeline (per site)::

    load_config -> read_inputs_from_excel -> prepare_model_inputs
        -> CLMMicrobe(cfg, inputs_tuple).run(scenario, params=None)

Usage examples
--------------
# Single site
    python scripts/runner_forward.py --sites MT35

# Several sites
    python scripts/runner_forward.py --sites MT35 T.09.C U.01.C

# All sites listed in the config's site_list
    python scripts/runner_forward.py --sites all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT       = _SCRIPT_DIR.parent                       # project root
_DEFAULT_CONFIG = _ROOT / "configs" / "config_new_bysites.yaml"
_DEFAULT_OUTPUT = _ROOT / "simulations"
_INPUT_DIR      = _ROOT / "data" / "inputs"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _resolve_sites(tokens: list[str], config_path: Path) -> list[str]:
    """Expand the 'all' alias into the full site_list from the config."""
    if tokens != ["all"]:
        return tokens
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    sl = raw.get("site_list", {})
    return list(sl.get("manual", []) + sl.get("auto", []) + sl.get("manual_in_plots", []))


def run_site(site: str, config_path: Path, scenario: str, output_dir: Path,
             verbose: bool) -> str:
    """Run one forward simulation; return a short status string."""
    from clm_microbe.io import load_config, read_inputs_from_excel, prepare_model_inputs
    from clm_microbe.core import CLMMicrobe

    xlsx = _INPUT_DIR / f"{site}_inputs.xlsx"
    if not xlsx.exists():
        return f"SKIP {site} (missing {xlsx.name})"

    cfg = load_config(str(config_path), site_name=site)
    inputs_dict = read_inputs_from_excel(cfg, site)
    inputs_tuple, cfg = prepare_model_inputs(cfg, inputs_dict)

    clm = CLMMicrobe(cfg=cfg, inputs_tuple=inputs_tuple)
    results_df = clm.run(scenario=scenario, params=None)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{site}_results.csv"
    results_df.to_csv(out_csv, index=False)
    return f"OK   {site} -> {out_csv.name}  ({results_df.shape[0]} rows)"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="runner_forward",
        description="Forward (no-calibration) CLM-Microbe runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sites", "-s", nargs="+", required=True, metavar="SITE",
                   help="Site IDs (e.g. MT35 T.09.C) or 'all' for every site "
                        "in the config's site_list. Each needs a matching "
                        "data/inputs/<SITE>_inputs.xlsx file.")
    p.add_argument("--scenario", default="S0_noMCMC",
                   help="Scenario label passed to CLMMicrobe.run() and used as "
                        "the output sub-directory name.")
    p.add_argument("--config", "-c", type=Path, default=_DEFAULT_CONFIG,
                   help="Path to the YAML configuration file.")
    p.add_argument("--output", "-o", type=Path, default=_DEFAULT_OUTPUT,
                   help="Output directory root (results go to <output>/<scenario>/).")
    p.add_argument("--quiet", action="store_true", default=False,
                   help="Suppress the model's verbose stdout during each run.")
    p.add_argument("--verbose", "-v", action="store_true", default=False,
                   help="Print full tracebacks on failure.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore")

    sites = _resolve_sites(args.sites, args.config)
    out_dir = args.output / args.scenario
    print(f"CLM-Microbe forward run | scenario={args.scenario} | {len(sites)} site(s)")

    import contextlib, io, traceback
    n_ok = 0
    for site in sites:
        try:
            if args.quiet:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    msg = run_site(site, args.config, args.scenario, out_dir, args.verbose)
            else:
                msg = run_site(site, args.config, args.scenario, out_dir, args.verbose)
            if msg.startswith("OK"):
                n_ok += 1
            print(f"  {msg}")
        except Exception as exc:                       # noqa: BLE001 — report and continue
            print(f"  ERR  {site}: {exc}")
            if args.verbose:
                traceback.print_exc()

    print(f"Done — {n_ok}/{len(sites)} site(s) written to {out_dir}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
