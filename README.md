# pyCLM-Microbe

**Process-based soil biogeochemical model (forward simulation)**

---

## Overview

pyCLM-Microbe is a process-based simulation model for soil carbon
decomposition, microbial dynamics, and CH₄/CO₂ gas exchange. It explicitly
represents heterotrophic respiration, acetoclastic and hydrogenotrophic
methanogenesis, methanotrophy, and anaerobic oxidation of methane (AOM), with
hydrothermal drivers coupled to the biogeochemical reaction network.

This repository packages the model in a clean, installable layout:

- **`core`** — the `CLMMicrobe` process model (the simulation engine)
- **`io`** — config loading, site-data loading, and model-input preparation
- **`utils`** — unit-conversion helpers

The model is run in **forward** mode: a simulation with fixed parameters,
producing time series of CO₂/CH₄ fluxes and related state variables.

---

## Installation

From the project root:

```bash
pip install -e .
```

Requirements: `numpy`, `pandas`, `matplotlib`, `pyyaml`, `openpyxl`.

The runner script also works from a clone without installation — it adds the
project root to `sys.path` automatically.

---

## Quick Start

The repository ships sample input files for six sites under `data/inputs/`
(`MT35`, `MT71`, `T.07.C`, `T.09.C`, `T.11.C`, `U.01.C`).

### Forward simulation

```bash
# Single site
python scripts/runner_forward.py --sites MT35

# Several sites
python scripts/runner_forward.py --sites MT35 T.09.C U.01.C

# Every site listed in the config
python scripts/runner_forward.py --sites all
```

Results are written to `simulations/<scenario>/<site>_results.csv`.

If installed, the console entry point `clm-forward` mirrors the script.

---

## Python API

```python
from clm_microbe.io import load_config, read_inputs_from_excel, prepare_model_inputs
from clm_microbe.core import CLMMicrobe

cfg = load_config("configs/config_new_bysites.yaml", site_name="MT35")
inputs_dict = read_inputs_from_excel(cfg, "MT35")
inputs_tuple, cfg = prepare_model_inputs(cfg, inputs_dict)

clm = CLMMicrobe(cfg=cfg, inputs_tuple=inputs_tuple)
df = clm.run(scenario="S0_noMCMC", params=None)
print(df[[c for c in df.columns if "ch4" in c or "co2" in c]].describe())
```

---

## Package Structure

```
pyCLM-Microbe/
├── clm_microbe/
│   ├── core/model.py            # CLMMicrobe process model
│   ├── io/
│   │   ├── config_loader.py     # load_config, ModelConfig
│   │   ├── data_loader.py       # read_inputs_from_excel
│   │   └── input_builder.py     # prepare_model_inputs
│   └── utils/units.py           # unit conversions
├── configs/config_new_bysites.yaml
├── data/inputs/<SITE>_inputs.xlsx
├── scripts/
│   └── runner_forward.py
├── pyproject.toml
└── README.md
```

---

## Configuration & Input Data

`configs/config_new_bysites.yaml` defines soil layers, physical constants,
parameters, the site list, and per-site observation paths. Site forcing/
observation data are read from `data/inputs/<SITE>_inputs.xlsx`. To run your
own site, add a matching `<SITE>_inputs.xlsx` and list the site in the
config's `site_list`.

---

## Citation

Please cite the original CLM-Microbe publications and this implementation as
appropriate for your work.
