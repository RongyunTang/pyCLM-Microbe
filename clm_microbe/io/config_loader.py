# load_config_auto.py
from __future__ import annotations
from dataclasses import dataclass, make_dataclass, field, is_dataclass
from typing import Any, Dict, List
from pathlib import Path
import yaml
import os
# ---------- tiny type inference ----------
def _infer_scalar_type(x: Any):
    if isinstance(x, bool):  return bool
    if isinstance(x, int):   return int
    if isinstance(x, float): return float
    return str  # fallback

def _infer_list_type(xs: list[Any]):
    # try float -> if any int/float treat as float; else infer per first elem
    if all(isinstance(x, (int, float)) for x in xs):
        return List[float]
    if xs and isinstance(xs[0], str):
        return List[str]
    return List[Any]

def make_layer_class(example: Dict[str, Any]):
    fields_def = []
    for k, v in (example or {}).items():
        ann = _infer_scalar_type(v) if not isinstance(v, list) else _infer_list_type(v)
        fields_def.append((k, ann, field()))
    return make_dataclass("Layer", fields_def)

def make_normal_prior_class():
    return make_dataclass("NormalPrior", [("mu", float), ("sigma", float)],)


# ---------- main load ----------
@dataclass()
class ModelConfig:
    # for path configs
    site_name: str
    site_type: str
    paths: Dict[str, Any]    
    prj_dir: Dict[str, Any]  
    output_files: Dict[str, Any]
    site_list: Dict[str, Any]
    observations: Dict[str, Any]
    
    # keep flexible for sections that are heterogeneous
    layers: Dict[str, Any]      # Dict[str, Layer]
    constants: Dict[str, Any]
    initial_layer_states: Dict[str, Any]  
    parameters: Dict[str, Any]
    diffusion: Dict[str, Any]
    objectives: Dict[str, Any]
    calibration: Dict[str, Any]

    # for running configs
    no_mcmc_single_site: Dict[str, Any]
    mcmc: Dict[str, Any]
    prior_predictive: Dict[str, Any]
    priors: Dict[str, Any]      # Dict[str, NormalPrior | Any]

def load_config(path: str, site_name: str) -> ModelConfig:
    # Load all configs from YAML 
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)
        if site_name in raw['site_list']['manual']:
            site_type = 'manual'
        elif site_name in raw['site_list']['auto']:
            site_type = 'auto'
        elif site_name in raw['site_list']['manual_in_plots']:
            site_type = 'manual'
        else:
            raise ValueError(f"Site name {site_name} not found in site_list.")
    
    # yaml_dir = cfg_path.parent
    paths_section = raw.get("paths", {}) or {}
    # handle nesting like your create_config has: {'paths': {...}}
    if "paths" in paths_section and isinstance(paths_section["paths"], dict):
        paths_section = paths_section["paths"]    
    project_dir = paths_section.get("project_dir", ".")

    # --- layers: mapping[str] -> Layer dataclass ---
    layers_raw = raw.get("layers", []) or []
    if not isinstance(layers_raw, list):
        raise TypeError("Expected 'layers' to be a list of mappings.")
    Layer = make_layer_class(layers_raw[0] if layers_raw else {})
    layers: list[Any] = [Layer(**props) for props in layers_raw]

    # --- priors: auto-wrap {mu,sigma} entries into NormalPrior ---
    NormalPrior = make_normal_prior_class()
    priors_raw = raw.get("priors", {}) or {}
    priors: Dict[str, Any] = {}
    for k, v in priors_raw.items():
        if isinstance(v, dict) and {"mu", "sigma"} <= v.keys():
            priors[k] = NormalPrior(mu=float(v["mu"]), sigma=float(v["sigma"]))
        else:
            priors[k] = v  # keep as-is (scalar or something else)

    output_dict = raw.get("output", {}) or {}
    # output_dict={'prior_flux_file': 'results_clm/prior_predict/prior_flux_samples.csv', 'prior_profiles_file': 'results_clm/prior_predict/prior_profile_samples.csv', 'prior_flux_q_file': 'results_clm/prior_predict/prior_flux_quantiles.csv', 'prior_profiles_q_file': 'results_clm/prior_predict/prior_profile_quantiles.csv', 'posterior_flux_file': 'results_clm/posterior_predict/posterior_flux_samples.csv', 'posterior_profiles_file': 'results_clm/posterior_predict/posterior_profile_samples.csv'}
    # add site_ID_prefix to output files if not present
    site_id_prefix = f"{site_name}_"
    for key in output_dict:
            result_dir = os.path.dirname(output_dict[key])
            file_name = os.path.basename(output_dict[key]) 
            if not file_name.startswith(site_id_prefix):
                new_file_name = site_id_prefix + file_name
                output_dict[key] = os.path.join(result_dir, new_file_name)

    # create ModelConfig object 
    cfg = ModelConfig(
        # for path configs
        paths=paths_section, 
        prj_dir=project_dir,
        output_files=output_dict,
        site_list=raw['site_list'],

        # for site-specific configs
        site_name=site_name,
        site_type=site_type,
        observations=raw.get('observations', {}).get(site_type, {}).get(site_name, {}),
        
        # for model configs
        layers=layers,
        constants=raw.get("constants", {}),
        initial_layer_states=raw.get("initial_layer_states", {}),
        parameters=raw.get("parameters", {}),
        diffusion=raw.get("diffusion", {}),
        objectives=raw.get("objectives", {}),
        calibration=raw.get("calibration", {}),
        mcmc=raw.get("mcmc", {}),
        priors=priors,

        # for simulation configs
        prior_predictive=raw.get("prior_predictive", raw.get("prior_predict", {})),
        no_mcmc_single_site=raw.get("no_mcmc_single_site", {}),

    )

    _validate_config(cfg)

    # update sitename 
    cfg.no_mcmc_single_site['site_name'] = site_name 

    return cfg

def _validate_config(cfg: ModelConfig):
    if not cfg.layers:
        raise ValueError("No layers defined.")
    names = [getattr(L, "name", None) for L in cfg.layers]
    if any(n is None for n in names):
        raise ValueError("Every layer must have a 'name'.")
    if len(set(names)) != len(names):
        raise ValueError(f"Layer names must be unique; got {names}")

    for L in cfg.layers:
        poro = getattr(L, "porosity", None)
        if poro is not None and not (0.0 <= poro <= 1.0):
            raise ValueError(f"Layer {L.name} porosity must be in [0,1], got {poro}")

# ---------- quick smoke test ----------
if __name__ == "__main__":
    yaml_path = './configs/config_new_bysites.yaml'
    cfg = load_config(yaml_path, site_name="TCM07")
    print(cfg)
    
