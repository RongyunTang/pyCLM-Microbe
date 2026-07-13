"""clm_microbe.core.model — CLM-Microbe process model.

The ``CLMMicrobe`` class is the hourly time-stepping biogeochemical simulator
(soil physics, decomposition, microbial methane cycling, gas transport).
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

R8 = np.float64
RGAS = 8.3145       # J/mol.K
RGAS_LATM = 0.0821  # L.atm/mol.K


class CLMMicrobe:
    def __init__(self, inputs_tuple, cfg):
        """Initialize the model with default parameters and state variables"""
        # Inputs and configuration
        self.inputs, self.for_calibration = inputs_tuple
        self.for_calibration = self.for_calibration.reset_index(drop=True) 
        self.inputs = self.inputs.reset_index(drop=True) 
        self.inputs['flux_ch4'] = self.inputs['flux_ch4'] * 1e-3 # convert from umol/m2/s to mmol/m2/s
        self.inputs['flux_co2'] = self.inputs['flux_co2'] * 1e-3 # convert from umol/m2/s to mmol/m2/s
        if any(self.inputs['SM'] > 1.0):
            self.inputs['SM'] = self.inputs['SM'] / 100.0 # convert from % to fraction if needed 
        if any(self.inputs['ST'] > 200.0):
            self.inputs['ST'] = self.inputs['ST'] - 273.15 # convert from Kelvin to Celsius if needed
        self.inputs['year'] = self.inputs['datetime'].dt.year
        self.cfg = cfg # Configuration dictionary for model settings 
        self.initialized = False
        self.site_drainage = (
            self.inputs['drainage'].iloc[0]
            if 'drainage' in self.inputs.columns and len(self.inputs) > 0
            else 'unknown'
        )
        self.params_all = self.cfg.parameters
        self.constants = self.cfg.constants
        self.nlevsoil = len(self.constants['CLM_layers_depth_cm']) # number of soil layers
        self.site_name = self.cfg.site_name
        
        ## add for calibration 
        self.alpha_O2 = self.params_all['alpha_O2']
        self.alpha_hydrogen = self.params_all['alpha_hydrogen']

        # Model parameters and state variables
        self.timestep = 1
        self.nr = len(self.inputs) # number of records
        self.scaler = 1e3 # convert to mol/m3 for initalization of atmospheric gases and kinetic parameters 

        """Set default parameter values"""
        self.watsat = 0.98 # original 0.5 not used; 
        self.soilvolume = 0.0001125
        self.bottlevolume = 0.00025
        self.sand = 25.0
        self.clay = 5.0
        self.silt = 10.0
        self.organic = 60.0
        self.initialaommethanotrophs = 0.0
        self.ACConcentration = 5000 # initial acetate concentration (mol/m3)

        # Microbial parameters with better organization
        self._init_microbial_parameters()
        self._init_soil_decomp_parameters() # decomposition base rates

        # Decomposition parameters — C:N ratios of the soil organic matter pools.
        self.cn_s1 = 12.0 # C:N ratio of soil organic matter pool 1
        self.cn_s2 = 12.0 # C:N ratio of soil organic matter pool 2
        self.cn_s3 = 10.0 # C:N ratio of soil organic matter pool 3
        self.cn_s4 = 10.0 # C:N ratio of soil organic matter pool 4

        # Respiration fractions
        self.rf_l1s1 = 0.39 # respiration fraction litter 1 -> SOM 1
        self.rf_l2s2 = 0.55 # respiration fraction litter 2 -> SOM 2
        self.rf_l3s3 = 0.29 # respiration fraction litter 3 -> SOM 3
        self.rf_s1s2 = 0.28 # respiration fraction SOM 1 -> SOM 2
        self.rf_s2s3 = 0.46 # respiration fraction SOM 2 -> SOM 3
        self.rf_s3s4 = 0.55 # respiration fraction SOM 3 -> SOM 4

        # CWD fractions
        self.cwd_fcel = 0.76 # cellulose fraction of coarse woody debris
        self.cwd_flig = 0.24 # lignin fraction of coarse woody debris

        # Denitrification proportion
        self.dnp = 0.01 # denitrification proportion (loss of N as N2O)
    
        # File names
        self.initialfile = ""
        self.soilparafile = ""
        self.microbeparfilename = ""
        self.outputfile = ""
        
        # State variables arrays
        self._init_arrays()
        
        # Flags and scalars
        self.FlagpH = 1
        self.FlagO2 = 1
    
        # add parameters for calibration 
        self.soilsample_scalar = 1
        self.O2airfrac = 0.2095 * np.ones(self.nr)   # O2 fraction (209,000 ppm)
        self.CO2airfrac = 4.15e-4 * np.ones(self.nr) # CO2 fraction (~415 ppm * e-6 to fraction)
        self.CH4airfrac = 1.8e-6 * np.ones(self.nr)  # CH4 fraction (~1.8 ppm * e-6 to fraction)
        self.H2airfrac = 5.0e-7 * np.ones(self.nr)   # H2 fraction (~0.5 ppm * e-6 to fraction)
        
        # anaerobic threshold for soluble m_dO2 (1umol/L = 1e-3 mol/m3)
        # decomp_basescale is disabled: the original CLM-Microbe base rates are
        # used, and the drainage-class correction is absorbed by the HR Q10
        # reference temperature instead.  CH4 pathways use a separate reference
        # temperature (T_ref_ch4 = 13.5°C) from decomposition.
        self.decomp_basescale = 1.0  # disabled; kept for interface compatibility
        self.T_ref_ch4 = 13.5        # reference temperature for all CH4 pathway Q10s
        self.porosity_scaler = 1.0 # scaling factor for soil porosity
        # Troeh f_D floor: 0.0 = pure aerobic model (saturated wetlands are
        # underestimated — a known limit of the aerobic-only framework).
        self.troeh_floor_fd = 0.0

        self.anearobic_threshold = 1e-3  # mol/m3, dissolved O2 threshold for anaerobic condition
        # AFP-based anaerobic gate: gate = K_AFP / (K_AFP + afp); gate→1 at saturation.
        self.K_AFP_inhib_methanogens = 0.02  # m³/m³, AFP at half-inhibition (~near-saturated threshold)
        self.wetland_list = []
        self.soil_depth = 0.05  # assuming organic soil layer of 0.1 m depth
        self.timestep = 1
        self.opt_vwc = 0.4  # optimal WFPS for microbial activity
        self.params_all['optimal_vwc'] = self.opt_vwc # add to params_all for MCMC access
        self.maxi_porosity  = 1.0   # theoretical max; effective porosity = maxi_porosity * porosity_scaler

        self.oxi_q10 = 3.0  # Q10 for oxidation reactions 
        self.decomp_q10 = 2.5  # Q10 for decomposition reactions
        self.opt_min_vwc = 0.15  # minimum WFPS for decomposition activity 0.45
        self.opt_max_scale = 0.5
        self.vwc_opt_max = 0.2    # optimal_max WFPS for maximum oxidation activity
        self.vwc_opt_min = 0.1     # optimal_min WFPS for maximum oxidation activity
        self.vwc_opt_max_scale = 0.5
        self.vwc_max = 1.0   # theoretical maximum WFPS; opt_max_vwc = opt_max_scale * 1.0

    def _init_microbial_parameters(self):
        """Initialize microbial parameters.

        Uses the hardcoded defaults below unless
        ``parameters.use_config_microbe_params`` is set, in which case values are
        read from ``cfg.parameters.microbe_par_dict`` (missing keys fall back to
        the hardcoded default).
        """
        # ── Hardcoded production defaults (the values that reproduce the manuscript) ──
        hardcoded_params = {
            'm_dKAce': 15 * 1e-3, # mmol/m3 -> mol/m3,  Half-saturation coefficient of available carbon mineralization

            'm_dAceProdACmax': 0.005, # mmol/m3/h -> mol/m3/s; acetate production rate from available carbon
            'm_dKAceProdO2': 0.001, # mmol/m3 -> mol/m3

            'm_dKCO2ProdAce': 8.25e-3 * 1e2, # 0.00825 μmol/m3 to mol/m3
            'm_dKH2ProdAce': 1.65e-3 * 1e2, # 0.0165 μmol/m3 (1.65e-2 umol/m3) to mol/m3
            'm_dH2ProdAcemax': 1.35e-3 * 1e2, # 0.01 1.1–12.4 mmol acetate/g/h -> mol acetae/g/s | Maximum reaction rate of conversion of H2 and CO2 to acetate

            'm_dGrowRH2Methanogens': 0.012, # d-1 | Growth rate of H2-CO2-dependent methanogens, balanced with death (0.007)
            'm_dDeadRH2Methanogens': 0.007, # d-1 -> s-1 | Death rate of H2-CO2-dependent methanogens, original paper value in unit -h
            'm_dYH2Methanogens': 0.1, # mol C(mol CO2-C) -1| Growth efficiency of H2-CO2-dependent methanogens

            'm_dGrowRAceMethanogens': 0.005, # d-1 | Growth rate of acetoclastic methanogens, balanced with death
            'm_dDeadRAceMethanogens': 0.001, # d-1 -> s-1 | Death rate of acetoclastic methanogens, original paper value in unit -h
            'm_dYAceMethanogens': 0.2, # mol C(mol acetate-C) -1 | Growth efficiency of acetoclastic methanogens

            # At factor=1 (saturated), growth = 0.002 = death, so methanotroph
            # biomass declines under normal aerobic conditions and grows only in
            # favorable bursts.
            'm_dGrowRMethanotrophs': 0.002, # d-1 | Growth rate of methanotrophs
            'm_dDeadRMethanotrophs': 0.002, # d-1 -> s-1 | Death rate of methanotrophs, original paper value in unit -h
            'm_dYMethanotrophs': 0.15,

            'm_dGrowRAOMMethanotrophs': 0.024, # d-1 -> s-1 | Growth rate of AOM methanotrophs, original paper value in unit -h
            'm_dDeadRAOMMethanotrophs': 0.002, # d-1 -> s-1 | Death rate of AOM methanotrophs, original paper value in unit -h
            'm_dYAOMMethanotrophs': 0.40,

            'm_dAceProdQ10': 2.0, # | teamperature sensitivity Q10 for acetate production
            'm_dACProdQ10': 1.5, # | teamperature sensitivity Q10 for available carbon mineralization
            'm_dACMinQ10': 1.5, # | teamperature sensitivity Q10 for available carbon mineralization
            'm_dAceH2min': 0.48e-3 * 1e-9,  # nmol/m3 ->mol/m3 | Minimum concentration of acetic acid production from H2
            'm_dCH4H2min': 0.27e-4 * 1e-9, # nmol/m3 ->mol/m3 | Minimum concentration of CH4 production from H2
            
            'm_dKH2ProdCH4': 7.75e-3 * 1e-6, # 7.75E-6 mmol/m3 -> mol/m3 | Half coefficient of H2 for methane production from H2
            'm_dKCO2ProdCH4': 1.98e-3 * 1e-8, # 3.1E-8 mmol/m3 -> mol/m3 | Half coefficient of CO2 for methane production from H2 ? 
            'm_dH2CH4ProdQ10': 3.5,

            'm_dH2AceProdQ10': 3.5,
            'm_dKCH4ProdAce': 5 * 1e-3, # 5 4–700 mmol/m3 -> mol/m3
            'm_dKCH4ProdO2': 10 * 1e-3 * 1e-3, # 0.01 0.0002–0.040 mmol/m3 -> mol/m3
            'm_dCH4ProdQ10': 4,

            'm_drCH4Prod': 0.5, # 0.5 molCH4(mol acetate) 1 -> mol | Rate of CH4 production
            # Note: the Michaelis term is currently commented out in
            # _calculate_methane_oxidation, because enabling it also requires the
            # O2 term, which zeros wetland oxidation.  The literature value would
            # be 5e-3 mol/m³ ('m_dKCH4OxidCH4': 5 * 1e-3) when the term is used.
            'm_dKCH4OxidCH4': 5 * 1e-8, # mol/m³ | Half-saturation coefficient of CH4 oxidation (Michaelis term disabled)
            
            'm_dKAOMCH4OxidCH4': 2 * 1e-3, # mmol/m3 (0.0025 mmol/L), Half-saturation coefficient of CH4
            'm_dKCH4OxidO2': 10 * 1e-3, # (0.5 mmol/L = 0.5 mol/m3) Half-saturation coefficient of CH4 oxidation for O2 concentration
            'm_dCH4OxidQ10': 2.0,

            'm_dAOMCH4OxidQ10': 2.0,
            'm_drAer': 2,
            'm_dKAerO2': 10,
            'm_dAerDecomQ10': 2.0,
            'm_drCH4Oxid': 2,
            'm_dKe': 0.03,
            'm_dCH4min': 0.5 * 1e-3, # mmol/m3 -> mol/m3 | Minimum concentration of CH4  

            'm_dAirCH4': 0.0893,
            'm_dAirH2': 0.0257,
            'm_dAirO2': 9.375e3,
            'm_dAirCO2': 16.295,

            'AOM': 0.0,
            'H2maxCH4': 1000,
            'AcemaxCH4': 2000,
        }

        # ── Mode selection (config vs hardcoded) ─────────────────────────────
        use_config = False
        try:
            params_obj = self.cfg.parameters
            if hasattr(params_obj, 'get'):
                use_config = bool(params_obj.get('use_config_microbe_params', False))
            else:
                use_config = bool(getattr(params_obj, 'use_config_microbe_params', False))
        except Exception:
            use_config = False

        # ── Load the config-driven dict if requested ────────────────────────
        config_params = {}
        if use_config:
            try:
                params_obj = self.cfg.parameters
                mp = None
                if hasattr(params_obj, 'get'):
                    mp = params_obj.get('microbe_par_dict', None)
                if mp is None:
                    mp = getattr(params_obj, 'microbe_par_dict', None)
                if mp is None:
                    config_params = {}
                elif isinstance(mp, dict):
                    config_params = dict(mp)
                elif hasattr(mp, 'items'):
                    config_params = dict(mp.items())
                elif hasattr(mp, '__dict__'):
                    config_params = {k: v for k, v in vars(mp).items() if not k.startswith('_')}
            except Exception:
                config_params = {}

        # ── Resolve final params: config overrides hardcoded entry-by-entry ──
        if use_config and config_params:
            microbial_params = dict(hardcoded_params)
            divergences = []
            for k, v_cfg in config_params.items():
                if k in hardcoded_params:
                    v_hard = hardcoded_params[k]
                    if v_cfg != v_hard:
                        try:
                            ratio = float(v_hard) / float(v_cfg) if float(v_cfg) != 0 else float('inf')
                        except Exception:
                            ratio = float('nan')
                        divergences.append((k, v_hard, v_cfg, ratio))
                microbial_params[k] = v_cfg
            print(f"[CLMMicrobe] microbial-parameter scheme = CONFIG ({len(config_params)} entries from cfg.parameters.microbe_par_dict)")
            if divergences:
                print(f"[CLMMicrobe] {len(divergences)} entries differ between hardcoded and config:")
                for k, vh, vc, r in divergences[:10]:
                    print(f"    {k:<28} hardcoded={vh!s:<14}  config={vc!s:<14}  hardcoded/config={r:.4g}x")
                if len(divergences) > 10:
                    print(f"    ... and {len(divergences)-10} more")
        else:
            if use_config:
                print(f"[CLMMicrobe] microbial-parameter scheme = HARDCODED (use_config_microbe_params=True but cfg.parameters.microbe_par_dict missing/empty; falling back)")
            else:
                print(f"[CLMMicrobe] microbial-parameter scheme = HARDCODED (47 production values)")
            microbial_params = hardcoded_params

        for key, value in microbial_params.items():
            setattr(self, key, value)

    def _init_soil_decomp_parameters(self):
        """Initialize soil parameters in a structured way"""
        self.soil_params = self.constants['CLM_base_decomposition_rate']

    def _init_arrays(self):
        """Initialize all arrays to None"""
        arrays = [
            'pH', 'hr', 'cwdc', 'cwdn', 'lit1c', 'lit1n', 'lit2c', 'lit2n',
            'lit3c', 'lit3n', 'som1c', 'som1n', 'som2c', 'som2n', 'som3c', 'som3n',
            'som4c', 'som4n', 'conc_ch4', 'conc_o2', 'conc_co2', 'conc_h2',
            'm_dAC', 'm_dAce', 'm_dOrgAcid', 'm_dCH4', 'm_dO2', 'm_dCO2', 'm_dH2',
            'm_dAceMethanogens', 'm_dH2Methanogens', 'm_dMethanotrophs', 'm_dAOMMethanotrophs',
            'soilpH', 'hco3', 'ch4_prod', 'co2_prod', 'h2_prod', 'ch4_oxid', 'o2_cons', 'net_ch4_flux', 'net_co2_flux'
        ]
        
        for array_name in arrays:
            setattr(self, array_name, None)

    def initialize(self):
        """One-time initialization"""
        if self.initialized:
            return self   
        self._get_user_inputs()
        self._allocate_arrays_ts() 
        self._read_CLM_initial_conditions()
        self._read_input_data()  # One-time I/O
        return self

    def run(self, scenario='control', params=None, reset=False):
        """Main simulation loop"""
        self.scenario = scenario # store scenario for output naming 
        
        try:
            #### Step 1: Get user inputs and initialize model ####
            self.initialize()

            ## Step 2: Update parameters if provided #### 
            if params is not None:
                for param_name, param_value in params.items():
                    if hasattr(self, param_name):
                        setattr(self, param_name, param_value)
                    else:
                        print(f"Warning: Parameter {param_name} not found in model")

            #### Step 3: Run the main simulation loop ####
            self._calculate_atmospheric_concentrations()
            self._set_initial_microbial_conditions()
            self._calculate_proper_diffusion_coefficients()
            self.vwc_scaler = self._calculate_water_scalar_vwc()

            #### Step 4: Run simulation all-timesteps #### 
            results_df = self._run_simulation()
            
            return results_df  

        except Exception as e:
            print(f"Error running simulation: {e}")
            raise
    
    def _get_user_inputs(self):
        """Get all user inputs"""
        
        self.nr = self.inputs.shape[0] # Using input dictionary for number of records

        self.output_path = self.cfg.paths['env']['RESULTS_DIR']  # Using config file for output path 

    def _allocate_arrays_ts(self):
        """Allocate memory for arrays"""
        if self.nr <= 0:
            raise ValueError("Number of simulation steps must be positive")
            
        size = self.nr
        
        # Driving forces
        self.forc_t = np.zeros(size, dtype=R8)
        self.forc_pbot = np.zeros(size, dtype=R8)
        self.t_soisno = np.zeros(size, dtype=R8)
        self.h2osoi_vol = np.zeros(size, dtype=R8)
        
        # State variables
        self.pH = np.zeros(size, dtype=R8)
        self.hr = np.zeros(size, dtype=R8)

        # per-pool HR contributions
        self.hr_litr1 = np.zeros(size, dtype=R8)
        self.hr_litr2 = np.zeros(size, dtype=R8)
        self.hr_litr3 = np.zeros(size, dtype=R8)
        self.hr_soil1 = np.zeros(size, dtype=R8)
        self.hr_soil2 = np.zeros(size, dtype=R8)
        self.hr_soil3 = np.zeros(size, dtype=R8)
        self.hr_soil4 = np.zeros(size, dtype=R8)
        
        # Concentrations
        self.conc_ch4 = np.zeros(size, dtype=R8)
        self.conc_o2 = np.zeros(size, dtype=R8)
        self.conc_co2 = np.zeros(size, dtype=R8)
        self.conc_h2 = np.zeros(size, dtype=R8)
        
        # Soil pools
        self.cwdc = np.zeros(size, dtype=R8)
        self.cwdn = np.zeros(size, dtype=R8)
        self.lit1c = np.zeros(size, dtype=R8)
        self.lit1n = np.zeros(size, dtype=R8)
        self.lit2c = np.zeros(size, dtype=R8)
        self.lit2n = np.zeros(size, dtype=R8)
        self.lit3c = np.zeros(size, dtype=R8)
        self.lit3n = np.zeros(size, dtype=R8)
        self.som1c = np.zeros(size, dtype=R8)
        self.som1n = np.zeros(size, dtype=R8)
        self.som2c = np.zeros(size, dtype=R8)
        self.som2n = np.zeros(size, dtype=R8)
        self.som3c = np.zeros(size, dtype=R8)
        self.som3n = np.zeros(size, dtype=R8)
        self.som4c = np.zeros(size, dtype=R8)
        self.som4n = np.zeros(size, dtype=R8)
        
        # Microbial parameters
        self.m_dAC = np.zeros(size, dtype=R8)
        self.m_dAce = np.zeros(size, dtype=R8)
        self.m_dOrgAcid = np.zeros(size, dtype=R8)
        self.m_dCH4 = np.zeros(size, dtype=R8)
        self.m_dO2 = np.zeros(size, dtype=R8)
        self.m_dCO2 = np.zeros(size, dtype=R8)
        self.m_dH2 = np.zeros(size, dtype=R8)
        self.m_dAceMethanogens = np.zeros(size, dtype=R8)
        self.m_dH2Methanogens = np.zeros(size, dtype=R8)
        self.m_dMethanotrophs = np.zeros(size, dtype=R8)
        self.m_dAOMMethanotrophs = np.zeros(size, dtype=R8)
        self.ch4_prod = np.zeros(size, dtype=R8)
        self.co2_prod = np.zeros(size, dtype=R8)
        self.h2_prod = np.zeros(size, dtype=R8)
        self.ch4_oxid = np.zeros(size, dtype=R8)
        self.net_ch4_flux = np.zeros(size, dtype=R8)
        self.net_co2_flux = np.zeros(size, dtype=R8)
        self.o2_cons = np.zeros(size, dtype=R8)
        self.soilpH = np.zeros(size, dtype=R8)
        self.hco3 = np.zeros(size, dtype=R8)
        self.ace_cons = np.zeros(size, dtype=R8)
        self.h2ch4_prod = np.zeros(size, dtype=R8)
        self.acco2_prod = np.zeros(size, dtype=R8)
        self.h2ace_prod = np.zeros(size, dtype=R8)
        self.h2co2_cons = np.zeros(size, dtype=R8) 

        self.sm_scaler = np.zeros(size, dtype=R8) 
        self.redox_status = list(range(size))

    def _read_CLM_initial_conditions(self):
        """Read initial conditions from file"""
        # rather than using initial file, we will extract values from input dictionary 
        # unit in kg C/m2
        pools_data = self.constants['CLM_pool_data']
        # multiply by 1000 to convert from kg C/m2 to g C/m2
        pools_data = [[pools_data[i][0] * 1000.0, pools_data[i][1] * 1000.0] for i in range(len(pools_data))]

        # update pools by a scaler based on soil sample mass
        pools_data = [[x[0] * self.soilsample_scalar, x[1] * self.soilsample_scalar] for x in pools_data]

        # Per-drainage pool-size scaler for the HR/CO2 magnitude shortfall: the
        # placeholder LIT1-3/SOM1 pools are smaller than is physically realistic.
        _drainage_pool_scalers = {
            "upland":       10.0,
            "transitional": 10.0,
            "transition":   10.0,
            "wetland":      40.0,
            "unknown":       1.0,  # fall back to original placeholder
        }
        _scaler = _drainage_pool_scalers.get(str(self.site_drainage).strip().lower(), 1.0)
        # Don't scale CWD (index 0) — it's already 65000 gC/m², physically reasonable
        pools_data = [
            entry if i == 0 else [entry[0] * _scaler, entry[1] * _scaler]
            for i, entry in enumerate(pools_data)
        ]
        self.cwdc[0], self.cwdn[0] = pools_data[0]  # orinigal in 65.0 kg C/m², 6.5 kg N/m² for CWD carbon stock
        self.lit1c[0], self.lit1n[0] = pools_data[1] # ~1 g C/m², ~0.07 g N/m² for litter 1 carbon stock
        self.lit2c[0], self.lit2n[0] = pools_data[2]
        self.lit3c[0], self.lit3n[0] = pools_data[3]
        self.som1c[0], self.som1n[0] = pools_data[4]
        self.som2c[0], self.som2n[0] = pools_data[5] 
        self.som3c[0], self.som3n[0] = pools_data[6]
        self.som4c[0], self.som4n[0] = pools_data[7]

        microbial_data = self.constants['CLM_microbial_initial']
        self.initialac, self.initialace, self.initialorgacid, self.initialacemeth, self.initialh2meth, self.initialmethanotrophs = microbial_data
        
        self.initialac = self.initialac          # mol/m3
        self.initialace = self.initialace        # mol/m3
        self.initialorgacid = self.initialorgacid  # mol/m3
        self.initialacemeth = self.initialacemeth
        self.initialh2meth = self.initialh2meth
        self.initialmethanotrophs = self.initialmethanotrophs * 4e5

        # Scale initial methanogen and methanotroph biomass by drainage type.
        # Scalers derived from Hinsby 2023 sequencing data (class-level medians).
        _d = str(self.site_drainage).strip().lower()
        _methanogen_scalers = {
            "wetland":      1.000,
            "transitional": 0.343,
            "transition":   0.343,
            "upland":       0.0023,
            "unknown":      1.000,
        }
        _methanotroph_scalers = {
            "wetland":      1.000,
            "transitional": 0.560,
            "transition":   0.560,
            "upland":       0.335,
            "unknown":      1.000,
        }
        _meth_s = _methanogen_scalers.get(_d, 1.0)
        _mt_s   = _methanotroph_scalers.get(_d, 1.0)

        self.initialacemeth       *= _meth_s
        self.initialh2meth        *= _meth_s
        self.initialmethanotrophs *= _mt_s
        
        self.pch4, self.pco2, self.po2, self.ph2 = self.constants['CLM_gas_initial']['values'] 
        # ensure DataFrame
        if isinstance(self.for_calibration, pd.Series):
            self.for_calibration = self.for_calibration.to_frame()
        self.originalsoilph = self.for_calibration['pH'].iloc[0] if 'pH' in self.for_calibration.columns else 4.15
        
        self.hrtoac = 0.1 # ratio of carbon converted to acetate during heterotrophic respiration  

    def _read_input_data(self):
        """Read forcing columns (AT, pressure, ST, SM, pH) from the input dict."""
        ######## read data from input dictionary ######## 
        self.forc_t = self.inputs['AT'] +  273.15  # atmospheric temperature (C to Kelvin)
        if 'pressure' not in self.inputs: # if pressure is not provided, use standard atmospheric pressure
            self.forc_pbot = np.full(self.nr, 101325.0)  # atmospheric pressure (Pa) 
        else:
            self.forc_pbot = self.inputs['pressure']  # atmospheric pressure (Pa)
        self.t_soisno = self.inputs['ST'] + 273.15  # soil temperature (C to Kelvin)
        self.h2osoi_vol = self.inputs['SM'] #  # volumetric soil water (0<=h2osoi_vol<=watsat) [m3/m3]  (nlevgrnd)
        if 'pH' in self.inputs.columns:
            self.soilpH = self.inputs['pH']
        elif 'pH' in self.for_calibration.columns:
            pH_mean = self.for_calibration['pH'].mean()
            self.soilpH = np.where(self.soilpH == 0, pH_mean, self.soilpH)
        else:
            self.soilpH = np.full(self.nr, self.originalsoilph) # use original soil pH if not provided in input data


    def calculate_henry_constant(self, soil_temp_series, gas_type='O2'):
        """Calculate Henry's constant for various gases (handles pandas Series)."""
        # Solubilities of Gases in Liquids: typical Bunsen coefficients and enthalpy of dissolution (J/mol)
        # Perry and Chilton ( 1973) @ 25 degree C 
        parameters = {
            'O2': {'H_ref': 1.3e-3, 'dH': 1700}, # 0.0013 mol O2 /(L atm), dH * (1/T - 1/T_ref) must be unitless
            'H2': {'H_ref': 7.8e-4, 'dH': 500},
            'N2O': {'H_ref': 2.5e-2, 'dH': 2600},
            'CO2': {'H_ref': 3.5e-2, 'dH': 2400},
            'CH4': {'H_ref': 1.4e-3, 'dH': 1700},
        }
        
        params = parameters.get(gas_type, parameters['O2'])  # default to O2
        H = params['H_ref'] * np.exp(params['dH'] * (1/(273.15 + soil_temp_series) - 1/298.15)) # unitless (mL gas / mL liquid)

        return H

    def _calculate_proper_diffusion_coefficients(self):
        """Calculate physically correct diffusion coefficients using Millington-Quirk"""
        
        # Reference diffusion coefficients in free air at 20°C (m²/s)
        self.DO2_air_ref = 2.0e-5    # Oxygen
        self.DCO2_air_ref = 1.6e-5   # Carbon dioxide  
        self.DCH4_air_ref = 2.1e-5   # Methane
        self.DH2_air_ref = 7.0e-5    # Hydrogen
        
        # Temperature correction
        T_current = self.inputs['ST'] + 273.15  # Kelvin
        T_ref = 293.15  # 20°C reference
        temp_factor = (T_current / T_ref)**1.75
        
        # Temperature-corrected free-air diffusion
        self.DO2_air = self.DO2_air_ref * temp_factor
        self.DCO2_air = self.DCO2_air_ref * temp_factor
        self.DCH4_air = self.DCH4_air_ref * temp_factor  
        self.DH2_air = self.DH2_air_ref * temp_factor
        
        # Millington-Quirk (1961) soil gas diffusion reduction factor
        # D_eff/D_0 = AFP^(4/3) / Φ^2  (original MQ 1961, exponent 1.333)
        air_filled_porosity = (self.maxi_porosity * self.porosity_scaler) - self.inputs['SM']
        air_filled_porosity = np.where(air_filled_porosity < 0.001, 0.001, air_filled_porosity)

        self.D_P = (air_filled_porosity**1.333) / ((self.maxi_porosity * self.porosity_scaler)**2)

        # Effective soil diffusion coefficients (m²/s)
        self.DO2_soil = self.DO2_air * self.D_P
        self.DCO2_soil = self.DCO2_air * self.D_P
        self.DCH4_soil = self.DCH4_air * self.D_P
        self.DH2_soil = self.DH2_air * self.D_P

    def _calculate_atmospheric_concentrations(self):
        """Convert atmospheric concentrations in bottle volume"""
        # Henry's Law constants (mol/L/atm) - adjust for your temperature

        # Usage in your class
        self.Henry_constant_O2 = self.calculate_henry_constant(self.inputs['ST'], 'O2') # unit: mol/L
        self.Henry_constant_H2 = self.calculate_henry_constant(self.inputs['ST'], 'H2')
        self.Henry_constant_CO2 = self.calculate_henry_constant(self.inputs['ST'], 'CO2')
        self.Henry_constant_CH4 = self.calculate_henry_constant(self.inputs['ST'], 'CH4')
        
        # Calculate proper diffusion coefficients
        self._calculate_proper_diffusion_coefficients()
        self.total_atmospheric_pressure = 101.325 # kPa
        self.total_atmospheric_pressure_atm = 1.0 # atm 

        # Calculate surface concentrations using Henry's Law (mol/m³)
        C_O2_surface = (self.O2airfrac[0] * self.total_atmospheric_pressure_atm * 
                    self.Henry_constant_O2 * 1000)  # mol/m³
        C_CO2_surface = (self.CO2airfrac[0] * self.total_atmospheric_pressure_atm * 
                        self.Henry_constant_CO2 * 1000)  # mol/m³
        C_CH4_surface = (self.CH4airfrac[0] * self.total_atmospheric_pressure_atm * 
                        self.Henry_constant_CH4 * 1000)  # mol/m³
        C_H2_surface = (self.H2airfrac[0] * self.total_atmospheric_pressure_atm * 
                    self.Henry_constant_H2 * 1000)  # mol/m³

        # Assume concentrations at depth are initially zero (maximum gradient)
        # In reality, these would be updated during the simulation
        C_O2_depth = 0.0  # mol/m³
        C_CO2_depth = 0.0  # mol/m³  
        C_CH4_depth = 0.0  # mol/m³
        C_H2_depth = 0.0  # mol/m³
        
        # Calculate concentration gradients (over soil depth) (mol/m4)
        dC_O2_dx = (C_O2_surface - C_O2_depth)/ self.soil_depth  # assuming soil_depth in meters
        dC_CO2_dx = (C_CO2_surface - C_CO2_depth)/ self.soil_depth
        dC_CH4_dx = (C_CH4_surface - C_CH4_depth)/ self.soil_depth
        dC_H2_dx = (C_H2_surface - C_H2_depth)/ self.soil_depth

        # Calculate diffusion fluxes using Fick's First Law: J = -D * dC/dx (mol/m²/s)
        # DO2_soil, DCO2_soil, DCH4_soil, DH2_soil are in m²/s 
        self.J_O2 = -self.DO2_soil * dC_O2_dx  # Negative sign for direction
        self.J_CO2 = -self.DCO2_soil * dC_CO2_dx
        self.J_CH4 = -self.DCH4_soil * dC_CH4_dx  
        self.J_H2 = -self.DH2_soil * dC_H2_dx

        # Convert fluxes to concentrations in soil volume over the timestep (mol/m³)
        self.conc_o2 = self.J_O2 * self.timestep / self.soil_depth 
        self.conc_co2 = self.J_CO2 * self.timestep / self.soil_depth
        self.conc_ch4 = self.J_CH4 * self.timestep / self.soil_depth
        self.conc_h2 = self.J_H2 * self.timestep / self.soil_depth
        
        # Ensure non-negative concentrations (mol/m³)
        self.conc_o2 = np.where(self.conc_o2 < 0, 0, self.conc_o2)
        self.conc_co2 = np.where(self.conc_co2 < 0, 0, self.conc_co2)
        self.conc_ch4 = np.where(self.conc_ch4 < 0, 0, self.conc_ch4)
        self.conc_h2 = np.where(self.conc_h2 < 0, 0, self.conc_h2)

    def _set_initial_microbial_conditions(self):
        """Set initial microbial biomass, converting g C in soil volume to g C/m3."""
        self.biomass_units = 'gC/m3'
        # initial substrate pools in g C/m3 
        self.m_dAC[0] = self.initialac /self.soilvolume # (0.0063 g C in soil volume) -> 56 g C/m3 (in deompotion model, unit is gC/m2/s)
        self.m_dAce[0] = self.initialace/self.soilvolume # 1.0e-06 g C in soil in soil volume -> g C/m3 
        self.m_dOrgAcid[0] = self.initialorgacid/self.soilvolume # 1e-6 g C i in soil volume -> g C/m3
        # initial microbial biomass in g C/m3 (typical)
        self.m_dAceMethanogens[0] = self.initialacemeth # 0.00376,  1.0e-06 g C in soil volume -> g C/m3
        self.m_dH2Methanogens[0] = self.initialh2meth # 0.00752 g C in soil volume -> g C/m3
        self.m_dMethanotrophs[0] = self.initialmethanotrophs # g C in soil volume -> mol C/m3
            
        self.m_dAOMMethanotrophs[0] = self.m_dMethanotrophs[0].copy()  # initial AOM methanotrophs equal to total methanotrophs
        
        self.net_ch4_flux[0] = 0
        self.net_co2_flux[0] = 0
        self.hco3[0] = 0.0
        self.microberecov = 1.0

        self.m_dMaxMethanotrophs = 1.5  # Maximum methanotroph population g C/m3
        self.m_dMaxAceMethanogens = 1.5   # Maximum acetoclast population g C/m3
        self.m_dMaxH2Methanogens = 1.5    # Maximum hydrogenotroph population g C/m3

    def _run_simulation(self):
        """Run the main simulation loop"""
        
        for n in range(self.nr):           
            self._simulation_step(n)

        #### Step 4: Post-process and output results #### 
        results_df = self._write_output(self.scenario)
        return results_df 
        
    def _simulation_step(self, n):
        """Perform one simulation step"""
        self.year = self.inputs['year'][n]
        self.gas_units = 'mol/m3'

        # Calculate surface concentrations for current air fractions (mol/m³)
        C_O2_surface = (self.O2airfrac[n] * self.total_atmospheric_pressure_atm * 
                    self.Henry_constant_O2[n] * 1000)
        C_CO2_surface = (self.CO2airfrac[n] * self.total_atmospheric_pressure_atm * 
                        self.Henry_constant_CO2[n] * 1000)
        C_CH4_surface = (self.CH4airfrac[n] * self.total_atmospheric_pressure_atm * 
                        self.Henry_constant_CH4[n] * 1000)
        C_H2_surface = (self.H2airfrac[n] * self.total_atmospheric_pressure_atm * 
                    self.Henry_constant_H2[n] * 1000)
        
        # Use previous concentrations as depth concentrations (updated by microbial processes)
        # At n=0, initialize with surface concentrations (equilibrium)
        if n == 0:
            self.conc_o2[n] = C_O2_surface * 0.005  # start with 0.5% of surface concentration
            self.conc_co2[n] = C_CO2_surface * 0.005
            self.conc_ch4[n] = C_CH4_surface * 0.005
            self.conc_h2[n] = C_H2_surface * 0.005
            
            self.J_CH4 = 0.0
            self.J_CO2 = 0.0
            self.J_O2 = 0.0
            self.J_H2 = 0.0

        else:
            # Use previous concentrations as depth concentrations
            C_O2_depth = self.conc_o2[n-1]
            C_CO2_depth = self.conc_co2[n-1]
            C_CH4_depth = self.conc_ch4[n-1] 
            C_H2_depth = self.conc_h2[n-1]
        
            # Calculate gradients over soil depth (mol/m4)
            dC_O2_dx = (C_O2_surface - C_O2_depth)/ self.soil_depth
            dC_CO2_dx = (C_CO2_surface - C_CO2_depth) / self.soil_depth 
            dC_CH4_dx = (C_CH4_surface - C_CH4_depth) / self.soil_depth
            dC_H2_dx = (C_H2_surface - C_H2_depth) / self.soil_depth
            
            # Calculate fluxes from gas to soil (m2/s * mol/m4 -> mol/m2/s) 
            self.J_O2 = self.DO2_soil[n] * dC_O2_dx
            self.J_CO2 = self.DCO2_soil[n] * dC_CO2_dx
            self.J_CH4 = self.DCH4_soil[n] * dC_CH4_dx
            self.J_H2 = self.DH2_soil[n] * dC_H2_dx
        
            # Convert flux to concentration in soil volume (mol/m2/s -> mol/m3)
            delta_conc_O2 = self.J_O2 * self.timestep / self.soil_depth
            delta_conc_CO2 = self.J_CO2 * self.timestep / self.soil_depth
            delta_conc_CH4 = self.J_CH4 * self.timestep / self.soil_depth
            delta_conc_H2 = self.J_H2 * self.timestep / self.soil_depth

            delta_conc_O2 = max(delta_conc_O2, 0)
            delta_conc_CO2 = max(delta_conc_CO2, 0)
            delta_conc_CH4 = max(delta_conc_CH4, 0)
            delta_conc_H2 = max(delta_conc_H2, 0)

            self.conc_o2[n] = self.conc_o2[n-1] + delta_conc_O2
            self.conc_co2[n] = self.conc_co2[n-1] + delta_conc_CO2
            self.conc_ch4[n] = self.conc_ch4[n-1] + delta_conc_CH4
            self.conc_h2[n] = self.conc_h2[n-1] + delta_conc_H2

        # # Ensure non-negative concentrations (mol/m3)
        self.conc_o2[n] = max(self.conc_o2[n], 0)
        self.conc_co2[n] = max(self.conc_co2[n], 0)
        self.conc_ch4[n] = max(self.conc_ch4[n], 0)
        self.conc_h2[n] = max(self.conc_h2[n], 0)

        hr_result = self.decomp(n,
            self.t_soisno[n], self.hr[n],
            self.cwdc[n], self.cwdn[n], self.lit1c[n], self.lit1n[n],
            self.lit2c[n], self.lit2n[n], self.lit3c[n], self.lit3n[n],
            self.som1c[n], self.som1n[n], self.som2c[n], self.som2n[n],
            self.som3c[n], self.som3n[n], self.som4c[n], self.som4n[n],
            self.conc_o2[n], self.m_dAC[n], self.hrtoac
        )

        # (hr in g/Cm3 in open volume, pools in gC/m2 in open space, m_dAC changes at a rate and depends on initial unit)
        self.hr[n], updated_pools, self.m_dAC[n] = hr_result
        if updated_pools:
            self._update_pools_from_dict(n, updated_pools)

        # Calculate O2 consumption from respiration and update concentrations (mol/m3)
        O2_consumed = self._calculate_O2_consumption(n) # aerobic respiration O2 consumption from atmospheric diffusion
        # Update headspace O2 fractions based on gaseous consumption

        # Microbial processes
        if self.biomass_units == 'gC':
            self.m_dAC[n] /= self.soilvolume # gC/m3 
            self.m_dAce[n] /= self.soilvolume # gC/m3
            self.m_dOrgAcid[n] /= self.soilvolume # gC/m3
        if self.gas_units == 'mol': 
            self.m_dCH4[n] = self.conc_ch4[n] / self.bottlevolume # mol/m3
            self.m_dO2[n] = self.conc_o2[n] / self.bottlevolume # mol/m3
            self.m_dCO2[n] = self.conc_co2[n] / self.bottlevolume # mol/m3
            self.m_dH2[n] = self.conc_h2[n] / self.bottlevolume # mol/m3 
        else: 
            self.m_dCH4[n] = self.conc_ch4[n] # mol/m3 in dissolved soil water 
            self.m_dO2[n] = self.conc_o2[n] # mol/m3
            self.m_dCO2[n] = self.conc_co2[n] # mol/m3 
            self.m_dH2[n] = self.conc_h2[n] # mol/m3 in dissolved soil water

        # Microbial recovery (default to 1.0)
        self.microberecov = self._calculate_microbial_recovery(n)
        microbe_results = self.microbe(
                n,
                self.m_dAC[n], self.m_dAce[n], self.m_dOrgAcid[n],
                self.m_dCH4[n], self.m_dO2[n], self.m_dCO2[n], self.m_dH2[n],
                self.m_dAceMethanogens[n], self.m_dH2Methanogens[n],
                self.m_dMethanotrophs[n], self.m_dAOMMethanotrophs[n],
                self.t_soisno[n] - 273.15, self.soilpH[n], self.hco3[n],
                self.microbeparfilename, self.microberecov
            )
        # Update microbial results
        self._update_from_microbe_results(n, microbe_results)
       
        # Convert back from concentration units to gC in soil volume and mol in bottle volume
        if self.biomass_units == 'gC':
            self.m_dAC[n] *= self.soilvolume # gC
            self.m_dAce[n] *= self.soilvolume # gC
            self.m_dOrgAcid[n] *= self.soilvolume # gC
        
        if self.gas_units == 'mol': 
            self.conc_ch4[n] = self.m_dCH4[n] * self.bottlevolume # mol
            self.conc_o2[n] = self.m_dO2[n] * self.bottlevolume # mol
            self.conc_co2[n] = self.m_dCO2[n] * self.bottlevolume # mol
            self.conc_h2[n] = self.m_dH2[n] * self.bottlevolume # mol
        else:
            self.conc_ch4[n] = self.m_dCH4[n] # mol in bottle volume
            self.conc_o2[n] = self.m_dO2[n] # mol
            self.conc_co2[n] = self.m_dCO2[n] # mol
            self.conc_h2[n] = self.m_dH2[n] # mol

        self.o2_cons[n] += self.hr[n] / 12.0

        if n < self.nr - 1:
            self._update_next_timestep(n)

    def _calculate_O2_consumption(self, n):
        """Calculate O2 consumption from heterotrophic respiration"""
        # Convert hr (gC/m²/s) to O2 consumption (mol/m³)
        # Assuming respiratory quotient ~1: 1 mol O2 per mol CO2 produced
        O2_consumption_flux = self.hr[n]/ 12.0  # gc/m3 to mol/m3
        O2_consumption_conc = (O2_consumption_flux * self.timestep)
        
        return O2_consumption_conc

    def _calculate_microbial_recovery(self, n):
        """Calculate microbial recovery factor"""
        if n < 0:  # This condition seems odd in the original code
            return max(0.0, (n / 480) ** 0.5)
        else:
            return 1.0

    def _update_from_microbe_results(self, n, microbe_results):

        # in gC in soil volume 
        self.m_dAC[n] = microbe_results['m_dAC']
        self.m_dAce[n] = microbe_results['m_dAce']
        self.m_dOrgAcid[n] = microbe_results['m_dOrgAcid']

        # in mol/m3 
        self.m_dCH4[n] = microbe_results['m_dCH4']
        self.m_dO2[n] = microbe_results['m_dO2']
        self.m_dCO2[n] = microbe_results['m_dCO2']
        self.m_dH2[n] = microbe_results['m_dH2']

        self.m_dAceMethanogens[n] = microbe_results['m_dAceMethanogens']
        self.m_dH2Methanogens[n] = microbe_results['m_dH2Methanogens']
        self.m_dMethanotrophs[n] = microbe_results['m_dMethanotrophs']
        self.m_dAOMMethanotrophs[n] = microbe_results['m_dAOMMethanotrophs']

        self.ch4_prod[n] = microbe_results['ch4_prod']
        self.co2_prod[n] = microbe_results['co2_prod']
        self.h2_prod[n] = microbe_results['h2_prod']
        self.ch4_oxid[n] = microbe_results['ch4_oxid']
        self.o2_cons[n] = microbe_results['o2_cons']
        # add more in mol/m3
        self.ace_cons[n] = microbe_results['ace_cons']
        self.h2ch4_prod[n] = microbe_results['h2ch4_prod']
        self.h2ace_prod[n] = microbe_results['h2ace_prod']   
        self.acco2_prod[n] = microbe_results['acco2_prod']
        self.h2co2_cons[n] = microbe_results['h2co2_cons']  

        # fluxes in mmol/m3 
        # hr: gC/s in bottle volume 
        self.net_ch4_flux[n] = microbe_results['ch4_flux'] # fluxes returned in mmol/m3
        self.net_co2_flux[n] = microbe_results['co2_flux']  # +  1e3 * self.hr[n] / 12 # fluxes returned in mmol/m3
        
        self.soilpH[n] = microbe_results['soilpH']
        
    def _update_pools_from_dict(self, n, pools_dict):
        """Update pools from dictionary"""
        for key, value in pools_dict.items():
            if hasattr(self, key):
                array = getattr(self, key)
                array[n] = value

    def _update_next_timestep(self, n):
        """Update state variables for next time step"""
        # Gas concentrations
        self.conc_o2[n+1] = self.conc_o2[n]
        self.conc_co2[n+1] = self.conc_co2[n]
        self.conc_ch4[n+1] = self.conc_ch4[n]
        self.conc_h2[n+1] = self.conc_h2[n]

        self.O2airfrac[n+1] = self.O2airfrac[n]
        self.CO2airfrac[n+1] = self.CO2airfrac[n]
        self.CH4airfrac[n+1] = self.CH4airfrac[n]
        self.H2airfrac[n+1] = self.H2airfrac[n]
        
        # Soil pools (gC/m2)
        pools = ['cwdc', 'cwdn', 'lit1c', 'lit1n', 'lit2c', 'lit2n',
                'lit3c', 'lit3n', 'som1c', 'som1n', 'som2c', 'som2n',
                'som3c', 'som3n', 'som4c', 'som4n']
        
        for pool in pools:
            array = getattr(self, pool)
            array[n+1] = array[n]
        
        # Microbial parameters
        microbial_arrays = ['m_dAC', 'm_dAce', 'm_dOrgAcid', 
                            'm_dAceMethanogens', 'm_dH2Methanogens', 'm_dMethanotrophs', 'm_dAOMMethanotrophs',
                          ]
        
        for array_name in microbial_arrays:
            array = getattr(self, array_name)
            array[n+1] = array[n]

    ######### Decomposition Module ######### 
    def decomp(self, n, t_soisno, hrsum, cwdc, cwdn,
               litr1c, litr1n, litr2c, litr2n, litr3c, litr3n,
               soil1c, soil1n, soil2c, soil2n, soil3c, soil3n, soil4c, soil4n,
               conc_o2, m_dAC, hrtoac ):
        """Full decomposition logic. Returns hr (decomposition rate, gC/m2/s)."""
        # Read soil parameters
        dr1, dr2, dr3, dr4, dr5, dr6, dr7, dr8 = self.soil_params

        dt = self.timestep # seconds
        dtd = dt
        k_l1 = -np.log(1-dr1) # decomposition rate constant litter 1
        k_l2 = -np.log(1-dr2) # decomposition rate constant litter 2
        k_l3 = -np.log(1-dr3) # decomposition rate constant litter 3
        k_s1 = -np.log(1-dr4) # decomposition rate constant SOM 1
        k_s2 = -np.log(1-dr5) # decomposition rate constant SOM 2
        k_s3 = -np.log(1-dr6) # decomposition rate constant SOM 3
        k_s4 = -np.log(1-dr7) # decomposition rate constant SOM 4
        k_frag = -np.log(1-dr8) # corrected fragmentation rate constant CWD
        
        k_l1 = 1-np.exp(-k_l1*dtd)
        k_l2 = 1-np.exp(-k_l2*dtd)
        k_l3 = 1-np.exp(-k_l3*dtd)
        k_s1 = 1-np.exp(-k_s1*dtd)
        k_s2 = 1-np.exp(-k_s2*dtd)
        k_s3 = 1-np.exp(-k_s3*dtd)
        k_s4 = 1-np.exp(-k_s4*dtd)
        k_frag = 1-np.exp(-k_frag*dtd)

        # factors affecting decomposition rates(temperature, soil moisture, oxygen availability, pH)

        # HR Q10 = 3.0 for boreal organic soils (range 1.5-3.0 supported in
        # literature; the high end suits SOM-rich peaty soils).  Decomposition
        # uses a reference temperature of 10°C; CH4 pathways use self.T_ref_ch4
        # (13.5°C) separately.
        t_scalar = 3.0 ** ((t_soisno - (273.15 + 10.0)) / 10.0)
        w_scalar =  self.vwc_scaler[n] # vwc_scaler from _calculate_water_scalar_vwc

        # Troeh (1982) gas diffusivity scalar.
        # AFP = θ_max × S_θ − θ_t  (θ_max = 1.0, S_θ = porosity_scaler, θ_t = VWC at step n)
        # f_D = ((AFP − 0.1) / (1 − 0.1))²  — no zero-clamp: at AFP≈0, gives ≈0.012, not 0.
        afp_n = max(self.maxi_porosity * self.porosity_scaler - self.h2osoi_vol[n], 1e-6)
        troeh_fd = max(((afp_n - 0.1) / 0.9) ** 2, self.troeh_floor_fd)
        rate_scalar = t_scalar * w_scalar * troeh_fd
        self.sm_scaler[n] = w_scalar  # store water scalar, 1 for optimal moisture, <1 for dry or saturated conditions 

        # Correct decomposition rates
        ck_l1 = k_l1 * rate_scalar
        ck_l2 = k_l2 * rate_scalar
        ck_l3 = k_l3 * rate_scalar
        ck_s1 = k_s1 * rate_scalar
        ck_s2 = k_s2 * rate_scalar
        ck_s3 = k_s3 * rate_scalar
        ck_s4 = k_s4 * rate_scalar
        ck_frag = k_frag * rate_scalar

        # Calculate C:N ratios for input value (single record)
        cn_l1, cn_l2, cn_l3 = self._calculate_cn_ratios(litr1c, litr1n, litr2c, litr2n, litr3c, litr3n)

        # Initialize potential fluxes state variables
        fluxes = self._initialize_fluxes()
        
        # Carbon loss: CWD fragmentation -> litter pools (gC/m2/s, open space) 
        if cwdc > 0:
            cwdc_loss = cwdc * ck_frag / dt # fragmentation rate for CWD carbon (gC/m2/s) 
            fluxes['cwdc_to_litr2c'] = cwdc_loss * self.cwd_fcel # cellulose fraction of coarse woody debris
            fluxes['cwdc_to_litr3c'] = cwdc_loss * self.cwd_flig # lignin fraction of coarse woody debris
            
        if cwdn > 0:
            cwdn_loss = cwdn * ck_frag / dt
            fluxes['cwdn_to_litr2n'] = cwdn_loss * self.cwd_fcel
            fluxes['cwdn_to_litr3n'] = cwdn_loss * self.cwd_flig

        # Litter and SOM decomposition, rate of carbon pools (gC/m2/s)
        fluxes = self._calculate_potential_fluxes_loss(fluxes, 
            litr1c, litr1n, litr2c, litr2n, litr3c, litr3n,
            soil1c, soil1n, soil2c, soil2n, soil3c, soil3n, soil4c, soil4n,
            cn_l1, cn_l2, cn_l3, ck_l1, ck_l2, ck_l3, ck_s1, ck_s2, ck_s3, ck_s4, dt
        )
        
        # Calculate immobilization and mineralization
        immob, gross_nmin = self._calculate_nitrogen_fluxes(fluxes)
        
        # Fraction of potential immobilization (simplified)
        fpi = 1.0  # This would normally come from CNAllocation

        # Calculate actual fluxes with nitrogen limitation (gC in soil volume)
        actual_fluxes = self._calculate_potential_hr(fluxes, fpi)

        # Calculate heterotrophic respiration (gC in open space)
        hr_total = self._calculate_heterotrophic_respiration(actual_fluxes) # o.045 gC/m2/s in soil

        # Update acetate pool with the HR-to-acetate ratio (hrtoac); acetate
        # production does not depend on O2, so use total HR before O2 limitation.
        if self.biomass_units == 'gC/m3':
            m_dAC += hr_total * hrtoac * self.timestep/self.soil_depth # gC/m3/s * fraction_to_availC -> gC/m3
        elif self.biomass_units == 'gC':
            m_dAC += hr_total * dt * hrtoac * self.soilvolume  # gC/m3/s * s in soil volume * fraction_to_availC

        # O2 limitation on HR disabled: aerobic/anaerobic partitioning is already
        # handled implicitly via AFP-driven diffusion scalars.
        o2limiting = 1
        hr_total *= o2limiting
        actual_fluxes = self._apply_oxygen_limitation(actual_fluxes, o2limiting)

        # capture per-pool HR contributions
        self.hr_litr1[n] = actual_fluxes.get('litr1_hr', 0.0)
        self.hr_litr2[n] = actual_fluxes.get('litr2_hr', 0.0)
        self.hr_litr3[n] = actual_fluxes.get('litr3_hr', 0.0)
        self.hr_soil1[n] = actual_fluxes.get('soil1_hr', 0.0)
        self.hr_soil2[n] = actual_fluxes.get('soil2_hr', 0.0)
        self.hr_soil3[n] = actual_fluxes.get('soil3_hr', 0.0)
        self.hr_soil4[n] = actual_fluxes.get('soil4_hr', 0.0)

        # Update soil pools (gC in soil volume)
        updated_pools = self._update_pools(
            cn_l1, cn_l2, cn_l3,
            cwdc, cwdn, litr1c, litr1n, litr2c, litr2n, litr3c, litr3n,
            soil1c, soil1n, soil2c, soil2n, soil3c, soil3n, soil4c, soil4n,
            actual_fluxes
        )

        return hr_total, updated_pools, m_dAC
        
    def _calculate_water_scalar_vwc(self, soil_type='medium', plot_component=False):
        """Calculate water scalar based on water-filled pore space()"""
        # WFPS capped at 1.0: VWC cannot physically exceed effective porosity.
        filled_proportion = np.clip(
            self.h2osoi_vol / (self.maxi_porosity * self.porosity_scaler), 0.0, 1.0)

        self.opt_max_vwc = self.opt_max_scale * self.vwc_max  # maximum optimal water content for decomposition activity
        # Guard against inverted bell (can occur at parameter-space boundaries).
        opt_min = self.opt_min_vwc
        opt_max = max(self.opt_max_vwc, opt_min + 0.05)
        if np.isscalar(filled_proportion):
            if filled_proportion < opt_min:
                w_scalar = filled_proportion / opt_min
            elif filled_proportion > opt_max:
                w_scalar = np.exp(-3.0 * (filled_proportion - opt_max) / (1.0 - opt_max))
            else:
                w_scalar = 1.0
        else:
            w_scalar = np.ones_like(filled_proportion)
            dry_mask = filled_proportion < opt_min
            wet_mask = filled_proportion > opt_max
            w_scalar[dry_mask] = filled_proportion[dry_mask] / opt_min
            w_scalar[wet_mask] = np.exp(-3.0 * (filled_proportion[wet_mask] - opt_max) / (1.0 - opt_max))

        if plot_component:
            filled_proportion_plot = np.linspace(0, 1, 100)
            w_scalar_plot = np.ones_like(filled_proportion_plot)
            dry_mask_plot = filled_proportion_plot < opt_min
            wet_mask_plot = filled_proportion_plot > opt_max
            w_scalar_plot[dry_mask_plot] = filled_proportion_plot[dry_mask_plot] / opt_min
            w_scalar_plot[wet_mask_plot] = np.exp(-3.0 * (filled_proportion_plot[wet_mask_plot] - opt_max) / (1.0 - opt_max))
            # Set the global font size
            
            plt.figure(figsize=(8, 6))
            plt.rcParams['font.size'] = 25  # Set to your desired font size
            plt.plot(filled_proportion_plot, w_scalar_plot, label='Water Stress', linewidth=4)
            plt.axvline(opt_min, color='r', linestyle='--', label='Opt Min VWC', linewidth=4)
            plt.axvline(opt_max, color='g', linestyle='--', label='Opt Max VWC', linewidth=4)
            plt.xlabel('Fractional Water-Filled Pore Space (VWC / VWC_sat)', fontsize=15)
            plt.ylabel('Water Stress of Aerobic Respiration', fontsize=15)
            plt.legend(frameon=False)
            plt.show()

        return np.clip(w_scalar, 0.0, 1.0)

    def _calculate_cn_ratios(self, litr1c, litr1n, litr2c, litr2n, litr3c, litr3n):
        """Calculate C:N ratios for litter pools"""
        cn_l1 = litr1c / litr1n if litr1n > 0 else 0.0
        cn_l2 = litr2c / litr2n if litr2n > 0 else 0.0
        cn_l3 = litr3c / litr3n if litr3n > 0 else 0.0
        
        return cn_l1, cn_l2, cn_l3
    
    def _initialize_fluxes(self):
        """Initialize flux dictionary"""
        return {
            'cwdc_to_litr2c': 0.0, 'cwdc_to_litr3c': 0.0,
            'cwdn_to_litr2n': 0.0, 'cwdn_to_litr3n': 0.0,
            'plitr1c_loss': 0.0, 'plitr2c_loss': 0.0, 'plitr3c_loss': 0.0,
            'psoil1c_loss': 0.0, 'psoil2c_loss': 0.0, 'psoil3c_loss': 0.0, 'psoil4c_loss': 0.0,
            'pmnf_l1s1': 0.0, 'pmnf_l2s2': 0.0, 'pmnf_l3s3': 0.0,
            'pmnf_s1s2': 0.0, 'pmnf_s2s3': 0.0, 'pmnf_s3s4': 0.0, 'pmnf_s4': 0.0,
            'litr1_hr': 0.0, 'litr2_hr': 0.0, 'litr3_hr': 0.0,
            'soil1_hr': 0.0, 'soil2_hr': 0.0, 'soil3_hr': 0.0, 'soil4_hr': 0.0
        }
    
    def _calculate_potential_fluxes_loss(self, fluxes, litr1c, litr1n, litr2c, litr2n, litr3c, litr3n,
                                soil1c, soil1n, soil2c, soil2n, soil3c, soil3n, soil4c, soil4n,
                                cn_l1, cn_l2, cn_l3, ck_l1, ck_l2, ck_l3, ck_s1, ck_s2, ck_s3, ck_s4, dt):
        """Calculate potential C losses from litter/SOM and potential Mineral N fluxes"""
        # Litter 1 -> SOM 1
        if litr1c > 0:
            fluxes['plitr1c_loss'] = litr1c * ck_l1 / dt
            ratio = self.cn_s1 / cn_l1 if litr1n > 0 else 0.0
            fluxes['pmnf_l1s1'] = (fluxes['plitr1c_loss'] * (1.0 - self.rf_l1s1 - ratio)) / self.cn_s1
        
        # Litter 2 -> SOM 2
        if litr2c > 0:
            fluxes['plitr2c_loss'] = litr2c * ck_l2 / dt
            ratio = self.cn_s2 / cn_l2 if litr2n > 0 else 0.0
            fluxes['pmnf_l2s2'] = (fluxes['plitr2c_loss'] * (1.0 - self.rf_l2s2 - ratio)) / self.cn_s2
        
        # Litter 3 -> SOM 3
        if litr3c > 0:
            fluxes['plitr3c_loss'] = litr3c * ck_l3 / dt
            ratio = self.cn_s3 / cn_l3 if litr3n > 0 else 0.0
            fluxes['pmnf_l3s3'] = (fluxes['plitr3c_loss'] * (1.0 - self.rf_l3s3 - ratio)) / self.cn_s3
        
        # SOM 1 -> SOM 2
        if soil1c > 0:
            fluxes['psoil1c_loss'] = soil1c * ck_s1 / dt
            fluxes['pmnf_s1s2'] = (fluxes['psoil1c_loss'] * (1.0 - self.rf_s1s2 - (self.cn_s2 / self.cn_s1))) / self.cn_s2
        
        # SOM 2 -> SOM 3
        if soil2c > 0:
            fluxes['psoil2c_loss'] = soil2c * ck_s2 / dt
            fluxes['pmnf_s2s3'] = (fluxes['psoil2c_loss'] * (1.0 - self.rf_s2s3 - (self.cn_s3 / self.cn_s2))) / self.cn_s3
        
        # SOM 3 -> SOM 4
        if soil3c > 0:
            fluxes['psoil3c_loss'] = soil3c * ck_s3 / dt
            fluxes['pmnf_s3s4'] = (fluxes['psoil3c_loss'] * (1.0 - self.rf_s3s4 - (self.cn_s4 / self.cn_s3))) / self.cn_s4
        
        # SOM 4 loss
        if soil4c > 0:
            fluxes['psoil4c_loss'] = soil4c * ck_s4 / dt
            fluxes['pmnf_s4'] = -fluxes['psoil4c_loss'] / self.cn_s4
        
        return fluxes
    
    def _calculate_nitrogen_fluxes(self, fluxes):
        """Calculate nitrogen immobilization and mineralization"""
        immob = np.zeros_like(fluxes['pmnf_l1s1'])
        gross_nmin = np.zeros_like(fluxes['pmnf_l1s1'])
        
        # Sum immobilization (positive pmnf) and mineralization (negative pmnf)
        for flux_name in ['pmnf_l1s1', 'pmnf_l2s2', 'pmnf_l3s3', 'pmnf_s1s2', 'pmnf_s2s3', 'pmnf_s3s4']:
            immob = np.where(fluxes[flux_name] > 0, immob + fluxes[flux_name], immob)
            gross_nmin = np.where(fluxes[flux_name] < 0, gross_nmin - fluxes[flux_name], gross_nmin) 

            #     immob += fluxes[flux_name]
            #     gross_nmin -= fluxes[flux_name]
        
        # Add SOM4 mineralization
        gross_nmin -= fluxes['pmnf_s4']
        
        return immob.sum(), gross_nmin.sum()

    def _calculate_potential_hr(self, fluxes, fpi):
        """Calculate potential heterotrophic respiration with nitrogen limitation"""
        potential = fluxes.copy()
        corresponding_pmnf = {'plitr1c_loss': 'pmnf_l1s1', 
                              'plitr2c_loss': 'pmnf_l2s2',
                              'plitr3c_loss': 'pmnf_l3s3',
                              'psoil1c_loss': 'pmnf_s1s2',
                              'psoil2c_loss': 'pmnf_s2s3',
                              'psoil3c_loss': 'pmnf_s3s4'}
        
       # Apply nitrogen limitation to immobilization fluxes
        for flux_name in ['plitr1c_loss', 'plitr2c_loss', 'plitr3c_loss', 
                        'psoil1c_loss', 'psoil2c_loss', 'psoil3c_loss']:
            corresponding_pmnf_flux = corresponding_pmnf[flux_name]
            mask = fluxes[corresponding_pmnf_flux] > 0
            potential[flux_name] = np.where(mask, potential[flux_name] * fpi, potential[flux_name])
            potential[corresponding_pmnf_flux] = np.where(mask, potential[corresponding_pmnf_flux] * fpi, potential[corresponding_pmnf_flux])

        # Calculate actual heterotrophic respiration
        potential['litr1_hr'] = self.rf_l1s1 * potential['plitr1c_loss']
        potential['litr2_hr'] = self.rf_l2s2 * potential['plitr2c_loss']
        potential['litr3_hr'] = self.rf_l3s3 * potential['plitr3c_loss']
        potential['soil1_hr'] = self.rf_s1s2 * potential['psoil1c_loss']
        potential['soil2_hr'] = self.rf_s2s3 * potential['psoil2c_loss']
        potential['soil3_hr'] = self.rf_s3s4 * potential['psoil3c_loss']
        potential['soil4_hr'] = potential['psoil4c_loss']

        return potential

    def _calculate_heterotrophic_respiration(self, fluxes):
        """Calculate total heterotrophic respiration"""
        # hrsum in gC/m2/s
        hrsum = fluxes['soil1_hr'] + fluxes['soil2_hr'] + fluxes['soil3_hr'] + fluxes['soil4_hr'] + \
                fluxes['litr1_hr'] + fluxes['litr2_hr'] + fluxes['litr3_hr']
        return hrsum

    def _calculate_oxygen_limitation(self, conc_o2, n):
        """Calculate oxygen limitation factor.

        Converts gas-phase conc_o2 (mol/m³ soil) to dissolved via Henry's law,
        then compares to the anaerobic threshold (mol/m³ water).
        """
        T_K = self.t_soisno[n]
        H_cc = self.Henry_constant_O2[n] * 0.08206 * T_K  # dimensionless, C_liquid / C_gas
        afp = max(self.maxi_porosity * self.porosity_scaler - self.h2osoi_vol[n], 1e-6)
        conc_o2_dissolved = (conc_o2 / afp) * H_cc  # mol/m³ water

        if conc_o2_dissolved < self.anearobic_threshold:
            return 0.0
        else:
            return min(1.0, 2.0 * conc_o2_dissolved / (conc_o2_dissolved + 0.000375))
    
    def _apply_oxygen_limitation(self, fluxes, o2limiting):
        """Apply oxygen limitation to all fluxes"""
        limited_fluxes = {}
        for key, value in fluxes.items():
            limited_fluxes[key] = value * o2limiting
        return limited_fluxes
    
    def _update_pools(self, 
                    cn_l1, cn_l2, cn_l3,
                      cwdc, cwdn, litr1c, litr1n, litr2c, litr2n, litr3c, litr3n,
                    soil1c, soil1n, soil2c, soil2n, soil3c, soil3n, soil4c, soil4n, fluxes):
        """Update all soil pools based on fluxes"""
        # Update carbon pools
        litr1c_updated = litr1c - fluxes['litr1_hr'] - fluxes['plitr1c_loss'] * (1 - self.rf_l1s1)
        litr2c_updated = litr2c - fluxes['litr2_hr'] - fluxes['plitr2c_loss'] * (1 - self.rf_l2s2)
        litr3c_updated = litr3c - fluxes['litr3_hr'] - fluxes['plitr3c_loss'] * (1 - self.rf_l3s3)
        
        soil1c_updated = (soil1c - fluxes['soil1_hr'] - fluxes['psoil1c_loss'] * (1 - self.rf_s1s2) +
                        fluxes['plitr1c_loss'] * (1 - self.rf_l1s1))
        soil2c_updated = (soil2c - fluxes['soil2_hr'] - fluxes['psoil2c_loss'] * (1 - self.rf_s2s3) +
                        fluxes['plitr2c_loss'] * (1 - self.rf_l2s2) + fluxes['psoil1c_loss'] * (1 - self.rf_s1s2))
        soil3c_updated = (soil3c - fluxes['soil3_hr'] - fluxes['psoil3c_loss'] * (1 - self.rf_s3s4) +
                        fluxes['plitr3c_loss'] * (1 - self.rf_l3s3) + fluxes['psoil2c_loss'] * (1 - self.rf_s2s3))
        soil4c_updated = (soil4c - fluxes['soil4_hr'] +
                        fluxes['psoil3c_loss'] * (1 - self.rf_s3s4))
        
        cwdc_updated = cwdc - (fluxes['cwdc_to_litr2c'] + fluxes['cwdc_to_litr3c'])
        
        # Update nitrogen pools (simplified)
        litr1n_updated = litr1n - fluxes['plitr1c_loss'] / cn_l1
        litr2n_updated = litr2n + fluxes['cwdn_to_litr2n'] - fluxes['plitr2c_loss'] / cn_l2
        litr3n_updated = litr3n + fluxes['cwdn_to_litr3n'] - fluxes['plitr3c_loss'] / cn_l3
        soil1n_updated = soil1n - fluxes['psoil1c_loss'] / float(self.cn_s1) + fluxes['plitr1c_loss'] / cn_l1
        soil2n_updated = soil2n - fluxes['psoil2c_loss'] / float(self.cn_s2) + fluxes['plitr2c_loss'] / cn_l2 + fluxes['psoil1c_loss'] / float(self.cn_s1)
        soil3n_updated = soil3n - fluxes['psoil3c_loss'] / float(self.cn_s3) + fluxes['plitr3c_loss'] / cn_l3 + fluxes['psoil2c_loss'] / float(self.cn_s2)
        soil4n_updated = soil4n - fluxes['psoil4c_loss'] / float(self.cn_s4) + fluxes['psoil3c_loss'] / float(self.cn_s3)
        cwdn_updated = cwdn - (fluxes['cwdn_to_litr2n'] + fluxes['cwdn_to_litr3n'])
        
        return {
            'cwdc': cwdc_updated, 'cwdn': cwdn_updated,
            'litr1c': litr1c_updated, 'litr1n': litr1n_updated,
            'litr2c': litr2c_updated, 'litr2n': litr2n_updated,
            'litr3c': litr3c_updated, 'litr3n': litr3n_updated,
            'soil1c': soil1c_updated, 'soil1n': soil1n_updated,
            'soil2c': soil2c_updated, 'soil2n': soil2n_updated,
            'soil3c': soil3c_updated, 'soil3n': soil3n_updated,
            'soil4c': soil4c_updated, 'soil4n': soil4n_updated
        }

    ########## Microbial Module ################## 
    def microbe(self, n, m_dAC:float, m_dAce:float, m_dOrgAcid:float, m_dCH4:float, m_dO2:float, m_dCO2:float, m_dH2:float,
                m_dAceMethanogens:float, m_dH2Methanogens:float, m_dMethanotrophs:float, m_dAOMMethanotrophs:float,
                soiltemp:float, soilpH, 
                hco3:float,
                microbeparfilename:str, 
                microberecov:str):
        """Microbial processes: methanogenesis, methanotrophy, acetate and hydrogen
        dynamics.  Returns updated pools and net CH4/CO2/H2/O2 fluxes.
        """
        # Apply microbial recovery
        self._apply_microbial_recovery(microberecov)

        # Calculate pH effect
        pHeffect = self._calculate_pH_effect(soilpH)

        # concentration unit conversion
        self.ACConcentration = max(m_dAC / 12.0, 0) # convert gC/m3 to mol/m3
        m_dAce = max(m_dAce / 12.0, 0)  # convert gC/m3 to mol/m3
        m_dOrgAcid = max(m_dOrgAcid / 12.0, 0)  # convert gC/m3 to mol/m3

        # Acetate production (all in mol/m3/s in open system)
        AceProd, ACH2Prod, ACCO2Prod, O2_consumption = self._calculate_acetate_production(
            n, m_dO2, soiltemp, pHeffect
        )
        ACConcentration_updated = self._update_AC_concentration(self.ACConcentration, AceProd)
        self.ACConcentration = ACConcentration_updated 
        self.m_dAC[n] = ACConcentration_updated * 12.0  # convert back to gC/m3 for output 

        # pool update from acetate production (all in mol/m3) 
        m_dH2 += ACH2Prod
        m_dCO2 += ACCO2Prod 
        m_dO2 -= O2_consumption 

        # Hydrogen-based processes (all in mol/m3)
        H2AceProd, H2CH4Prod, H2Cons, CO2cons = self._calculate_hydrogen_processes(n,
            m_dO2, m_dH2, m_dCO2, m_dH2Methanogens, soiltemp, pHeffect, m_dCH4)
        m_dH2 -= H2Cons
        m_dH2 = max(m_dH2, 0.0) 
        m_dCH4 += H2CH4Prod 
        m_dCO2 -= CO2cons  
        m_dCO2 = max(m_dCO2, 0.0) 

        # Acetate consumption (mol/m3)
        AceCons = self._calculate_acetate_consumption(n,
            m_dO2, m_dAce, m_dAceMethanogens, soiltemp, pHeffect, m_dCH4, m_dCO2
        )
        m_dAce = self._update_acetate_pool(m_dAce, AceProd, H2AceProd, AceCons)

        # updated on Feb 24, 2026: was disabled. 
        m_dCO2 -=  2 * AceCons  # CO2 produced from acetoclastic methanogenesis 
        m_dCH4 += AceCons  # CH4 produced from acetoclastic methanogenesis 

        # Methane production (all in mol/m3)
        CH4Prod = AceCons + H2CH4Prod  # (AceCons, H2CH4Prod)
        
        # Methane oxidation (all in mol/m3) in aerobic methanotrophy only
        CH4Oxid = self._calculate_methane_oxidation(n,
            m_dCH4, m_dO2, m_dMethanotrophs, soiltemp, pHeffect
        )
        m_dCH4 -= CH4Oxid
        m_dCH4 = max(m_dCH4, 0.0)  # floor: substrate pool must stay non-negative
        # m_dCO2 += CH4Oxid  # CO2 produced from CH4 oxidation

        # AOM methane oxidation (all in mol/m3)
        AOMCH4Oxid = self._calculate_aom_oxidation(m_dO2, CH4Prod, n)
        m_dCH4 -= AOMCH4Oxid
        m_dCH4 = max(m_dCH4, 0.0)  # floor: substrate pool must stay non-negative

        # Oxygen consumption (all in mol/m3)
       #  methanotroph can use oxygen from both dissolved O2 and O2 in atmosphere
        CH4O2Cons = self.m_drCH4Oxid * CH4Oxid
        m_dO2 -= CH4O2Cons  # consume dissolved O2 first, then from atmosphere
        # Floor m_dO2 at the atmospheric-resupply equilibrium (Henry's law): O2 is
        # replenished from the atmosphere through air-filled pore space.
        C_O2_eq_n = (self.O2airfrac[n] * self.total_atmospheric_pressure_atm
                     * self.Henry_constant_O2[n] * 1000.0)  # mol/m³
        m_dO2 = max(m_dO2, C_O2_eq_n)

        # CO2 production (all in mol/m3)
        CO2Prod = CH4Oxid + AceCons
        H2CO2Cons = H2AceProd + H2CH4Prod

        # Microbial growth and death (biomass in mol/m3, converted at initialization)
        (m_dAceMethanogens, m_dH2Methanogens, m_dMethanotrophs,
         m_dAOMMethanotrophs) = self._calculate_microbial_dynamics(n,
            AceCons, H2CH4Prod, CH4Oxid, AOMCH4Oxid,
            m_dAceMethanogens, m_dH2Methanogens, m_dMethanotrophs, m_dAOMMethanotrophs,
            pHeffect
        )
        # Update AC pool (m_dAce, m_dOrgAcid updated in mol/m3)
        m_dAC = ACConcentration_updated * 12.0 # from mol/m3 back to gC/m3
        m_dAce = m_dAce * 12.0  # from mol/m3 back to gC/m3
        m_dOrgAcid = m_dOrgAcid * 12.0  # from mol/m3 back to gC/m3 

        # Update soil pH
        soilpH_updated = self._update_soil_pH(soilpH, m_dAce)

        # Set output variables (mol/m3-> mol/m2/s)
        ch4_prod_val = CH4Prod * self.soil_depth  # mol/m3 to mol/m2/s
        co2_prod_val = CO2Prod * self.soil_depth  # mol/m3 to mol/m2/s
        h2_prod_val = ACH2Prod * self.soil_depth  # mol/m3 to mol/m2/s
        ch4_oxid_val = (CH4Oxid  + AOMCH4Oxid) * self.soil_depth  # mol/m3 to mol/m2/s
        o2_cons_val = (CH4O2Cons + O2_consumption) * self.soil_depth  # mol/m3 to mol/m2/s

        ch4_flux = (ch4_prod_val - ch4_oxid_val) * 1e3 # mol/m2/s to mmol/m2/s
        # co2_prod_val is already multiplied by soil_depth (mol/m²/day) but
        # H2CO2Cons is in mol/m³/day, so multiply H2CO2Cons by soil_depth to keep
        # both terms in mol/m²/day before subtracting.
        co2_flux = (co2_prod_val - H2CO2Cons * self.soil_depth) * 1e3 + self.hr[n] * 1e3 /12 # mol/m2/s to mmol/m2/s

        result_dict = {
            # units in gC/m3 
            'm_dAC': m_dAC, 'm_dAce': m_dAce, 'm_dOrgAcid': m_dOrgAcid,
            # units in mol/m3
            'm_dCH4': m_dCH4, 'm_dO2': m_dO2, 'm_dCO2': m_dCO2, 'm_dH2': m_dH2,
            
            'm_dAceMethanogens': m_dAceMethanogens,
            'm_dH2Methanogens': m_dH2Methanogens,
            'm_dMethanotrophs': m_dMethanotrophs,
            'm_dAOMMethanotrophs': m_dAOMMethanotrophs,
            
            'ch4_prod': ch4_prod_val, 
            'ace_cons': AceCons,
            'h2ch4_prod':H2CH4Prod,
            'h2ace_prod': H2AceProd,
            'co2_prod': co2_prod_val,
            'acco2_prod': ACCO2Prod,    
            'ch4_oxid': ch4_oxid_val,
            'h2_prod': h2_prod_val, 
            'o2_cons': o2_cons_val, 
            'h2co2_cons': H2CO2Cons,

            # units in mmol/m2/s
            'ch4_flux': ch4_flux,
            'co2_flux': co2_flux, 
            # uintless 
            'soilpH': soilpH_updated
        }

        return result_dict
    
    def _apply_microbial_recovery(self, microberecov):
        """Apply microbial recovery factors"""
        self.m_dGrowRH2Methanogens *= microberecov
        self.m_dDeadRH2Methanogens *= microberecov
        self.m_dGrowRAceMethanogens *= microberecov
        self.m_dDeadRAceMethanogens *= microberecov
        self.m_dGrowRMethanotrophs *= microberecov
        self.m_dDeadRMethanotrophs *= microberecov
    
    def _calculate_pH_effect(self, soilpH):
        """Calculate pH effect on microbial processes"""

        if isinstance(self.for_calibration, pd.DataFrame) and 'pH' in self.for_calibration.columns:
            if not self.for_calibration['pH'].isnull().all():
                soilpH = self.for_calibration['pH'].iloc[0]
            else:
                soilpH = 4.15
        else:
            soilpH = 4.15

        # ensure DataFrame
        if isinstance(self.for_calibration, pd.Series):
            self.for_calibration = self.for_calibration.to_frame()
        
        if soilpH is None:
            soilpH = 4.15  # default value if None
        pHmax = soilpH.max() if isinstance(soilpH, np.ndarray) else 6.0
        pHmin = soilpH.min() if isinstance(soilpH, np.ndarray) else 3.0
        pHopt = np.mean([pHmin, pHmax])
        
        pHeffect = ((soilpH - pHmin) * (soilpH - pHmax) / 
                    ((soilpH - pHmin) * (soilpH - pHmax) - (soilpH - pHopt) * (soilpH - pHopt)))
        # pHeffect is along depth, convert to a single value
        if isinstance(pHeffect, (pd.Series, np.ndarray)):
            pHeffect = pHeffect.mean()

        return pHeffect
    
    def _calculate_acetate_production(self, n, m_dO2, soiltemp, pHeffect):
        """Calculate acetate production from AC (aerobic + anaerobic pathways), mol/m3."""
        aerobic_ratio = (self.maxi_porosity * self.porosity_scaler - self.h2osoi_vol[n]) 
        anaerobic_ratio = self.h2osoi_vol[n]

        if aerobic_ratio < 0:
            aerobic_ratio = 0.0001

        # Anaerobic acetate production
        AceProd_anaerobic = anaerobic_ratio * (self.m_dAceProdACmax * (self.ACConcentration / (self.ACConcentration + self.m_dKAce)) *
                      (self.m_dACMinQ10 ** ((soiltemp - self.T_ref_ch4) / 10.0)) * pHeffect)
        ACH2Prod_anaerobic = AceProd_anaerobic * 1.0
        ACCO2Prod_anaerobic = 0.5 * AceProd_anaerobic

        # Aerobic acetate production
        AceProd_aerobic = aerobic_ratio * self.m_dAceProdACmax * (self.ACConcentration / (self.ACConcentration +  self.m_dKAce)) * (self.m_dAceProdQ10 ** ((soiltemp - self.T_ref_ch4) / 10.0)) * pHeffect
        ACCO2Prod_aerobic = AceProd_aerobic
        ACH2Prod_aerobic = 0.0
    
        # Total acetate production
        AceProd = AceProd_anaerobic + AceProd_aerobic
        ACH2Prod = ACH2Prod_anaerobic + ACH2Prod_aerobic
        ACCO2Prod = ACCO2Prod_anaerobic + ACCO2Prod_aerobic
        O2_consumption = self.m_drAer * ACCO2Prod

        return AceProd, ACH2Prod, ACCO2Prod, O2_consumption

    def _update_AC_concentration(self, ACConcentration, AceProd):
        """Update AC concentration after acetate production"""
        if ACConcentration > (AceProd * 1.5):
            ACConcentration_updated = ACConcentration - AceProd * 1.5
        else:
            AceProd_adj = ACConcentration / 1.5
            ACConcentration_updated = ACConcentration - AceProd_adj * 1.5

        return max(0.0, ACConcentration_updated)
    
    def _calculate_hydrogen_processes(self,n, m_dO2, m_dH2, m_dCO2, m_dH2Methanogens, soiltemp, pHeffect, m_dCH4):
        """Calculate hydrogen-based acetate and methane production (mol/m3)."""
        def safe_exp(x, max_x=700):
            """Safely compute exp(x) without overflow."""
            if x > max_x:
                return float('inf')
            elif x < -max_x:
                return 0.0
            else:
                return math.exp(x)
        
        # AFP-based anaerobic gate: water-filled pore volume × aeration suppression.
        # afp_gate → 1 when soil is saturated; → 0 as air-filled porosity grows.
        # No Henry's law equilibrium assumed — inhibition set by bulk pore-space aeration.
        afp = max(self.maxi_porosity * self.porosity_scaler - self.h2osoi_vol[n], 0.0)
        anaerobic_ratio = self.h2osoi_vol[n]  # linear — anaerobic volume scales with water-filled pores
        afp_gate = self.K_AFP_inhib_methanogens / (self.K_AFP_inhib_methanogens + afp)
        anaerobic_scaling = anaerobic_ratio * afp_gate

        # Use Henry-equilibrium DISSOLVED substrate for H2-methanogenesis M-M
        # kinetics.  The Km values (1.98e-11 for CO2, 7.75e-9 for H2) are
        # dissolved-phase nanomolar — tuned for the substrate that methanogens
        # actually consume — so use explicit Henry's-law dissolved values here.
        diss_CO2 = (self.CO2airfrac[n] * self.total_atmospheric_pressure_atm *
                    self.Henry_constant_CO2[n] * 1000.0)   # mol/m^3 dissolved
        diss_H2  = (self.H2airfrac[n]  * self.total_atmospheric_pressure_atm *
                    self.Henry_constant_H2[n]  * 1000.0)   # mol/m^3 dissolved

        H2CH4Prod_val = anaerobic_scaling * (self.m_dGrowRH2Methanogens / self.m_dYH2Methanogens * m_dH2Methanogens *
                        diss_H2  / (diss_H2  + self.m_dKH2ProdCH4) *
                        diss_CO2 / (diss_CO2 + self.m_dKCO2ProdCH4) *
                        (self.m_dH2CH4ProdQ10 ** ((soiltemp - self.T_ref_ch4) / 10.0)) * pHeffect)

        # Hydrogen to acetate production
        if m_dH2 > self.m_dAceH2min:
            H2AceProd_val = (self.m_dH2ProdAcemax * m_dH2 / (m_dH2 + self.m_dKH2ProdAce) *
                        m_dCO2 / (m_dCO2 + self.m_dKCO2ProdAce) *
                        (self.m_dH2AceProdQ10 ** ((soiltemp - self.T_ref_ch4) / 10.0)) * pHeffect)
        else:
            H2AceProd_val = 0.0
        
        # Hydrogen consumption
        H2Cons_val = 4.0 * H2AceProd_val + 4.0 * H2CH4Prod_val
        CO2cons_val = H2CH4Prod_val   # CO2 consumed during hydrogenotrophic methanogenesis

        return H2AceProd_val, H2CH4Prod_val, H2Cons_val, CO2cons_val

    def _calculate_acetate_consumption(self, n, m_dO2, m_dAce, m_dAceMethanogens, soiltemp, pHeffect, m_dCH4, m_dCO2):
        """Calculate acetate consumption by methanogens: CH₃COOH -> CH₄ + CO₂ (mol/m3)."""
        # AFP-based anaerobic gate (same formulation as H2-methanogenesis).
        afp = max(self.maxi_porosity * self.porosity_scaler - self.h2osoi_vol[n], 0.0)
        anaerobic_ratio = self.h2osoi_vol[n]
        afp_gate = self.K_AFP_inhib_methanogens / (self.K_AFP_inhib_methanogens + afp)

        AceCons = anaerobic_ratio * afp_gate * (self.m_dGrowRAceMethanogens / self.m_dYAceMethanogens * m_dAceMethanogens *
            m_dAce / (m_dAce + self.m_dKCH4ProdAce) *
            (self.m_dCH4ProdQ10 ** ((soiltemp - self.T_ref_ch4) / 10.0)) * pHeffect
            )

        return AceCons
    
    def _update_acetate_pool(self, m_dAce, AceProd, H2AceProd, AceCons):
        """Update acetate pool"""
        if m_dAce > AceCons * 2:
            m_dAce_updated = m_dAce - AceCons * 2
        else:
            AceCons_adj = 0.45 * m_dAce
            m_dAce_updated = m_dAce - AceCons_adj * 2
        
        m_dAce_updated += (AceProd + H2AceProd)
        return max(0.0, m_dAce_updated)
    
    def _calculate_methane_oxidation(self, n, m_dCH4, m_dO2, m_dMethanotrophs, soiltemp, pHeffect):
        """Calculate methane oxidation by methanotrophs"""
        # factors range in 0 - 1
        # Population Growth = (Consumption per Cell / Efficiency) * Population Size
        growth_rate = (self.m_dGrowRMethanotrophs / self.m_dYMethanotrophs * m_dMethanotrophs)  # original version

        # Methanotroph Q10 = 3, capturing the strong CH4 sink bursts observed at
        # warm-wet events.
        temp_effect = 3** ((soiltemp - self.T_ref_ch4) / 10.0)
        # these are constrains for gas fooding methanotrophs, so we only consider o2 and ch4 limitation from soil pores
        afsp = self.maxi_porosity * self.porosity_scaler - self.h2osoi_vol[n] # airfilling porosity , methanotrophs can use gasous o2 from soil pores
        if afsp < 0:
            afsp = 0.0001 # if the calculated air-filled porosity is negative, we set it to total porosity, assuming all pores are air-filled and available for gas diffusion (this is a simplification, but it prevents negative values and allows the model to continue functioning)
        # calculate gas diffusivity limitation based on air-filled porosity (Millington & Quirk model)

        sm = self.h2osoi_vol[n]
        stress = self.soil_moisture_stress(sm, plot_components=False)
        sm_limitation = np.clip(stress, 0.0001, 1)

        # The sm_limitation curve (shared with HR) handles both wet and dry
        # moisture stress and encodes O2 availability implicitly via WFPS, so the
        # O2 and CH4 Michaelis terms are intentionally disabled here.
        oxy_val = growth_rate * temp_effect * pHeffect * sm_limitation
        return oxy_val

    def soil_moisture_stress(self, vwc, plot_components=False):
        """Calculate water scalar based on water-filled pore space()"""
        filled_proportion = vwc / (self.maxi_porosity * self.porosity_scaler)
        self.vwc_opt_max = self.vwc_opt_max_scale * self.vwc_max
        opt_min = self.vwc_opt_min
        # Guard against an inverted trapezoid when vwc_opt_max_scale collapses the
        # optimal plateau.
        opt_max = max(self.vwc_opt_max, opt_min + 0.05)
        if np.isscalar(filled_proportion):
            if filled_proportion < opt_min:
                # Dry side - linear increase
                w_scalar = filled_proportion / opt_min  # also works
                # exponential increase option
            elif filled_proportion > opt_max:
                # Wet side - exponential decline due to anaerobiosis.
                # Exponent -3; a steeper -10 gives a sharper oxic/anoxic transition.
                w_scalar = np.exp(-3.0 * (filled_proportion - opt_max) / (1.0 - opt_max))
            else:
                # Optimal range
                w_scalar = 1.0
        else:
            w_scalar = np.ones_like(filled_proportion)
            dry_mask = filled_proportion < opt_min
            wet_mask = filled_proportion > opt_max

            w_scalar[dry_mask] = filled_proportion[dry_mask] / opt_min
            # Exponent -3; a steeper -10 gives a sharper oxic/anoxic transition.
            w_scalar[wet_mask] = np.exp(-3.0 * (filled_proportion[wet_mask] - opt_max) / (1.0 - opt_max))
            # Alternative (sharper transition):
            # w_scalar[wet_mask] = np.exp(-10.0 * (filled_proportion[wet_mask] - opt_max) / (1.0 - opt_max))

        if plot_components:
            filled_proportion_plot = np.linspace(0, 1, 100)
            w_scalar_plot = np.ones_like(filled_proportion_plot)
            dry_mask_plot = filled_proportion_plot < opt_min
            wet_mask_plot = filled_proportion_plot > opt_max
            w_scalar_plot[dry_mask_plot] = filled_proportion_plot[dry_mask_plot] / opt_min
            w_scalar_plot[wet_mask_plot] = np.exp(-3.0 * (filled_proportion_plot[wet_mask_plot] - opt_max) / (1.0 - opt_max))
            
            plt.figure(figsize=(8, 6))
            plt.rcParams['font.size'] = 25  # Set to your desired font size
            plt.plot(filled_proportion_plot, w_scalar_plot, label='Water Stress', linewidth=4)
            plt.axvline(opt_min, color='r', linestyle='--', label='Opt Min VWC', linewidth=4)
            plt.axvline(opt_max, color='g', linestyle='--', label='Opt Max VWC', linewidth=4)
            plt.xlabel('Fractional Water-Filled Pore Space (VWC / VWC_sat)', fontsize=15)
            plt.ylabel('Water Stress of Aerobic Respiration', fontsize=15)
            plt.legend(frameon=False)
            plt.show()

        return np.clip(w_scalar, 0.0, 1.0)

    def _calculate_aom_oxidation(self, m_dO2, CH4Prod, n):
        """Calculate anaerobic oxidation of methane (AOM)"""
        anaerobic_ratio = self.h2osoi_vol[n]
        AOMCH4Oxid = self.AOM * CH4Prod * anaerobic_ratio

        return AOMCH4Oxid
    
    def _calculate_microbial_dynamics(self, n, AceCons, H2CH4Prod, CH4Oxid, AOMCH4Oxid,
                                    m_dAceMethanogens, m_dH2Methanogens, m_dMethanotrophs, m_dAOMMethanotrophs,
                                    pHeffect):
        """Calculate microbial growth and death (Monod biomass-substrate dynamics)."""
        # Calculate carrying capacity factors (0-1), clip to prevent >1
        
        #     # Growth limitation factors (0-1)
        
        # Ace methanogens
        AceMethanogenGrowth = self.m_dYAceMethanogens * AceCons # * growth_limitation_ace  # convert from mol/m3 to mol/m3/d
        AceMethanogenDying = (self.m_dDeadRAceMethanogens * m_dAceMethanogens * # (1.0 + carrying_ace**2) *
                             (m_dAceMethanogens / (m_dAceMethanogens + 0.01)))  # rate at d-1, matched with parameters
        
        # H2 methanogens
        H2MethanogenGrowth = self.m_dYH2Methanogens * H2CH4Prod #*  growth_limitation_h2  # convert from mol/m3/s to mol/m3/d
        H2MethanogenDying = (self.m_dDeadRH2Methanogens * m_dH2Methanogens * # (1.0 + carrying_h2**2) *
                            (m_dH2Methanogens / (m_dH2Methanogens + 0.01)))  # rate at d-1, matched with parameters

        # Methanotrophs
        MethanotrophGrowth = self.m_dYMethanotrophs * CH4Oxid #*  growth_limitation_mt  # convert from mol/m3/s to mol/m3/h
        # Saturation threshold 0.0001 so death reaches its nominal rate at typical
        # m_d (~0.004) instead of being suppressed, eliminating the long-term
        # biomass drift that otherwise accelerates the CH4 sink at long-time-series
        # sites.
        MethanotrophDying = (self.m_dDeadRMethanotrophs * m_dMethanotrophs *
                            (m_dMethanotrophs / (m_dMethanotrophs + 0.0001)))  # rate at d-1
        
        # AOM methanotrophs (simplified, disabled in the original model)
        AOMMethanotrophGrowth = 0.0  # self.m_dYAOMMethanotrophs * AOMCH4Oxid * pHeffect
        AOMMethanotrophDying = 0.0  # self.m_dDeadRAOMMethanotrophs * m_dAOMMethanotrophs * pHeffect
        
        # Update microbial populations
        m_dAceMethanogens_updated = m_dAceMethanogens + (AceMethanogenGrowth - AceMethanogenDying)
        m_dH2Methanogens_updated = m_dH2Methanogens + (H2MethanogenGrowth - H2MethanogenDying)
        m_dMethanotrophs_updated = m_dMethanotrophs + (MethanotrophGrowth - MethanotrophDying) 
        m_dAOMMethanotrophs_updated = m_dAOMMethanotrophs + (AOMMethanotrophGrowth - AOMMethanotrophDying)
        
        # Ensure non-negative populations
        if isinstance(m_dMethanotrophs_updated, (float, int, np.ndarray)):
            m_dAceMethanogens_updated = max(0.0, m_dAceMethanogens_updated)
            m_dH2Methanogens_updated = max(0.0, m_dH2Methanogens_updated)
            m_dMethanotrophs_updated = max(0.0, m_dMethanotrophs_updated)
            m_dAOMMethanotrophs_updated = max(0.0, m_dAOMMethanotrophs_updated)
            # prevent overflow
            m_dAceMethanogens_updated = min(m_dAceMethanogens_updated, self.m_dMaxAceMethanogens)
            m_dH2Methanogens_updated = min(m_dH2Methanogens_updated, self.m_dMaxH2Methanogens)
            m_dMethanotrophs_updated = min(m_dMethanotrophs_updated, self.m_dMaxMethanotrophs)
            m_dAOMMethanotrophs_updated = min(m_dAOMMethanotrophs_updated, self.m_dMaxMethanotrophs)
        else: 
            import pytensor.tensor as pt
            m_dAceMethanogens_updated = pt.maximum(0.0, m_dAceMethanogens_updated)
            m_dH2Methanogens_updated = pt.maximum(0.0, m_dH2Methanogens_updated)
            m_dMethanotrophs_updated = pt.maximum(0.0, m_dMethanotrophs_updated)
            m_dAOMMethanotrophs_updated = pt.maximum(0.0, m_dAOMMethanotrophs_updated)
            # prevent overflow
            m_dAceMethanogens_updated = pt.minimum(m_dAceMethanogens_updated, self.m_dMaxAceMethanogens)
            m_dH2Methanogens_updated = pt.minimum(m_dH2Methanogens_updated, self.m_dMaxH2Methanogens)
            m_dMethanotrophs_updated = pt.minimum(m_dMethanotrophs_updated, self.m_dMaxMethanotrophs)
            m_dAOMMethanotrophs_updated = pt.minimum(m_dAOMMethanotrophs_updated, self.m_dMaxMethanotrophs)

        return (m_dAceMethanogens_updated, m_dH2Methanogens_updated,
                m_dMethanotrophs_updated, m_dAOMMethanotrophs_updated)
    
    def _update_soil_pH(self, soilpH, m_dAce):
        """Update soil pH based on acetate dissociation"""
        
        if soilpH > 5.5:
            # pKa for acetic acid is 4.75, 0.42% dissociation
            H_plus = 10 ** (-self.originalsoilph) + 0.0042 * 0.001 * m_dAce
            soilpH_updated = -math.log10(H_plus) if H_plus > 0 else soilpH
            soilpH_updated = max(5.5, soilpH_updated)
        else:
            soilpH_updated = soilpH
        
        return soilpH_updated

    def _write_output(self, scenario):
        """Write output to file"""
        import os 
        import pandas as pd 
        from datetime import datetime

        out_df = np.column_stack((
            # datetime,
            self.m_dAC, self.m_dAce, self.m_dOrgAcid, self.m_dCH4, self.m_dO2, self.m_dCO2, self.m_dH2,
            self.m_dAceMethanogens, self.m_dH2Methanogens, self.m_dMethanotrophs, self.m_dAOMMethanotrophs,
            self.ch4_prod, self.co2_prod, self.h2_prod, 
            self.ch4_oxid, self.o2_cons, self.conc_ch4, self.conc_co2, self.conc_h2, 
            self.ace_cons, self.h2ch4_prod, self.acco2_prod, self.h2ace_prod, self.h2co2_cons,
            self.net_ch4_flux, self.net_co2_flux, 
            self.soilpH
        ))

        out_cols = ['m_dAC', 'm_dAce', 'm_dOrgAcid', 
                    'm_dCH4', 'm_dO2', 'm_dCO2', 'm_dH2',
                'm_dAceMethanogens', 'm_dH2Methanogens', 'm_dMethanotrophs', 'm_dAOMMethanotrophs',
                'ch4_prod', 'co2_prod', 'h2_prod', 'ch4_oxid', 'o2_cons', 'conc_ch4', 'conc_co2', 'conc_h2',
                'ace_cons', 'h2ch4_prod', 'acco2_prod', 'h2ace_prod', 'h2co2_cons',
                'flux_ch4', 'flux_co2', 'soilpH']
        
        out_df = pd.DataFrame(out_df, columns=out_cols)
        return out_df

    def set_calibration_mode(self, 
                             params_dict, 
                             ):
        """Enable calibration mode and set parameters to calibrate (params_dict)."""
        self.calibration_mode = True
        self.calibrated_params = params_dict
        
        # Apply parameters immediately
        self._apply_calibrated_parameters()
    
    def _apply_calibrated_parameters(self):
        """Apply calibrated parameters to model state"""
        try:
            for param_name, param_value in self.calibrated_params.items():
                if hasattr(self, param_name):
                    setattr(self, param_name, param_value)
                    print(f"Set {param_name} = {param_value}")
                else:
                    print(f"Warning: Parameter {param_name} not found in model")
        except Exception as e:
            print(f"Error applying calibrated parameters: {e}")

    def _apply_parameters_to_model(self, param_names, param_values):
        """Apply parameters to CLM model"""
        params_dict = dict(zip(param_names, param_values))
        for param_name, param_value in params_dict.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                print(f"Warning: Parameter {param_name} not found in model")

