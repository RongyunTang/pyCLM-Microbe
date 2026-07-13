import pandas as pd
import numpy as np
import re
import pickle 
from clm_microbe.utils import units as unit_converter
from typing import Optional, Dict
from clm_microbe.io.config_loader import ModelConfig


def load_raw_excel(cfg: ModelConfig) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load observations from the provided paths with robust path & schema checks.
    """
    # read values from cfg (ModelConfig object) 
    input_file = cfg.observations 
    # site_name = cfg.site_name
    site_type = cfg.site_type 

    # inputs are stored in a dictionary, keys are time series or layer-specific properties 
    # dict_ts = {}
    # dict_layer = {}
    

    if site_type not in ['manual', 'auto', 'manual_in_plots']:
        raise ValueError(f"site_type {site_type} not recognized. Should be 'manual' or 'auto'.")
    else:
        # read every sheet in the input excel file, classify into time series or layer-specific properties by the horizon column 
        # if no values in the horizon column, then it is a time series 
        xls = pd.ExcelFile(input_file)
        sheet_names = xls.sheet_names
        # create input_dict to store all sheets
        input_dict = dict.fromkeys(sheet_names)
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            # add key and value to the input_dict
            input_dict[sheet] = df
    return input_dict 

def preparing_inputs_dict(input_dict: dict):
    # load observations from inputs dict
    # all_vars = ts_vars + layer_vars 
    # print("Time series variables available:", ts_vars)
    # print("Layer-specific variables available:", layer_vars)

    # fluxes and air temperature are only measured by time series
    flux_ch4_obs = input_dict.get("FluxCH4")
    flux_co2_obs = input_dict.get("FluxCO2")
    AT_obs = input_dict.get("AT")
    # soil moisture is only measured by layers
    SM_obs = input_dict.get("SM")

    if 'Pressure' in input_dict: # only exists for auto sites by time series
        pressure_obs = input_dict.get("Pressure")
        PVRT_obs = input_dict.get("PV_rt")
    else:
        pressure_obs = None 
        PVRT_obs = None

    if 'pH' in input_dict: # pH only exists for manual sites by layer
        pH_obs = input_dict.get("pH")
    else:
        pH_obs = None

    if 'TN_TC' in input_dict: # TN_TC only exists for manual sites by layer
        TN_TC_obs = input_dict.get("TN_TC")
        TC_obs = TN_TC_obs[TN_TC_obs['measurement'].str.contains('TC', flags=re.IGNORECASE, regex=True)]
        # select unit as gC/kg
        TC_obs = TC_obs[TC_obs['unit'].str.contains('gC/kg', flags=re.IGNORECASE, regex=True)]
        
        # TN_obs = TN_TC_obs[TN_TC_obs['measurement'].str.contains('TN', flags=re.IGNORECASE, regex=True)]
        # TN_obs = TN_obs[TN_obs['unit'].str.contains('gN/kg', flags=re.IGNORECASE, regex=True)]
       
    else:
        TN_TC_obs = None
        TC_obs = None  

    if 'profile(gas)' in input_dict: # profile(gas) only exists for auto sites by time series
        profiles_obs = input_dict.get("profile(gas)")
        profile_CH4 = profiles_obs[profiles_obs['measurement']=='ConCH4'] if profiles_obs is not None else None
        profile_CO2 = profiles_obs[profiles_obs['measurement']=='ConCO2'] if profiles_obs is not None else None
        profile_O2 = profiles_obs[profiles_obs['measurement']=='ConO2'] if profiles_obs is not None else None
    else:
        profiles_obs = None
        profile_CH4 = None
        profile_CO2 = None
        profile_O2 = None

    if 'ST' in input_dict: # TS points for mannual sites and profile for auto sites
        ST_obs = input_dict.get("ST")
    else:
        ST_obs = None

    if 'VSM' in input_dict: # VSM only exists for manual sites by layer
        VSM_obs = input_dict.get("VSM")
    else:
        VSM_obs = None

    inputs_dict = {
        'drivers': [ST_obs, SM_obs, TC_obs, pH_obs, AT_obs], # inputs that drive the model
        'for_calibration': [profile_CH4, profile_CO2, profile_O2, flux_ch4_obs, flux_co2_obs, pressure_obs, PVRT_obs] # inputs for model calibration
    }
    return inputs_dict

def preprocess_units(inputs_dict: dict):
    # preprocess the temporal and spatial scales of the observations 
    drivers = inputs_dict['drivers']
    for_calibration = inputs_dict['for_calibration']
    ST_obs, SM_obs, TC_obs, pH_obs, AT_obs = drivers
    profiles_CH4, profiles_CO2, profiles_O2, flux_ch4_obs, flux_co2_obs, pressure_obs, PVRT_obs = for_calibration

    # remove empty rows with NaN values in the value column 
    ST_obs = remove_empty_rows(ST_obs)
    SM_obs = remove_empty_rows(SM_obs)
    TC_obs = remove_empty_rows(TC_obs)
    pH_obs = remove_empty_rows(pH_obs)

    profiles_CH4 = remove_empty_rows(profiles_CH4)
    profiles_CO2 = remove_empty_rows(profiles_CO2)
    profiles_O2 = remove_empty_rows(profiles_O2)
    flux_ch4_obs = remove_empty_rows(flux_ch4_obs)
    flux_co2_obs = remove_empty_rows(flux_co2_obs)
    AT_obs = remove_empty_rows(AT_obs)
    pressure_obs = remove_empty_rows(pressure_obs)
    PVRT_obs = remove_empty_rows(PVRT_obs)

    # Check units and convert to model units where needed.
    # Kept as-is: ST/AT (degC), SM (%), pH (unitless).
    # Converted: TC gC/kg->mol/m3; profile CO2/CH4/O2 ppm->nmol/m3; flux CH4/CO2 mgC/m2/hr->umol/m2/h.
    #### keep AT and ST data in degC ####

    #### conver units for other variables #### 
    # # AT: degC to K
    # if AT_obs is not None and AT_obs['unit'].nunique() == 'degC':
    #     AT_obs.loc[AT_obs['unit'] == 'degC', 'value'] = unit_converter.degC_to_K(AT_obs.loc[AT_obs['unit'] == 'degC', 'value'])
    # else:
    #     pass
    # # ST: degC to K 
    # if ST_obs is not None and ST_obs['unit'].nunique() == 'degC':
    #     ST_obs.loc[ST_obs['unit'] == 'degC', 'value'] = unit_converter.degC_to_K(ST_obs.loc[ST_obs['unit'] == 'degC', 'value'])
    # else:
    #     pass  
   
    # # SM: % to fraction of 1 
    # if SM_obs is not None and SM_obs['unit'].nunique() == '%': 
    #     SM_obs.loc[SM_obs['unit'] == '%', 'value'] = unit_converter.percent_to_fraction(SM_obs.loc[SM_obs['unit'] == '%', 'value'])
    # else:
    #     pass
    # pH: unitless to unitless (no conversion needed)
    
    # TN_TC: gC/kg to mol/m3 
    if TC_obs is not None:
        drainage_types = TC_obs['drainage'].copy()
        BD_mapping = {'upland': cfg.constants['BD']['upland'],
                      'transition': cfg.constants['BD']['transition'],
                      'wetland': cfg.constants['BD']['wetland']}
        drainage_types = drainage_types.map(BD_mapping) # map treatment to corresponding BD values 
        BD_values = drainage_types.values
        TC_obs["value"] = unit_converter.gC_per_kg_to_mol_per_m3(TC_obs["value"].values, BD=BD_values) 
        TC_obs['unit'] = 'mol/m3' 
    else:
        pass

    # soil gas profiles: ppm to nmol/m3
    if profiles_CH4 is not None:
        profiles_CH4["value"] = unit_converter.ppm_to_nmol_m3(profiles_CH4["value"])
        profiles_CO2["value"] = unit_converter.ppm_to_nmol_m3(profiles_CO2["value"])
        profiles_O2["value"] = unit_converter.ppm_to_nmol_m3(profiles_O2["value"])
        profiles_CH4['unit'] = 'nmol/m3'
        profiles_CO2['unit'] = 'nmol/m3'
        profiles_O2['unit'] = 'nmol/m3'

    # Unit-aware flux conversion.  Two xlsx provenances are possible:
    #   (a) unit label is "mgC/m2/hr" → convert to umol/m2/s.
    #   (b) unit label is "umol/m2/s" → already converted; do NOT re-apply.
    # We inspect the `unit` column on each flux DataFrame and skip the conversion
    # when the unit is already in the target form.
    def _needs_conversion(df) -> bool:
        if df is None or 'unit' not in df.columns:
            return True  # safest default; old behaviour
        units = df['unit'].dropna().astype(str).str.strip().str.lower().unique()
        # If any unit is umol/... or already-µmol form, treat as already-converted.
        already = any(('umol' in u or 'µmol' in u) for u in units)
        return not already

    if flux_ch4_obs is not None:
        if _needs_conversion(flux_ch4_obs):
            flux_ch4_obs["value"] = unit_converter.mgC_m2_hr_to_umol_CH4_m2_s(flux_ch4_obs["value"])
            flux_ch4_obs['unit'] = 'umol/m2/s'
        # else: already umol/m2/s — leave value alone (just normalise the label)
        else:
            flux_ch4_obs['unit'] = 'umol/m2/s'
    if flux_co2_obs is not None:
        if _needs_conversion(flux_co2_obs):
            flux_co2_obs["value"] = unit_converter.mgC_m2_hr_to_umol_CO2_m2_s(flux_co2_obs["value"])
            flux_co2_obs['unit'] = 'umol/m2/s'
        else:
            flux_co2_obs['unit'] = 'umol/m2/s'

    # updated_driver = pd.concat([ST_obs, SM_obs, TC_obs, pH_obs, AT_obs], axis=0, ignore_index=True)
    # updated_for_calibration = pd.concat([profiles_CH4, profiles_CO2, profiles_O2, flux_ch4_obs, flux_co2_obs, pressure_obs, PVRT_obs], axis=0, ignore_index=True)
    # out_dict = {
    #     'drivers': updated_driver,
    #     'for_calibration': updated_for_calibration
    # } 
    
    out_dict = {
        'ST': ST_obs,
        'SM': SM_obs,
        'AT': AT_obs,
        'flux_ch4': flux_ch4_obs,
        'flux_co2': flux_co2_obs,
        'TC': TC_obs,
        'pH': pH_obs,
        'profile_ch4': profiles_CH4,
        'profile_co2': profiles_CO2,
        'profile_o2': profiles_O2,
        'pressure': pressure_obs,
        'PVRT': PVRT_obs
    }

    return out_dict

def has_valid_depth_or_horizon(df):
    """Check if DataFrame has valid depth or horizon data for layered processing."""
    if df is None or df.empty:
        return False
    
    # Check if depth column exists and has valid values
    if 'depth' in df.columns:
        depth_valid = df['depth'].notna().any() and (df['depth'] != '').any()
        if depth_valid:
            return True
    
    # Check if horizon column exists and has valid values
    if 'horizon' in df.columns:
        horizon_valid = df['horizon'].notna().any() and (df['horizon'] != '').any()
        if horizon_valid:
            return True
    
    return False

def get_layer_groups(df):
    """Get unique layers for processing - prioritize depth over horizon."""
    if 'depth' in df.columns and df['depth'].notna().any() and (df['depth'] != '').any():
        return df['depth'].dropna().unique()
    elif 'horizon' in df.columns and df['horizon'].notna().any() and (df['horizon'] != '').any():
        return df['horizon'].dropna().unique()
    else:
        return [None]

def process_time_series_for_layer(g, layer_name, layer_type):
    """Process time series for a single depth/horizon layer."""
    print(f"  Processing {layer_type} layer: {layer_name}")
    
    g = g.sort_values('datetime').reset_index(drop=True)
    g['time_diff'] = g['datetime'].diff()
    
    # Identify gaps between 2h and 12h
    gap_indices = g.index[
        (g['time_diff'] > pd.Timedelta(hours=2)) & 
        (g['time_diff'] < pd.Timedelta(hours=12))
    ].tolist()

    new_rows = []
    for idx in gap_indices:
        if idx == 0:  # Skip first row (no previous row)
            continue
            
        prev_time = g.iloc[idx-1]['datetime']
        next_time = g.iloc[idx]['datetime']
        gap_hours = (next_time - prev_time).total_seconds() / 3600
        
        # Number of rows to insert = (gap / 2h) - 1
        n_insert = max(0, int(gap_hours // 2) - 1)
        
        for j in range(n_insert):
            new_date = prev_time + pd.Timedelta(hours=2*(j+1))
            new_row = g.iloc[idx-1].copy()
            new_row['datetime'] = new_date
            new_row['value'] = np.nan  # Will be interpolated
            new_rows.append(new_row)

    if new_rows:
        g = pd.concat([g, pd.DataFrame(new_rows)], ignore_index=True)
        g = g.sort_values('datetime').reset_index(drop=True)
        g['value'] = g['value'].interpolate(method='linear')
    
    g['timestep'] = 2 * 3600
    return g

def preprocess_time_steps(cfg, inputs_dict):
    all_keys = list(inputs_dict.keys())

    # 1. check missing keys
    expected_driver_keys = ['ST', 'AT', 'SM', 'pH', 'TC']
    expected_calibration_keys = ['profile_ch4', 'profile_co2', 'profile_o2', 'flux_ch4', 'flux_co2', 'pressure', 'PVRT']

    # 2. check required time steps
    # time_steps = cfg.no_mcmc_single_site['dt']  # seconds
    # if time_steps <= 0:
    #     raise ValueError(f"Invalid time step: {time_steps}")
    
    # add datetime column 
    for key, df in inputs_dict.items():
        if df is not None:
            datetime_str = df['DATE'].astype(str) + ' ' + df['Time_of_day'].astype(str)
            df = df.drop(columns=['DATE', 'Time_of_day'], errors='ignore') 
            df['datetime'] = pd.to_datetime(datetime_str, errors='coerce')
            df = df.sort_values(by='datetime')
            inputs_dict[key] = df

    # do half-month modeling for manual sites and 2hrs modeling for auto sites
    site_name = cfg.no_mcmc_single_site['site_id']
    var_rows = []
    if site_name in cfg.site_list['manual'] or site_name in cfg.site_list['manual_in_plots']:
        if "SM" in all_keys:
            SM_obs = inputs_dict["SM"]
            # select only V values not H values for horizon column
            SM_obs = SM_obs[SM_obs['horizon'] == 'V'] if SM_obs is not None else None
            inputs_dict["SM"] = SM_obs
        # Manual sites → half-month timestep
        for key, df in inputs_dict.items():
            if df is not None and not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.normalize()
                df_list = []
                horizons = df['horizon'].unique() if 'horizon' in df.columns else [None]
                for horizon in horizons:
                    if 'horizon' in df.columns and not pd.isna(horizon):
                        g = df[df['horizon'] == horizon].copy()
                    else:
                        g = df.copy()
                    g = g.sort_values('datetime')
                    g['time_diff'] = (g['datetime'] - g['datetime'].shift()).dt.days
                    # replacing NAN of time_diff with 15 for the first row
                    g.iloc[0, g.columns.get_loc('time_diff')] = 15
                    new_rows = g.loc[(g['time_diff'] > 30) & (g['time_diff'] < 70)].copy()
                    new_rows_ids = new_rows.index.tolist()
                    new_dates = [g.loc[i-1, 'datetime'] + pd.Timedelta(days=15) for i in new_rows_ids]
                    new_rows['datetime'] = new_dates
                    new_rows['value'] = np.nan  # value will be interpolated
                    if not new_rows.empty:
                        g = pd.concat([g, new_rows], ignore_index=True)
                    g = g.sort_values('datetime').reset_index(drop=True)
                    # linear interpolate value
                    g['value'] = g['value'].interpolate(method='linear')
                    g['time_diff'] = (g['datetime'] - g['datetime'].shift()).dt.days

                    g.iloc[0, g.columns.get_loc('time_diff')] = 15
                    # time_diff larger than 75d are set to 15d
                    g.loc[g['time_diff'] > 75, 'time_diff'] = 15
                    g['timestep'] = g['time_diff'] * 24 * 3600  # convert days to seconds
                    print(f"Inserted {len(new_rows)} new rows, with total {len(g)} rows for {key} with horizons {horizons}.")
                    df_list.append(g)
                if len(df_list) > 1:
                    df = pd.concat(df_list, ignore_index=True)
                else:
                    df = df_list[0] if df_list else pd.DataFrame()
                df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d')
                inputs_dict[key] = df
    else:
        # Auto sites → 2-hour timestep
        print("Auto sites detected, processing 2-hour timesteps.") 
        # for key, df in inputs_dict.items():         
        #     if df is not None and not df.empty:
        #         print(f"Processing {key} df for time steps.") 
        #         df['datetime'] = pd.to_datetime(df['datetime']).dt.round('min')
        #         df_list = []
        #         horizons = df['horizon'].unique() if 'horizon' in df.columns else [None]
        #         for horizon in horizons:
        #             if 'horizon' in df.columns and not pd.isna(horizon):
        #                 g = df[df['horizon'] == horizon].copy()
        #             else:
        #                 g = df.copy()

        #             g = g.sort_values('datetime').reset_index(drop=True)
        #             g['time_diff'] = g['datetime'].diff()
        #             # identify gaps between 2h and 12h
        #             gap_indices = g.index[(g['time_diff'] > pd.Timedelta(hours=2)) & 
        #                                 (g['time_diff'] < pd.Timedelta(hours=12))].tolist()

        #             new_rows = []
        #             for i in gap_indices:
        #                 prev_row_idx = g.index[g.index.get_loc(i) - 1]  # Get previous index
        #                 prev_time = g.loc[prev_row_idx, 'datetime']
        #                 next_time = g.loc[i, 'datetime']
        #                 # prev_time = g.iloc[i-1]['datetime']
        #                 # next_time = g.iloc[i]['datetime']
        #                 gap_hours = (next_time - prev_time).total_seconds() / 3600
                        
        #                 # number of rows to insert = (gap / 2h) - 1
        #                 n_insert = int(gap_hours // 2) - 1
                        
        #                 for j in range(n_insert):
        #                     new_date = prev_time + pd.Timedelta(hours=2*(j+1))
        #                     new_row = g.iloc[i-1].copy()
        #                     new_row['datetime'] = new_date
        #                     new_row['value'] = np.nan  # will be interpolated
        #                     new_rows.append(new_row)

        #             if new_rows:
        #                 g = pd.concat([g, pd.DataFrame(new_rows)], ignore_index=True)
        #                 g = g.sort_values('datetime')
        #                 g['value'] = g['value'].interpolate(method='linear')
        #             g['timestep'] = 2 * 3600
        #             df_list.append(g)
        #         # merge horizons back
        #         if len(df_list) > 1:
        #             df = pd.concat(df_list, ignore_index=True)
        #         elif len(df_list) == 1:
        #             df = df_list[0]
        #         else:
        #             df = pd.DataFrame()

        #         df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        #         inputs_dict[key] = df
        #         nrows = len(df)

        #     var_rows.append({'variable': key, 'rows': nrows})
        #     print(f"{key} has {nrows} rows after preprocessing and checking.")


        # Main processing loop
        for key, df in inputs_dict.items():         
            if df is not None and not df.empty:
                print(f"Processing {key} df for time steps.") 
                df['datetime'] = pd.to_datetime(df['datetime']).dt.round('min')
                df_list = []
                
                # Check if we need to process by layers
                if has_valid_depth_or_horizon(df):
                    layers = get_layer_groups(df)
                    layer_type = 'depth' if 'depth' in df.columns and df['depth'].notna().any() else 'horizon'
                    print(f"  Found {len(layers)} {layer_type} layers: {list(layers)}")
                    
                    for layer in layers:
                        if layer is not None:
                            if layer_type == 'depth':
                                g = df[df['depth'] == layer].copy()
                            else:
                                g = df[df['horizon'] == layer].copy()
                        else:
                            g = df[df[layer_type].isna() | (df[layer_type] == '')].copy()
                        
                        if not g.empty:
                            processed_g = process_time_series_for_layer(g, layer, layer_type)
                            df_list.append(processed_g)
                else:
                    # Process as single layer if no valid depth/horizon
                    print("  No valid depth/horizon found, processing as single layer")
                    g = df.copy()
                    processed_g = process_time_series_for_layer(g, 'single_layer', 'single')
                    df_list.append(processed_g)
                
                # Merge all processed layers back
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                    df = df.sort_values(['datetime', 'depth' if 'depth' in df.columns else 'horizon'], 
                                    na_position='first')
                else:
                    df = pd.DataFrame()

                df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                inputs_dict[key] = df
                nrows = len(df)
                
                print(f"  {key} now has {nrows} rows across {len(df_list)} layers after processing")
            else:
                print(f"{key} is None or empty.") 
                nrows = 0

            var_rows.append({'variable': key, 'rows': nrows})
            print(f"{key} has {nrows} rows after preprocessing and checking.\n")

    return inputs_dict

# def preprocess_depth(inputs_dict):
#     """
#     Assign depth ranges (start_cm, end_cm, depth_cm) to each horizon in all dataframes.
#     Modifies inputs_dict in place.
#     """
#     # Define horizon → depth mapping (cm)
#     horizon_depths = {
#         # Manual horizons
#         "OT": (0, -5),      # O top
#         "OB": (-5, -10),     # O bottom
#         "MA": (-10, -20),    # Mineral A
#         "MB": (-20, -30),    # Mineral B
#         "V": (0, -10),      # Vertical peat/layer
#         "H": (0, -1),     # Horizontal peat/layer

#         # Profiles
#         "TOP": (0, -5),
#         "BOTTOM": (-5, -10),
#         # Default case (no horizon info)
#         None: (0, 0),
#         "nan": (0, 0),
#     }

#     for key, df in inputs_dict.items():
#         if df is not None and not df.empty:
#             if "horizon" in df.columns:
#                 df = df.copy()
#                 # Assign depths
#                 df["start_cm"] = df["horizon"].map(
#                     lambda h: horizon_depths.get(str(h), (None, None))[0]
#                 )
#                 df["end_cm"] = df["horizon"].map(
#                     lambda h: horizon_depths.get(str(h), (None, None))[1]
#                 )
#                 # df["depth_cm"] = df[["start_cm", "end_cm"]].mean(axis=1)
#                 df["depth_cm"] = abs(df["end_cm"] - df["start_cm"])
#                 inputs_dict[key] = df
#                 print(f"Assigned depths for {key} with horizons {df['horizon'].unique()}.")
#             else:
#                 print(f"{key} has no horizon column, skipping depth assignment.")
#     return inputs_dict

def preprocess_depth(inputs_dict):
    """
    Assign depth ranges (start_cm, end_cm, depth_cm) to each horizon in dataframes.
    Only assigns horizon depths if no valid depth information already exists.
    Modifies inputs_dict in place.
    """
    # Define horizon → depth mapping (cm)
    horizon_depths = {
        # Manual horizons
        "OT": (0, -5),      # O top
        "OB": (-5, -10),    # O bottom
        "MA": (-10, -20),   # Mineral A
        "MB": (-20, -30),   # Mineral B
        "V": (0, -10),      # Vertical peat/layer
        "H": (0, -1),       # Horizontal peat/layer

        # Profiles
        "TOP": (0, -5),
        "BOTTOM": (-5, -10),
        # Default case (no horizon info)
        None: (0, 0),
        "nan": (0, 0),
        "": (0, 0),
    }

    def has_valid_depth_info(df):
        """Check if DataFrame already has valid depth information."""
        depth_columns = ['depth', 'depth_cm', 'start_cm', 'end_cm']
        
        for col in depth_columns:
            if col in df.columns:
                # Check if column has any non-null, non-zero values
                if df[col].notna().any():
                    valid_values = df[col].dropna()
                    if len(valid_values) > 0 and not (valid_values == 0).all():
                        print(f"  Found valid depth information in '{col}' column")
                        return True
        return False

    for key, df in inputs_dict.items():
        if df is not None and not df.empty:
            print(f"Processing depth information for {key}...")
            
            # Check if valid depth information already exists
            if has_valid_depth_info(df):
                print(f"  {key} already has valid depth information, skipping horizon depth assignment.")
                continue
            
            # If no valid depth info, check for horizon column
            if "horizon" in df.columns:
                df = df.copy()
                
                # Clean horizon values
                df['horizon_clean'] = df['horizon'].astype(str).str.strip().replace({'nan': '', 'None': ''})
                
                # Filter out empty horizons
                valid_horizons = df['horizon_clean'] != ''
                
                if valid_horizons.any():
                    # Assign depths only for valid horizons
                    df.loc[valid_horizons, "start_cm"] = df.loc[valid_horizons, "horizon_clean"].map(
                        lambda h: horizon_depths.get(h, (None, None))[0]
                    )
                    df.loc[valid_horizons, "end_cm"] = df.loc[valid_horizons, "horizon_clean"].map(
                        lambda h: horizon_depths.get(h, (None, None))[1]
                    )
                    
                    # Calculate depth only where we have valid start/end
                    valid_depths = df["start_cm"].notna() & df["end_cm"].notna()
                    df.loc[valid_depths, "depth_cm"] = abs(df.loc[valid_depths, "end_cm"] - df.loc[valid_depths, "start_cm"])
                    
                    # Drop temporary column
                    df = df.drop('horizon_clean', axis=1)
                    
                    horizons_assigned = df.loc[valid_horizons, 'horizon'].unique()
                    print(f"  Assigned depths for horizons: {list(horizons_assigned)}")
                else:
                    df = df.drop('horizon_clean', axis=1)
                    print(f"  {key} has horizon column but no valid horizon values, skipping depth assignment.")
                
                inputs_dict[key] = df
            else:
                print(f"  {key} has no horizon column and no depth information, cannot assign depths.")
    
    return inputs_dict



def merge_vars_to_df(input_dict):
    merged_ts_df = pd.DataFrame()
    merged_static_df = pd.DataFrame()
    for key in input_dict:
        if key in ['ST', 'SM', 'AT', 'flux_ch4', 'flux_co2', 'pressure', 'PVRT']:
            df = input_dict[key]
            if df is not None and not df.empty:
                new_df = df[['datetime', 'value']].copy() 
                new_df.rename(columns={'value': key}, inplace=True)
                merged_ts_df = pd.merge(merged_ts_df, new_df, on='datetime', how='outer') if not merged_ts_df.empty else new_df 
        elif key in ['TC', 'pH', 'profile_ch4', 'profile_co2', 'profile_o2']:
            df = input_dict[key]
            if df is not None and not df.empty:
                new_df = df[['horizon', 'value']].copy() 
                new_df.rename(columns={'value': key}, inplace=True)
                merged_static_df = pd.concat([merged_static_df, new_df], ignore_index=True) if not merged_static_df.empty else new_df
        else:
            df = input_dict[key]
            if df is not None and not df.empty:
                df['variable'] = key
                merged_ts_df = pd.concat([merged_ts_df, df], ignore_index=True) 
    
    for key, df in input_dict.items():
        if df is not None and not df.empty:
            df['variable'] = key
            merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df

# --- helper functions for load_observations() -------------------------------
def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:    
    """Drop rows where the value column is NaN."""
    if df is None:
        return None
    return df[df["value"].notna()].reset_index(drop=True)

# --- End of helper functions --------------------------------- ----------------------  

def read_inputs_from_excel(cfg: ModelConfig, site_name: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Load observations from the provided paths with robust path & schema checks.

    Normalises sheet names and column conventions so the downstream loader chain
    works with either the legacy xlsx layout (lowercase sheets, pre-combined
    ``datetime``) or the v2 layout (PascalCase sheets, separate ``DATE`` +
    ``Time_of_day``).
    """
    xls = pd.ExcelFile(cfg.paths['env']['INPUT_DIR'] + f"/{site_name}_inputs.xlsx")
    sheet_names = xls.sheet_names
    input_dict: Dict[str, Optional[pd.DataFrame]] = {}

    # Mapping new-pipeline sheet → legacy sheet name (lower-case form the loader expects)
    _SHEET_RENAME = {
        "FluxCO2":    "flux_co2",
        "FluxCH4":    "flux_ch4",
        "ST":         "ST",
        "SM":         "SM",
        "AT":         "AT",
        "pH":         "pH",
        "TN_TC":      "TC",            # legacy name 'TC' is the TC subset of TN_TC
        "GasProfile": "GasProfile",    # auto-site only
        # 'profile(gas)' is split into profile_co2 / profile_ch4 / profile_o2 below
    }

    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)

        # Combine DATE + Time_of_day → datetime if needed
        if df is not None and not df.empty:
            cols = list(df.columns)
            if "datetime" not in cols and "DATE" in cols and "Time_of_day" in cols:
                dt_str = df["DATE"].astype(str) + " " + df["Time_of_day"].astype(str)
                df = df.drop(columns=["DATE", "Time_of_day"], errors="ignore")
                df["datetime"] = pd.to_datetime(dt_str, errors="coerce")
                df = df.sort_values(by="datetime").reset_index(drop=True)

        # Split profile(gas) into per-species DataFrames (legacy contract)
        if sheet.lower() == "profile(gas)" and df is not None:
            for species_label, legacy_key in [("ConCH4", "profile_ch4"),
                                              ("ConCO2", "profile_co2"),
                                              ("ConO2",  "profile_o2")]:
                if "measurement" in df.columns:
                    sub = df[df["measurement"] == species_label].copy()
                    input_dict[legacy_key] = sub if not sub.empty else None
            continue  # do not also store 'profile(gas)' under its raw name

        # For TN_TC, slice out only the TC rows (matches legacy 'TC' content)
        if sheet == "TN_TC" and df is not None and "measurement" in df.columns:
            tc_sub = df[df["measurement"].astype(str).str.contains("TC", case=False, regex=True)]
            input_dict["TC"] = tc_sub if not tc_sub.empty else None
            continue

        # Standard rename + store
        target_key = _SHEET_RENAME.get(sheet, sheet)
        input_dict[target_key] = df

    return input_dict


# ---------------------- Example usage ----------------------
if __name__ == "__main__":

    from loader_config_by_site import load_config

    yaml_path = './configs/config_new_bysites.yaml'
    site_names_mannual = ['MT'+ f"{i:02d}" for i in range(1, 111)]  # MT001 to MT110
    site_names_auto = ['U.01.C', 'U.03.C', 'U.05.C', 'U.02.t', 'U.04.t', 'U.06.t', 'T.07.C', 'T.09.C', 'T.11.C', 'T.08.t',
                       'T.10.t', 'T.12.t']
    site_names_mannualinautoplots = ['TCM07', 'TCM09', 'TCM11', 
                                     'TTM08', 'TTM10', 'TTM12', 
                                     'UCM01', 'UCM03', 'UCM05',
                                    'UTM02', 'UTM06', 'UTM51']
    # site_names_all = site_names_mannual + site_names_auto + site_names_mannualinautoplots
    site_names_all = site_names_auto

    for site_name in site_names_all:
        cfg = load_config(yaml_path, site_name=site_name)
        # Load observations
        raw_df = load_raw_excel(cfg)
        prepared_dict = preparing_inputs_dict(raw_df)
        # pdb.set_trace()
        # Preprocess inputs
        inputs_dict_units = preprocess_units(prepared_dict)
        # Check inputs
        inputs_dict_time = preprocess_time_steps(cfg, inputs_dict_units)
        inputs_dict_checked = preprocess_depth(inputs_dict_time)
        # save to excel with each key as a sheet
        path_xls = f"./data/inputs/{site_name}_inputs.xlsx"
        with pd.ExcelWriter(path_xls) as writer:
            for key, df in inputs_dict_checked.items():
                if df is not None:
                    df.to_excel(writer, sheet_name=key, index=False)
                    print(f"{key} has {len(df)} rows after preprocessing and checking.")
                else:
                    print(f"{key} is None.")

        # path_output = f"./data/inputs/{site_name}_inputs_dict.pkl"
        # pickle.dump(inputs_dict_checked, open(path_output, "wb"))
