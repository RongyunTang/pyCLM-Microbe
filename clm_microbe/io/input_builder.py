"""Data-prep helpers for the CLM-Microbe pipeline.

This module exposes three functions used to prepare model inputs:

  • compile_profile_data(cfg, inputs)      — build per-year depth-profile arrays
  • compile_time_series_data(cfg, inputs)  — align ST / SM / AT / flux time series
  • prepare_model_inputs(cfg, inputs)      — top-level entry; returns (inputs_ts, inputs_profile)
"""
import numpy as np
import pandas as pd
import os
from clm_microbe.io.config_loader import load_config, ModelConfig
from clm_microbe.io.data_loader import read_inputs_from_excel
from typing import Tuple
from functools import reduce
import re

def compile_profile_data(cfg, inputs):
    """
    Compile and interpolate soil profile data to CLM layers.
    Handles multiple years/campaigns (each with ~2 depth records).
    """
    print('Compiling profile data...')

    # Extract possible profile variables
    profile_vars = {
        'TC': inputs.get('TC', None),
        'pH': inputs.get('pH', None),
        'pressure': inputs.get('pressure', None),
        'PVRT': inputs.get('PVRT', None),
        'CH4': inputs.get('profile_ch4', None),
        'CO2': inputs.get('profile_co2', None),
        'O2': inputs.get('profile_o2', None),
    }

    # Fixed CLM depth levels (positive down)
    clm_depth_cm = np.array(cfg.constants['CLM_layers_depth_cm'])
    depths_for_interp = -clm_depth_cm  # negative = downward
    print("CLM layers (cm):", depths_for_interp)

    def interp_profile(group, depths=depths_for_interp):
        """Interpolate/extrapolate profile values for one datetime group."""
        start_depths = np.asarray(group['start_cm'])
        end_depths = np.asarray(group['end_cm'])
        values = np.asarray(group['value'])

        # Use layer midpoints
        mid_depths = (start_depths + end_depths) / 2

        # Clean NaNs
        mask = ~np.isnan(mid_depths) & ~np.isnan(values)
        mid_depths, values = mid_depths[mask], values[mask]

        if len(mid_depths) == 0:
            return np.full_like(depths, np.nan, dtype=float)

        # Sort to ensure monotonic
        order = np.argsort(mid_depths)
        mid_depths, values = mid_depths[order], values[order]

        # Interpolate within range, extrapolate beyond
        return np.interp(depths, mid_depths, values,
                         left=values[0], right=values[-1])

    # Initialize with depth column
    profiles_out = []
    for name, var in profile_vars.items():
        # Skip None or empty entries
        if var is None or var.empty:
            print(f"Warning: Variable '{name}' is {'None' if var is None else 'empty'} and will be skipped.")
            continue
        # Try to identify datetime column if not exactly named 'datetime'
        datetime_col = None
        for col in var.columns:
            if 'datetime' in col.lower() or 'time' in col.lower() or 'date' in col.lower():
                datetime_col = col
                break
        if datetime_col is None:
            print(f"Warning: No datetime-like column found for variable '{name}'. Will skip.")
            continue
        try:
            # Ensure datetime column is properly formatted
            if not pd.api.types.is_datetime64_any_dtype(var[datetime_col]):
                var[datetime_col] = pd.to_datetime(var[datetime_col], errors='coerce')
                # Remove rows where datetime conversion failed
                var = var.dropna(subset=[datetime_col])
            if var.empty:
                print(f"Warning: No valid datetime values found for variable '{name}' after conversion.")
                continue
            # Group by time and interpolate each profile
            all_records = []
            for t, group in var.groupby(datetime_col):
                if len(group) < 2:
                    continue  # Skip groups with insufficient data
                try:
                    interp_vals = interp_profile(group)
                    df_out = pd.DataFrame({
                        'datetime': t,
                        'depth_cm': depths_for_interp,
                        name: interp_vals
                    })
                    all_records.append(df_out)
                except Exception as e:
                    print(f"Error interpolating {name} at {t}: {str(e)}")
                    continue

            if all_records:
                var_df = pd.concat(all_records, ignore_index=True)
                profiles_out.append(var_df)
                print(f"Successfully processed {name}: {len(all_records)} profiles")
        except Exception as e:
            print(f"Error processing variable '{name}': {str(e)}")
            continue

    # Merge all variables together on [datetime, depth_cm]
    # Handle case where no profile data is available
    if not profiles_out:
        print("Warning: No profile data available after processing all variables.")
        print("Creating empty profile dataframe with standard depths...")
        return None
        
        # Create an empty dataframe with the proper structure
        empty_profile = pd.DataFrame({
            'datetime': pd.Timestamp.now(),  # or use a default date
            'depth_cm': depths_for_interp,
            'dummy_var': np.nan  # dummy variable to maintain structure
        })
        profiles_out.append(empty_profile)

      # Merge all variables into single dataframe
    try:
        if len(profiles_out) == 1:
            inputs_profile = profiles_out[0]
            # If we only have the dummy variable, drop it
            if 'dummy_var' in inputs_profile.columns:
                inputs_profile = inputs_profile.drop('dummy_var', axis=1)
        else:
            inputs_profile = reduce(
                lambda left, right: pd.merge(left, right, on=['datetime', 'depth_cm'], how='outer'),
                profiles_out
            )

        print(f"Final profile data shape: {inputs_profile.shape}")
        
    except Exception as e:
        print(f"Error merging profile data: {str(e)}")
        print("Returning empty profile dataframe...")
        inputs_profile = pd.DataFrame(columns=['datetime', 'depth_cm'])

    # inputs_profile = reduce(
    #     lambda left, right: pd.merge(left, right, on=['datetime', 'depth_cm'], how='outer'),
    #     profiles_out
    # )
    # inputs_profile.sort_values(['datetime', 'depth_cm']).reset_index(drop=True)
    inputs_profile['year'] = pd.to_datetime(inputs_profile['datetime']).dt.year
    profile_by_year = {
        y: df.drop(columns='year').reset_index(drop=True)
        for y, df in inputs_profile.groupby('year')
    }
    
    def fill_missing_profiles(profile_dict):
        years = sorted(profile_dict.keys())
        filled = {}

        for i, y in enumerate(years):
            df = profile_dict[y].copy()

            # For each column (except datetime/depth), fill NaN
            for col in df.columns:
                if col in ['datetime', 'depth_cm']:
                    continue

                if df[col].isna().any():
                    # Search nearest year(s) with data
                    for offset in range(1, len(years)):
                        # look backward
                        if i - offset >= 0:
                            prev_y = years[i - offset]
                            if profile_dict[prev_y][col].notna().any():
                                df[col] = df[col].fillna(profile_dict[prev_y][col])
                                break
                        # look forward
                        if i + offset < len(years):
                            next_y = years[i + offset]
                            if profile_dict[next_y][col].notna().any():
                                df[col] = df[col].fillna(profile_dict[next_y][col])
                                break

            filled[y] = df

        return filled

    profile_dict = fill_missing_profiles(profile_by_year)

    return profile_dict

def compile_time_series_data(cfg, inputs):
    ##### time series data ##### 
    ST_obs = inputs.get('ST', None) # soil temperature (primary sensor)
    SM_obs = inputs.get('SM', None) # soil moisture
    AT_obs = inputs.get('AT', None) # air temperature
    flux_ch4_obs = inputs.get('flux_ch4', None) # surface flux CH4
    flux_co2_obs = inputs.get('flux_co2', None) # surface flux CO2

    # ST2 fallback (disabled — see diagnose_ch4_divergence.py notes):
    # Extending to pre-treatment window introduces non-stationarity at T.08.t
    # (source→sink reversal) and a 7-month gap in the active treatment period.
    # Calibrating over both windows with a static parameter set worsens both periods.
    # ST2_raw = inputs.get('ST2', None)  # kept for future reference

    ##### compile time series data into a dataframe #####
    # Helper to safely extract values
    def safe_values(obs):
        return obs['value'].values if obs is not None else np.full(1, np.nan)
    # remove any rows with NaN or -9999 values
    def clean_obs(obs, obs_name):
        if obs is None or obs.empty:
            return obs
        obs_clean = obs.copy()
        initial_len = len(obs_clean)
        obs_clean = obs_clean[obs_clean['value'].notna() & (obs_clean['value'] != -9999)].reset_index(drop=True)
        final_len = len(obs_clean)
        if final_len < initial_len:
            print(f"Cleaned {initial_len - final_len} invalid records from {obs_name}")
        return obs_clean
    ST_obs = clean_obs(ST_obs, 'ST_obs')
    SM_obs = clean_obs(SM_obs, 'SM_obs')
    AT_obs = clean_obs(AT_obs, 'AT_obs')
    flux_ch4_obs = clean_obs(flux_ch4_obs, 'flux_ch4_obs')
    flux_co2_obs = clean_obs(flux_co2_obs, 'flux_co2_obs')

    # check data lengths 
    lengths = [len(df) for df in [ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs] if df is not None and not df.empty]

    def process_depth_filtering(df, df_name):
        """Safely process depth filtering for a dataset"""
        if df is None or df.empty:
            return df
            
        # Create a copy to avoid SettingWithCopyWarning
        df_processed = df.copy()
        
        # Check for valid depth data
        if 'depth' in df_processed.columns:
            valid_depths = df_processed['depth'].dropna().unique()
            if 'cm' in str(valid_depths).lower():
                print(f"Note: Depth values in {df_name} contain 'cm' units, ensure consistency.")
                # remove 'cm' if present for numeric comparison 
                df_processed['depth'] = df_processed['depth'].apply(lambda x: float(str(x).replace('cm', '').strip()) if pd.notna(x) and 'cm' in str(x).lower() else x) 
                valid_depths = df_processed['depth'].dropna().unique() 

            if len(valid_depths) > 0:
                print(f"{df_name} has depth levels: {valid_depths} and rows: {len(df_processed)}")
                top_depth = min([float(d) for d in valid_depths if pd.notna(d)])  
                df_processed['depth'] = df_processed['depth'].astype(float) # ensure depth is float for comparison
                df_processed = df_processed[df_processed['depth']==top_depth] # filter to top depth

                # Process datetime
                if 'datetime' in df_processed.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df_processed['datetime']):
                        df_processed['datetime'] = pd.to_datetime(df_processed['datetime'], errors='coerce')
                        print(f"  Converted 'datetime' column to datetime dtype")
                    
                    # Sort by datetime (return sorted copy instead of inplace)
                    df_processed = df_processed.sort_values('datetime')
                else:
                    print(f"Warning: No 'datetime' column found in {df_name}. Skipping datetime processing.")

                return df_processed
            else:
                print(f"{df_name}: Depth column exists but all values are NaN")
                return df_processed
        else:
            print(f"{df_name}: No depth column found, using all {len(df_processed)} records")
            return df_processed

    # Process each dataset with depth filtering
    print("Processing datasets for depth filtering...")
    ST_obs = process_depth_filtering(ST_obs, 'ST_obs')
    SM_obs = process_depth_filtering(SM_obs, 'SM_obs')
    AT_obs = process_depth_filtering(AT_obs, 'AT_obs')
    flux_ch4_obs = process_depth_filtering(flux_ch4_obs, 'flux_ch4_obs')
    flux_co2_obs = process_depth_filtering(flux_co2_obs, 'flux_co2_obs')

    # process datetime column( like 01:59:00 -> 02:00:00)
    def process_datetime_column(df, df_name):
        """Safely process datetime column for a dataset"""
        if df is None or df.empty:
            return df
        df_copy = df.copy()
        # round to nearest hour 
        df_copy['datetime'] = pd.to_datetime(df_copy['datetime'], errors='coerce').dt.round('h')
        return df_copy
    ST_obs = process_datetime_column(ST_obs, 'ST_obs')
    SM_obs = process_datetime_column(SM_obs, 'SM_obs')
    AT_obs = process_datetime_column(AT_obs, 'AT_obs')
    flux_ch4_obs = process_datetime_column(flux_ch4_obs, 'flux_ch4_obs')
    flux_co2_obs = process_datetime_column(flux_co2_obs, 'flux_co2_obs')
 
    # Remove any empty dataframes from consideration
    valid_dfs = [(i, df) for i, df in enumerate([ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs]) 
                if df is not None and not df.empty]
    # Check if we have multiple valid dataframes with different lengths
    valid_lengths = [len(df) for _, df in valid_dfs]
    valid_start_dates = [pd.to_datetime(df['datetime'], errors='coerce').min() for _, df in valid_dfs] 
    valid_end_dates = [pd.to_datetime(df['datetime'], errors='coerce').max() for _, df in valid_dfs]
    overlap_start_date = max(valid_start_dates) if valid_start_dates else None
    overlap_end_date = min(valid_end_dates) if valid_end_dates else None
    print(f"Valid dataset lengths after range cut: {valid_lengths}")
    print(f"Overlap start: {overlap_start_date}, Overlap end: {overlap_end_date}")

    for df in [ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs]:
        if df is not None and not df.empty and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if overlap_start_date and overlap_end_date:
                df = df[(df['datetime'] >= overlap_start_date) & (df['datetime'] <= overlap_end_date)]
                # Update the original reference
                if df is ST_obs:
                    ST_obs = df
                elif df is SM_obs:
                    SM_obs = df
                elif df is AT_obs:
                    AT_obs = df
                elif df is flux_ch4_obs:
                    flux_ch4_obs = df
                elif df is flux_co2_obs:
                    flux_co2_obs = df
                # print(f"  {df} records after aligning to overlap date range with records: {len(df)}")
    
    def check_all_same_length(ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs):

        valid_lengths = [df['datetime'].dropna().unique().shape[0] for df in [ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs] if df is not None and not df.empty]
        print(f"Valid dataset lengths before time points cut: {valid_lengths}") 
        if len(set(valid_lengths)) > 1:
            # print(f"Found datasets with different lengths: {valid_lengths}")
            # checking overlapping time points
            # Find the minimum length and corresponding reference dataframe
            min_length = min(valid_lengths)
            reference_idx = valid_lengths.index(min_length) # index of the dataframe with minimum length 
            reference_df = valid_dfs[reference_idx][-1]
            reference_df['datetime'] = pd.to_datetime(reference_df['datetime'], errors='coerce')
            print(f"Using dataset with minimum length {min_length} as reference")

            start_date = reference_df['datetime'].iloc[0]
            end_date = reference_df['datetime'].iloc[-1]
            time_series = reference_df['datetime'].dropna().unique()
            print(f"Reference date range: {start_date} to {end_date} with {len(time_series)} unique timestamps") 
            # print(time_series)
            # Create a dictionary to store updated dataframes
            updated_dfs = {
                'ST_obs': ST_obs,
                'SM_obs': SM_obs, 
                'AT_obs': AT_obs,
                'flux_ch4_obs': flux_ch4_obs,
                'flux_co2_obs': flux_co2_obs
            }
            # Subset each dataframe to match reference datetime values
            for df_name in ['ST_obs', 'SM_obs', 'AT_obs', 'flux_ch4_obs', 'flux_co2_obs']:
                df = updated_dfs[df_name]
                if df is not None and not df.empty:
                    print(f"Subsetting {df_name} from {len(df)} to match reference datetime range")
                    
                    # Ensure datetime is properly formatted
                    df = df.copy()
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    
                    # Filter to only include rows with datetime values present in reference
                    # mask = df['datetime'].isin(time_series)
                    # subset_df = df[mask].copy()
                    subset_df = df[df['datetime'].isin(time_series)].copy() 
                    print(f"  {df_name}: {len(df)} → {len(subset_df)} records after subsetting")
                    updated_dfs[df_name] = subset_df
                    time_series = subset_df['datetime'].dropna().unique() if len(subset_df) < len(time_series) else time_series 
                    print(f"  Updated time series length: {len(time_series)}") 

            # Update the original variables
            ST_obs = updated_dfs['ST_obs'].reset_index(drop=True).copy()
            SM_obs = updated_dfs['SM_obs'].reset_index(drop=True).copy()
            AT_obs = updated_dfs['AT_obs'].reset_index(drop=True).copy()
            flux_ch4_obs = updated_dfs['flux_ch4_obs'].reset_index(drop=True).copy()
            flux_co2_obs = updated_dfs['flux_co2_obs'].reset_index(drop=True).copy()
            # remove duplicates if any 
            ST_obs = ST_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            SM_obs = SM_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            AT_obs = AT_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            flux_ch4_obs = flux_ch4_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            flux_co2_obs = flux_co2_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            # remove invalid datetime rows if any
            ST_obs = ST_obs.dropna(subset=['datetime']).reset_index(drop=True)
            SM_obs = SM_obs.dropna(subset=['datetime']).reset_index(drop=True)
            AT_obs = AT_obs.dropna(subset=['datetime']).reset_index(drop=True)
            flux_ch4_obs = flux_ch4_obs.dropna(subset=['datetime']).reset_index (drop=True)
            flux_co2_obs = flux_co2_obs.dropna(subset=['datetime']).reset_index(drop=True)  
            
            # Final length check
            return ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs
        else:
            print("All datasets have the same length, no subsetting needed.")
            # remove duplicates if any
            ST_obs = ST_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            SM_obs = SM_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            AT_obs = AT_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            flux_ch4_obs = flux_ch4_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            flux_co2_obs = flux_co2_obs.drop_duplicates(subset='datetime').reset_index(drop=True)
            # remove invalid datetime rows if any 
            ST_obs = ST_obs.dropna(subset=['datetime']).reset_index(drop=True)
            SM_obs = SM_obs.dropna(subset=['datetime']).reset_index(drop=True)
            AT_obs = AT_obs.dropna(subset=['datetime']).reset_index(drop=True)
            flux_ch4_obs = flux_ch4_obs.dropna(subset=['datetime']).reset_index(drop=True)
            flux_co2_obs = flux_co2_obs.dropna(subset=['datetime']).reset_index(drop=True)
    
            return ST_obs.copy(), SM_obs.copy(), AT_obs.copy(), flux_ch4_obs.copy(), flux_co2_obs.copy()

    ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs = check_all_same_length(ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs)
    ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs = check_all_same_length(ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs) # run twice to ensure all aligned
    ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs = check_all_same_length(ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs) # run thrice to ensure all aligned
    

    # Final validation check
    final_lengths = [df['datetime'].dropna().unique().shape[0] for df in [ST_obs, SM_obs, AT_obs, flux_ch4_obs, flux_co2_obs] if df is not None and not df.empty]
    if len(set(final_lengths)) > 1:
        print(f"Warning: Datasets still have different lengths after processing: {final_lengths}")
    else:
        print(f"All datasets aligned with length: {final_lengths[0]}")
    # pdb.set_trace()
    inputs_ts = pd.DataFrame({
        'datetime': pd.to_datetime(flux_ch4_obs['datetime']),
        'ST': ST_obs['value'].values,
        'SM': SM_obs['value'].values,
        'AT': AT_obs['value'].values,
        'flux_ch4': flux_ch4_obs['value'].values,
        'flux_co2': flux_co2_obs['value'].values,
    })

    inputs_ts['year'] = inputs_ts['datetime'].dt.year # prepare dict by year in dict
    ts_dict_by_year = {y: df.drop(columns='year').reset_index(drop=True) for y, df in inputs_ts.groupby('year')}
    return ts_dict_by_year 

def prepare_model_inputs(cfg, inputs: dict):
    # extract observations from inputs
    print("compiling inputs or model...")
    ts_dict = compile_time_series_data(cfg, inputs)
    print('finished time series data compilation.')

    ##### profile data #####
    print('compiling profile data...')
    profile_dict = compile_profile_data(cfg, inputs)
    print('finished profile data compilation.')
    print('profile dict by year:')

    ##### check years ##### 
    # if profile years less than time series years, fill with nearest year profile data 
    if profile_dict is not None:
        print(f"Profile years available: {sorted(profile_dict.keys())}")
        profile_years = sorted(profile_dict.keys())
        ts_years = sorted(ts_dict.keys())
        print(f"Time series years: {ts_years}")
        print(f"Profile years: {profile_years}")
        
        for y in ts_years:
            if y not in profile_years:
                # find nearest year
                nearest_year = min(profile_years, key=lambda x: abs(x - y))
                nearest_year_dict = profile_dict[nearest_year].copy()    
                # change the datetime to match the target year 
                nearest_year_dict['datetime'] = pd.to_datetime(nearest_year_dict['datetime']).apply(
                    lambda dt: dt.replace(year=y).strftime('%Y-%m-%d')
                )
                # assign to the missing year
                profile_dict[y] = nearest_year_dict
                print(f"Filling missing profile for year {y} with data from year {nearest_year}.")
    
    else: 
        print("No profile data available.")
        ts_years = sorted(ts_dict.keys()) 
        profile_dict = {}
        for y in ts_years:
            print(f"Creating empty profile for year {y}.")
            # Create an empty dataframe with the proper structure
            empty_profile = pd.DataFrame({
                'datetime': np.repeat(ts_dict[y]['datetime'], len(cfg.constants['CLM_layers_depth_cm'])),
                'depth_cm': np.repeat(np.array(cfg.constants['CLM_layers_depth_cm']), len(ts_dict[y])),
                'TC': np.nan,
                'pH': np.nan,
                'pressure': np.nan,
                'PVRT': np.nan,
                'CH4': np.nan,
                'CO2': np.nan,
                'O2': np.nan
            })
            profile_dict[y] = empty_profile


    ##### concatenate all years data #####
    inputs_profile = pd.concat(profile_dict.values(), ignore_index=True)
    inputs_profile = inputs_profile.sort_values(['datetime', 'depth_cm'], ascending=[True, False]).reset_index(drop=True)
    inputs_ts = pd.concat(ts_dict.values(), ignore_index=True)
    inputs_ts = inputs_ts.sort_values('datetime').reset_index(drop=True) 
    inputs_ts['timestep'] = inputs_ts['datetime'].diff().dt.total_seconds().fillna(0).astype(int)
    inputs_ts.loc[0, 'timestep'] = 15 * 60 * 60  # assume first timestep is 15 days
    inputs_ts['year'] = pd.to_datetime(inputs_ts['datetime']).dt.year
    inputs_ts['drainage'] = inputs['ST']['drainage'][0]
    inputs_profile['year'] = pd.to_datetime(inputs_profile['datetime']).dt.year
      
    # print('final profile data:')
    # print(inputs_profile)
    # print('final time series data:')
    # print(inputs_ts)
    inputs_tuple = (inputs_ts, inputs_profile)
    return inputs_tuple, cfg
