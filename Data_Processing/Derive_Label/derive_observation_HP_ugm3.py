#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import xarray as xr
import warnings
import gc
from collections import defaultdict
warnings.filterwarnings('ignore')

# Configuration
EARTH_RADIUS_KM = 6371.0

# File paths
# no2_monthly_observations_china_ugm3
obs_file = '/my-projects2/1.project/NO2_ground_complied/global/no2_monthly_observations.csv'
lat_grid_file = '/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global_MAP.npy'
lon_grid_file = '/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global_MAP.npy'
gchp_dir = '/my-projects2/1.project/GeoNO2/'
output_dir = '/my-projects2/1.project/NO2_DL_global/TrainingDatasets/Global_NO2_BenchMark/'

os.makedirs(output_dir, exist_ok=True)

def get_nearest_point_index(sitelon, sitelat, lon_grid_shape, lat_grid_shape, grid_min_lon, grid_min_lat):
    """
    Optimized version that doesn't need full grid arrays in memory
    """
    det = 0.01
    index_lon = np.round((sitelon - grid_min_lon) / det).astype(np.int32)
    index_lat = np.round((sitelat - grid_min_lat) / det).astype(np.int32)
    
    # Clip to bounds
    index_lat = np.clip(index_lat, 0, lat_grid_shape[0] - 1)
    index_lon = np.clip(index_lon, 0, lon_grid_shape[1] - 1)
    
    return index_lon, index_lat

def apply_corrections_vectorized(base, obs, alkylnitrates, hno3, pan, alpha=0.15, beta=0.95):
    """Apply correction factors to observations."""
    denom = base + alkylnitrates + alpha * hno3 + beta * pan
    denom_safe = np.where(denom == 0, 1, denom)
    cf = obs * (base / denom_safe)
    return cf

def load_observations_optimized(years):
    """Load and preprocess observations with memory optimization."""
    print("Loading observations...")
    
    # Load only required columns to save memory
    required_cols = ['lat', 'lon', 'year', 'mon', 'no2_ppb']
    obs_df = pd.read_csv(obs_file, usecols=required_cols, low_memory=False)
    
    # Filter years immediately
    obs_df = obs_df[obs_df['year'].isin(years)]
    
    # Remove invalid observations
    valid_mask = ~(pd.isna(obs_df['lat']) | pd.isna(obs_df['lon']) | pd.isna(obs_df['no2_ppb']))
    obs_clean = obs_df[valid_mask].copy()
    del obs_df  # Free memory
    gc.collect()
    
    print(f"Valid observations: {len(obs_clean):,}")
    
    # Use more memory-efficient grouping
    site_coords = obs_clean[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    site_coords['site_id'] = range(len(site_coords))
    
    # Merge back efficiently
    obs_clean = obs_clean.merge(site_coords, on=['lat', 'lon'])
    
    print(f"Unique observation sites: {len(site_coords)}")
    print(f"Total observations: {len(obs_clean):,}")
    
    return obs_clean, site_coords

def get_gchp_coordinates_only(years):
    """Get GCHP grid coordinates without loading data."""
    print("Getting GCHP grid coordinates...")
    
    for year in years:
        year_dir = os.path.join(gchp_dir, str(year))
        for month in range(1, 13):
            month_str = f"{month:02d}"
            gchp_file = os.path.join(year_dir, f'1x1km.GeoNO2.{year}{month_str}.MonMean.nc')
            
            if os.path.exists(gchp_file):
                try:
                    # Open with minimal memory usage
                    with xr.open_dataset(gchp_file, engine='netcdf4') as ds:
                        lat_1d = ds.lat.values.astype(np.float32)
                        lon_1d = ds.lon.values.astype(np.float32)
                        
                        # Just get coordinate info, not full grid
                        grid_shape = (len(lat_1d), len(lon_1d))
                        min_lat, max_lat = float(lat_1d.min()), float(lat_1d.max())
                        min_lon, max_lon = float(lon_1d.min()), float(lon_1d.max())
                        
                        return {
                            'lat_1d': lat_1d,
                            'lon_1d': lon_1d,
                            'grid_shape': grid_shape,
                            'lat_range': (min_lat, max_lat),
                            'lon_range': (min_lon, max_lon),
                            'min_lat': min_lat,
                            'min_lon': min_lon
                        }
                        
                except Exception as e:
                    print(f"Error reading coordinates from {gchp_file}: {e}")
                    continue
    
    raise ValueError("No valid GCHP files found!")

def match_sites_to_gchp_optimized(site_coords, gchp_coords):
    """Match sites to GCHP grid with minimal memory usage."""
    print("Matching observation sites to GCHP grid...")
    
    sitelat = site_coords['lat'].values
    sitelon = site_coords['lon'].values
    
    # Use optimized index calculation
    index_lon, index_lat = get_nearest_point_index(
        sitelon=sitelon, 
        sitelat=sitelat,
        lon_grid_shape=gchp_coords['grid_shape'], 
        lat_grid_shape=gchp_coords['grid_shape'],
        grid_min_lon=gchp_coords['min_lon'],
        grid_min_lat=gchp_coords['min_lat']
    )
    
    # Calculate GCHP coordinates for matched sites
    gchp_lat_matched = gchp_coords['lat_1d'][index_lat]
    gchp_lon_matched = gchp_coords['lon_1d'][index_lon]
    
    # Add matching info
    site_coords_matched = site_coords.copy()
    site_coords_matched['gchp_lat_index'] = index_lat.astype(np.int32)
    site_coords_matched['gchp_lon_index'] = index_lon.astype(np.int32)
    site_coords_matched['gchp_lat'] = gchp_lat_matched
    site_coords_matched['gchp_lon'] = gchp_lon_matched
    
    print(f"Grid index ranges:")
    print(f"  Lat indices: {index_lat.min()} to {index_lat.max()}")
    print(f"  Lon indices: {index_lon.min()} to {index_lon.max()}")
    
    # Group by grid indices efficiently
    print("Grouping sites by GCHP grid cell...")
    
    grouped = site_coords_matched.groupby(['gchp_lat_index', 'gchp_lon_index']).agg({
        'site_id': lambda x: list(x),
        'lat': 'mean',
        'lon': 'mean',
        'gchp_lat': 'first',
        'gchp_lon': 'first'
    }).reset_index()
    
    grouped['grouped_site_id'] = range(len(grouped))
    grouped['num_original_sites'] = grouped['site_id'].apply(len)
    
    print(f"Original sites: {len(site_coords_matched)}")
    print(f"Grouped sites: {len(grouped)}")
    
    return grouped, site_coords_matched

def load_single_gchp_file(filepath):
    """Load single GCHP file with minimal memory footprint."""
    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            return {
                'gchp_NO2': ds['gchp_NO2'].values.astype(np.float32),
                'geophysical_NO2': ds['filled_GeoNO2'].values.astype(np.float32),
                'gchp_alkylnitrates': ds['gchp_alkylnitrates'].values.astype(np.float32),
                'gchp_HNO3': ds['gchp_HNO3'].values.astype(np.float32),
                'gchp_PAN': ds['gchp_PAN'].values.astype(np.float32)
            }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_time_chunk(obs_df, grouped_sites, years, start_time_idx, chunk_size, 
                      obs_arrays, geo_arrays, time_coords):
    """Process a chunk of time steps to reduce memory usage."""
    
    end_time_idx = min(start_time_idx + chunk_size, len(time_coords))
    chunk_times = time_coords[start_time_idx:end_time_idx]
    
    print(f"Processing time chunk {start_time_idx}-{end_time_idx-1} ({len(chunk_times)} time steps)")
    
    for local_idx, timestamp in enumerate(chunk_times):
        global_time_idx = start_time_idx + local_idx
        year = timestamp.year
        month = timestamp.month
        
        # Load GCHP data for this month only
        gchp_file = os.path.join(gchp_dir, str(year), f'1x1km.GeoNO2.{year}{month:02d}.MonMean.nc')
        
        if not os.path.exists(gchp_file):
            continue
            
        gchp_data = load_single_gchp_file(gchp_file)
        if gchp_data is None:
            continue
        
        # Process each grouped site
        for group_idx, (_, group_row) in enumerate(grouped_sites.iterrows()):
            gchp_lat_idx = int(group_row['gchp_lat_index'])
            gchp_lon_idx = int(group_row['gchp_lon_index'])
            
            # Extract geophysical data
            geo_arrays[group_idx, global_time_idx] = gchp_data['geophysical_NO2'][gchp_lat_idx, gchp_lon_idx]
            
            # Process observations for this group and time
            original_site_ids = group_row['site_id']
            group_obs_values = []
            
            for site_id in original_site_ids:
                site_obs = obs_df[(obs_df['site_id'] == site_id) & 
                                (obs_df['year'] == year) & 
                                (obs_df['mon'] == month)]
                if len(site_obs) > 0:
                    group_obs_values.extend(site_obs['no2_ppb'].values)
            
            if len(group_obs_values) > 0:
                obs_value = np.mean(group_obs_values)
                
                # Apply correction
                obs_corrected_value = apply_corrections_vectorized(
                    gchp_data['gchp_NO2'][gchp_lat_idx, gchp_lon_idx],
                    obs_value,
                    gchp_data['gchp_alkylnitrates'][gchp_lat_idx, gchp_lon_idx],
                    gchp_data['gchp_HNO3'][gchp_lat_idx, gchp_lon_idx],
                    gchp_data['gchp_PAN'][gchp_lat_idx, gchp_lon_idx]
                )
                
                obs_arrays[group_idx, global_time_idx] = obs_corrected_value
        
        # Clean up GCHP data immediately
        del gchp_data
        gc.collect()

def create_datasets_incremental(obs_df, grouped_sites, years):
    """Create datasets using incremental processing to minimize memory usage."""
    print("Creating datasets with incremental processing...")
    
    # Create time coordinates
    time_coords = []
    for year in years:
        for month in range(1, 13):
            time_coords.append(pd.Timestamp(year, month, 1))
    
    n_groups = len(grouped_sites)
    n_times = len(time_coords)
    
    print(f"Initializing arrays: {n_groups} sites x {n_times} times")
    
    # Initialize output arrays
    obs_corrected = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    geo_data = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    
    # Process in chunks to reduce memory usage
    chunk_size = 12  # Process one year at a time
    
    for start_idx in range(0, n_times, chunk_size):
        process_time_chunk(obs_df, grouped_sites, years, start_idx, chunk_size,
                          obs_corrected, geo_data, time_coords)
        
        # Force garbage collection after each chunk
        gc.collect()
    
    print("Creating xarray datasets...")
    
    # Create observation dataset
    ds_obs = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                obs_corrected,
                {
                    'units': 'ppb',
                    'long_name': 'Corrected NO2 observations (grouped by grid cell)',
                    '_FillValue': np.nan}),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords={
            'sites': ('sites', grouped_sites['grouped_site_id'].values),
            'time': ('time', time_coords),
        },
        attrs={
            'title': 'Monthly NO2 Observations (Grouped by GCHP Grid Cell)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Create geophysical dataset
    ds_geo = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                geo_data,
                {
                    'units': 'ppb',
                    'long_name': 'Geophysical NO2 (grouped by grid cell)',
                    '_FillValue': np.nan
                }
            ),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords=ds_obs.coords,
        attrs={
            'title': 'Monthly Geophysical NO2 (Grouped by GCHP Grid Cell)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Create bias dataset
    bias_data = obs_corrected - geo_data
    ds_bias = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                bias_data,
                {
                    'units': 'ppb',
                    'long_name': 'NO2 bias (grouped corrected observations - geophysical)',
                    '_FillValue': np.nan
                }
            ),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords=ds_obs.coords,
        attrs={
            'title': 'Monthly NO2 Bias (Grouped Observations - Geophysical)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    return ds_obs, ds_geo, ds_bias

def save_datasets_efficiently(ds_obs, ds_geo, ds_bias):
    """Save datasets with compression and chunking for efficiency."""
    print("Saving datasets with optimized compression...")
    
    ppb_to_ugm3_factor = 46.0 / 24.45
    
    # Add NO2_ug/m3 variable to observation dataset
    print("Adding NO2_ug/m3 conversion to observation dataset...")
    ds_obs_with_conversion = ds_obs.copy()
    ds_obs_with_conversion['NO2_ugm3'] = (
        ('sites', 'time'),
        ds_obs['NO2'].values * ppb_to_ugm3_factor,
        {
            'units': 'μg/m³',
            'long_name': 'Corrected NO2 observations in μg/m³ (grouped by grid cell)',
            'conversion_note': f'Converted from ppb using factor {ppb_to_ugm3_factor:.6f}',
            '_FillValue': np.nan
        }
    )
    
    # Add NO2_ug/m3 variable to geophysical dataset
    print("Adding NO2_ug/m3 conversion to geophysical dataset...")
    ds_geo_with_conversion = ds_geo.copy()
    ds_geo_with_conversion['NO2_ugm3'] = (
        ('sites', 'time'),
        ds_geo['NO2'].values * ppb_to_ugm3_factor,
        {
            'units': 'μg/m³',
            'long_name': 'Geophysical NO2 in μg/m³ (grouped by grid cell)',
            'conversion_note': f'Converted from ppb using factor {ppb_to_ugm3_factor:.6f}',
            '_FillValue': np.nan
        }
    )
    
    # Add NO2_ug/m3 variable to bias dataset
    print("Adding NO2_ug/m3 conversion to bias dataset...")
    ds_bias_with_conversion = ds_bias.copy()
    ds_bias_with_conversion['NO2_ugm3'] = (
        ('sites', 'time'),
        ds_bias['NO2'].values * ppb_to_ugm3_factor,
        {
            'units': 'μg/m³',
            'long_name': 'NO2 bias in μg/m³ (grouped corrected observations - geophysical)',
            'conversion_note': f'Converted from ppb using factor {ppb_to_ugm3_factor:.6f}',
            '_FillValue': np.nan
        }
    )
    
    # Optimized encoding for better compression and faster I/O
    encoding = {
        'NO2': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_obs_with_conversion.dims['sites']), min(12, ds_obs_with_conversion.dims['time']))
        },
        'NO2_ugm3': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_obs_with_conversion.dims['sites']), min(12, ds_obs_with_conversion.dims['time']))
        }
    }
    
    encoding_geo = {
        'NO2': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_geo_with_conversion.dims['sites']), min(12, ds_geo_with_conversion.dims['time']))
        },
        'NO2_ugm3': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_geo_with_conversion.dims['sites']), min(12, ds_geo_with_conversion.dims['time']))
        }
    }
    
    encoding_bias = {
        'NO2': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_bias_with_conversion.dims['sites']), min(12, ds_bias_with_conversion.dims['time']))
        },
        'NO2_ugm3': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_bias_with_conversion.dims['sites']), min(12, ds_bias_with_conversion.dims['time']))
        }
    }

    
    # Save files
    obs_file_out = os.path.join(output_dir, 'NO2_observation_monthly_grouped_ugm3.nc')
    ds_obs_with_conversion.to_netcdf(obs_file_out, encoding=encoding)
    print(f"Saved: {obs_file_out}")
    
    geo_file_out = os.path.join(output_dir, 'NO2_geophysical_monthly_grouped_ugm3.nc')
    ds_geo_with_conversion.to_netcdf(geo_file_out, encoding=encoding_geo)
    print(f"Saved: {geo_file_out}")
    
    bias_file_out = os.path.join(output_dir, 'NO2_bias_monthly_grouped_ugm3.nc')
    ds_bias_with_conversion.to_netcdf(bias_file_out, encoding=encoding_bias)
    print(f"Saved: {bias_file_out}")

def print_memory_usage():
    """Print current memory usage."""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")

def main():
    """Optimized main execution function."""
    print("Starting optimized NO2 analysis...")
    print("="*60)
    
    print_memory_usage()
    
    # Step 1: Load observations (optimized)
    years = [2019, 2020, 2021, 2022, 2023]
    obs_df, site_coords = load_observations_optimized(years)
    print_memory_usage()
    
    # Step 2: Get GCHP coordinates only (not full data)
    gchp_coords = get_gchp_coordinates_only(years)
    print_memory_usage()
    
    # Step 3: Match sites to GCHP grid (optimized)
    grouped_sites, _ = match_sites_to_gchp_optimized(site_coords, gchp_coords)
    del site_coords  # Free memory
    gc.collect()
    print_memory_usage()
    
    # Step 4: Create datasets with incremental processing
    ds_obs, ds_geo, ds_bias = create_datasets_incremental(obs_df, grouped_sites, years)
    del obs_df  # Free memory
    gc.collect()
    print_memory_usage()
    
    # Step 5: Save with optimized compression
    save_datasets_efficiently(ds_obs, ds_geo, ds_bias)
    
    # Print summary statistics
    obs_valid = ~np.isnan(ds_obs['NO2'].values)
    print(f"\nSummary:")
    print(f"Grouped sites: {len(grouped_sites)}")
    print(f"Valid observations: {np.sum(obs_valid):,}")
    print(f"Data coverage: {100*np.mean(obs_valid):.1f}%")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    
    print_memory_usage()

if __name__ == "__main__":
    main()