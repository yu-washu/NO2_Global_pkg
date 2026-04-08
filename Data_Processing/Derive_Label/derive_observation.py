#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

# Configuration
EARTH_RADIUS_KM = 6371.0

# File paths
obs_file = '/my-projects2/1.project/NO2_ground_complied/global/combined_global_no2_2005-2023_v5.csv'
lat_grid_file = '/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global_MAP.npy'
lon_grid_file = '/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global_MAP.npy'
gchp_dir = '/my-projects2/1.project/GeoNO2/'
output_dir = '/my-projects2/1.project/NO2_DL_global/TrainingDatasets/Global_NO2_v5/'

os.makedirs(output_dir, exist_ok=True)

def get_nearest_point_index(sitelon, sitelat, lon_grid, lat_grid):
    """
    Get the index of stations on the grids map using your original logic
    inputs:
        sitelon, sitelat: stations location arrays
        lon_grid, lat_grid: grids longitude and latitude (2D arrays)
    return:
        index_lon, index_lat: grid indices for each site
    """
    # step1: get the spatial resolution; Default: the latitude and longitude have the same resolution
    det = 0.01
    # step2:
    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)
    index_lon = np.round((sitelon - lon_min) / det)
    index_lat = np.round((sitelat - lat_min) / det)
    index_lon = index_lon.astype(int)
    index_lat = index_lat.astype(int)
    
    # Ensure indices are within bounds
    lat_max_idx = lat_grid.shape[0] - 1
    lon_max_idx = lat_grid.shape[1] - 1
    
    index_lat = np.clip(index_lat, 0, lat_max_idx)
    index_lon = np.clip(index_lon, 0, lon_max_idx)
    
    return index_lon, index_lat

def apply_corrections_vectorized(base, obs, alkylnitrates, hno3, pan, alpha=0.15, beta=0.95):
    """Apply correction factors to observations."""
    denom = base + alkylnitrates + alpha * hno3 + beta * pan
    
    # Apply eps only when denominator is zero
    denom_safe = np.where(denom == 0, 1, denom)
    cf = obs * (base / denom_safe)
    return cf

def load_observations(years):
    """Load observations without gridding."""
    print("Loading observations...")
    
    # Load observations
    obs_df = pd.read_csv(obs_file, low_memory=False)
    obs_df = obs_df[obs_df['year'].isin(years)]  # Fixed
    
    # Remove invalid observations
    valid_obs = ~(np.isnan(obs_df['lat']) | np.isnan(obs_df['lon']) | np.isnan(obs_df['no2']))
    obs_clean = obs_df[valid_obs].copy()
    print(f"Valid observations: {len(obs_clean)}")
    
    # Create unique site information
    site_info = obs_clean.groupby(['lat', 'lon']).size().reset_index(name='count')
    site_info['site_id'] = range(len(site_info))
    
    # Merge site IDs back to observations
    obs_clean = obs_clean.merge(site_info[['lat', 'lon', 'site_id']], on=['lat', 'lon'])
    
    print(f"Observation coordinate ranges:")
    print(f"  Lat: {obs_clean['lat'].min():.3f} to {obs_clean['lat'].max():.3f}")
    print(f"  Lon: {obs_clean['lon'].min():.3f} to {obs_clean['lon'].max():.3f}")
    print(f"Unique observation sites: {len(site_info)}")
    print(f"Total observations: {len(obs_clean)}")
    
    return obs_clean, site_info

def load_gchp_data_for_years(years):
    """Load GCHP data for all years and months."""
    print("Loading GCHP data...")
    
    gchp_data = {}
    gchp_coords = None
    
    for year in years:
        year_dir = os.path.join(gchp_dir, str(year))
        gchp_data[year] = {}
        
        for month in range(1, 13):
            month_str = f"{month:02d}"
            gchp_file = os.path.join(year_dir, f'1x1km.GeoNO2.{year}{month_str}.MonMean.nc')
            
            if os.path.exists(gchp_file):
                try:
                    ds = xr.open_dataset(gchp_file, engine='netcdf4').squeeze()
                    
                    # Store coordinates from first file
                    if gchp_coords is None:
                        gchp_lat_1d = np.array(ds.lat, dtype=np.float32)
                        gchp_lon_1d = np.array(ds.lon, dtype=np.float32)
                        
                        # Create 2D coordinate grids
                        gchp_lon_2d, gchp_lat_2d = np.meshgrid(gchp_lon_1d, gchp_lat_1d, indexing='xy')
                        
                        gchp_coords = {
                            'lat_1d': gchp_lat_1d,
                            'lon_1d': gchp_lon_1d,
                            'lat_2d': gchp_lat_2d,
                            'lon_2d': gchp_lon_2d
                        }
                        print(f"GCHP grid shape: {gchp_lat_2d.shape}")
                        print(f"GCHP lat range: {gchp_lat_2d.min():.3f} to {gchp_lat_2d.max():.3f}")
                        print(f"GCHP lon range: {gchp_lon_2d.min():.3f} to {gchp_lon_2d.max():.3f}")
                    
                    # Store monthly data (keep as 2D arrays)
                    gchp_data[year][month] = {
                        'gchp_NO2': np.array(ds['gchp_NO2']),
                        'geophysical_NO2': np.array(ds['filled_GeoNO2']),
                        'gchp_alkylnitrates': np.array(ds['gchp_alkylnitrates']),
                        'gchp_HNO3': np.array(ds['gchp_HNO3']),
                        'gchp_PAN': np.array(ds['gchp_PAN'])
                    }
                    
                except Exception as e:
                    print(f"Error loading {gchp_file}: {e}")
                    continue
            else:
                print(f"GCHP file not found: {gchp_file}")
    
    return gchp_data, gchp_coords

def match_sites_to_gchp_and_group(site_info, gchp_coords):
    """Match observation sites to GCHP grid and group sites in same grid cell."""
    print("Matching observation sites to GCHP grid and grouping by grid cell...")
    
    # Get GCHP grid coordinates
    gchp_lat_2d = gchp_coords['lat_2d']
    gchp_lon_2d = gchp_coords['lon_2d']
    
    # Use observation coordinates directly (no gridding)
    sitelat = site_info['lat'].values
    sitelon = site_info['lon'].values
    
    print(f"GCHP grid coordinate ranges:")
    print(f"  Lat: {gchp_lat_2d.min():.3f} to {gchp_lat_2d.max():.3f}")
    print(f"  Lon: {gchp_lon_2d.min():.3f} to {gchp_lon_2d.max():.3f}")
    
    # Use your original logic to find nearest indices
    index_lon, index_lat = get_nearest_point_index(
        sitelon=sitelon, 
        sitelat=sitelat,
        lon_grid=gchp_lon_2d, 
        lat_grid=gchp_lat_2d
    )
    
    print(f"Grid index ranges:")
    print(f"  Lat indices: {index_lat.min()} to {index_lat.max()} (max allowed: {gchp_lat_2d.shape[0]-1})")
    print(f"  Lon indices: {index_lon.min()} to {index_lon.max()} (max allowed: {gchp_lat_2d.shape[1]-1})")
    
    # Add GCHP matching info to site_info
    site_info_matched = site_info.copy()
    site_info_matched['gchp_lat_index'] = index_lat.astype(int)
    site_info_matched['gchp_lon_index'] = index_lon.astype(int)
    site_info_matched['gchp_lat'] = gchp_lat_2d[index_lat, index_lon]
    site_info_matched['gchp_lon'] = gchp_lon_2d[index_lat, index_lon]
    
    # Calculate distances for verification
    lat_diff = site_info_matched['lat'] - site_info_matched['gchp_lat']
    lon_diff = site_info_matched['lon'] - site_info_matched['gchp_lon']
    distances_deg = np.sqrt(lat_diff**2 + lon_diff**2)
    site_info_matched['gchp_distance_deg'] = distances_deg
    
    # NEW: Group sites by GCHP grid indices
    print("\nGrouping sites by GCHP grid cell...")
    
    # Group sites that fall in the same GCHP grid cell
    grouped_sites = site_info_matched.groupby(['gchp_lat_index', 'gchp_lon_index']).agg({
        'site_id': lambda x: list(x),  # Keep list of original site IDs
        'lat': 'mean',  # Average the observation coordinates
        'lon': 'mean',
        'count': 'sum',  # Sum observation counts
        'gchp_lat': 'first',  # GCHP coordinates are the same for all sites in group
        'gchp_lon': 'first',
        'gchp_distance_deg': 'mean'  # Average distance
    }).reset_index()
    
    # Create new grouped site IDs
    grouped_sites['grouped_site_id'] = range(len(grouped_sites))
    grouped_sites['num_original_sites'] = grouped_sites['site_id'].apply(len)
    
    print(f"Original sites: {len(site_info_matched)}")
    print(f"Grouped sites: {len(grouped_sites)}")
    print(f"Sites saved by grouping: {len(site_info_matched) - len(grouped_sites)}")
    
    # Show grouping statistics
    multi_site_groups = grouped_sites[grouped_sites['num_original_sites'] > 1]
    if len(multi_site_groups) > 0:
        print(f"\nGrid cells with multiple sites: {len(multi_site_groups)}")
        print(f"Max sites in one grid cell: {grouped_sites['num_original_sites'].max()}")
        print(f"Total sites in multi-site grid cells: {multi_site_groups['num_original_sites'].sum()}")
        
        # Show examples
        print(f"\nExamples of grouped sites:")
        for i, row in multi_site_groups.head(3).iterrows():
            print(f"  Grid [{row['gchp_lat_index']}, {row['gchp_lon_index']}]: " +
                  f"{row['num_original_sites']} sites → 1 grouped site")
            print(f"    Original site IDs: {row['site_id']}")
            print(f"    Averaged coordinates: ({row['lat']:.3f}, {row['lon']:.3f})")
    
    return grouped_sites, site_info_matched

def create_observation_netcdf_grouped(obs_df, grouped_sites, site_info_matched, gchp_data):
    """Create observation NetCDF with sites grouped by grid cell and averaged."""
    print("Creating grouped observation NetCDF...")
    
    # Create time dimension
    filtered_df = obs_df[(obs_df['year'] >= 2005) & (obs_df['year'] <= 2023)]
    years = sorted(filtered_df['year'].unique())
    time_coords = []
    for year in years:
        for month in range(1, 13):
            time_coords.append(pd.Timestamp(year, month, 1))
    
    n_groups = len(grouped_sites)
    n_times = len(time_coords)
    
    # Initialize arrays with NaN
    obs_corrected = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    obs_raw = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    
    print(f"Creating arrays: {n_groups} grouped sites x {n_times} time steps")
    
    # Track data availability
    groups_with_data = set()
    total_filled_points = 0
    
    # Process each grouped site
    for group_idx, (_, group_row) in enumerate(grouped_sites.iterrows()):
        original_site_ids = group_row['site_id']  # List of original site IDs in this group
        gchp_lat_idx = int(group_row['gchp_lat_index'])
        gchp_lon_idx = int(group_row['gchp_lon_index'])
        
        group_has_data = False
        
        for time_idx, timestamp in enumerate(time_coords):
            year = timestamp.year
            month = timestamp.month
            
            # Collect observations from all sites in this group for this time
            group_obs_values = []
            for site_id in original_site_ids:
                site_obs = filtered_df[(filtered_df['site_id'] == site_id) & 
                                (filtered_df['year'] == year) & 
                                (filtered_df['mon'] == month)]
                if len(site_obs) > 0:
                    group_obs_values.extend(site_obs['no2'].values)
            
            # If we have observations for this group and time, average them
            if len(group_obs_values) > 0 and year in gchp_data and month in gchp_data[year]:
                # Average all observations in this group for this time
                obs_value = np.mean(group_obs_values)
                obs_raw[group_idx, time_idx] = obs_value
                
                # Apply correction using GCHP data at the grid indices
                gchp_month_data = gchp_data[year][month]
                obs_corrected_value = apply_corrections_vectorized(
                    gchp_month_data['gchp_NO2'][gchp_lat_idx, gchp_lon_idx],
                    obs_value,
                    gchp_month_data['gchp_alkylnitrates'][gchp_lat_idx, gchp_lon_idx],
                    gchp_month_data['gchp_HNO3'][gchp_lat_idx, gchp_lon_idx],
                    gchp_month_data['gchp_PAN'][gchp_lat_idx, gchp_lon_idx]
                )
                obs_corrected[group_idx, time_idx] = obs_corrected_value
                
                group_has_data = True
                total_filled_points += 1
        
        if group_has_data:
            groups_with_data.add(group_idx)
    
    # Report data coverage
    print(f"Data coverage:")
    print(f"  Grouped sites with data: {len(groups_with_data)} / {n_groups}")
    print(f"  Total filled data points: {total_filled_points:,} / {n_groups * n_times:,} ({100*total_filled_points/(n_groups * n_times):.1f}%)")
    
    # Create coordinate arrays for the grouped sites
    grouped_site_ids = grouped_sites['grouped_site_id'].values
    grouped_lats = grouped_sites['lat'].values
    grouped_lons = grouped_sites['lon'].values
    grouped_gchp_lats = grouped_sites['gchp_lat'].values
    grouped_gchp_lons = grouped_sites['gchp_lon'].values
    
    # Create xarray Dataset
    ds_obs = xr.Dataset(
        data_vars={
            'NO2_cf': (
                ('sites', 'time'),
                obs_corrected,
                {
                    'units': 'ppb',
                    'long_name': 'Corrected NO2 observations (grouped by grid cell)',
                    'description': 'Ground observations averaged within GCHP grid cells, with correction factors applied',
                    '_FillValue': np.nan
                }
            ),
            'NO2_raw': (
                ('sites', 'time'),
                obs_raw,
                {
                    'units': 'ppb',
                    'long_name': 'Raw NO2 observations (grouped by grid cell)',
                    'description': 'Ground observations averaged within GCHP grid cells, no correction',
                    '_FillValue': np.nan
                }
            ),
            'num_original_sites': (
                'sites',
                grouped_sites['num_original_sites'].values,
                {'long_name': 'Number of original observation sites averaged in this grid cell'}
            )
        },
        coords={
            'sites': (
                'sites',
                grouped_site_ids,
                {'long_name': 'Grouped site identifier'}
            ),
            'time': (
                'time',
                time_coords,
                {'long_name': 'Time'}
            ),
            'latitude': (
                'sites',
                grouped_lats,
                {'units': 'degrees_north', 'long_name': 'Average observation site latitude within grid cell'}
            ),
            'longitude': (
                'sites',
                grouped_lons,
                {'units': 'degrees_east', 'long_name': 'Average observation site longitude within grid cell'}
            ),
            'gchp_latitude': (
                'sites',
                grouped_gchp_lats,
                {'units': 'degrees_north', 'long_name': 'GCHP grid latitude'}
            ),
            'gchp_longitude': (
                'sites',
                grouped_gchp_lons,
                {'units': 'degrees_east', 'long_name': 'GCHP grid longitude'}
            )
        },
        attrs={
            'title': 'Monthly NO2 Observations (Grouped by GCHP Grid Cell)',
            'description': 'Ground-based NO2 observations averaged within GCHP grid cells',
            'time_coverage_start': f"{min(years)}-01-01",
            'time_coverage_end': f"{max(years)}-12-31",
            'spatial_coverage': f"{n_groups} GCHP grid cells with observations",
            'correction_method': 'apply_corrections_vectorized with alpha=0.15, beta=0.95',
            'spatial_matching': 'Sites grouped by GCHP grid cell and averaged',
            'averaging_method': 'Multiple sites in same grid cell averaged temporally and spatially',
            'original_sites': f"{len(site_info_matched)} original sites → {n_groups} grouped sites",
            'missing_data_handling': 'NaN for months without observations',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Save to file
    obs_file_out = os.path.join(output_dir, 'NO2_observation_monthly_grouped.nc')
    encoding = {
        'NO2_cf': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'NO2_raw': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'num_original_sites': {'zlib': True, 'complevel': 4, 'dtype': 'int32'}
    }
    ds_obs.to_netcdf(obs_file_out, encoding=encoding)
    print(f"Saved grouped observation file: {obs_file_out}")
    
    return ds_obs, time_coords

def create_geophysical_netcdf_grouped(grouped_sites, gchp_data, time_coords):
    """Create geophysical NetCDF for grouped sites."""
    print("Creating grouped geophysical NetCDF...")
    
    n_groups = len(grouped_sites)
    n_times = len(time_coords)
    
    # Initialize array
    geophysical_no2 = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    
    # Fill array using grid indices
    for group_idx, (_, group_row) in enumerate(grouped_sites.iterrows()):
        gchp_lat_idx = int(group_row['gchp_lat_index'])
        gchp_lon_idx = int(group_row['gchp_lon_index'])
        
        for time_idx, timestamp in enumerate(time_coords):
            year = timestamp.year
            month = timestamp.month
            
            if year in gchp_data and month in gchp_data[year]:
                geophysical_no2[group_idx, time_idx] = gchp_data[year][month]['geophysical_NO2'][gchp_lat_idx, gchp_lon_idx]
    
    # Create xarray Dataset
    ds_geo = xr.Dataset(
        data_vars={
            'NO2_geophysical': (
                ('sites', 'time'),
                geophysical_no2,
                {
                    'units': 'ppb',
                    'long_name': 'Geophysical NO2 (grouped by grid cell)',
                    'description': 'GCHP geophysical NO2 at grouped observation sites',
                    '_FillValue': np.nan
                }
            )
        },
        coords={
            'sites': (
                'sites',
                grouped_sites['grouped_site_id'].values,
                {'long_name': 'Grouped site identifier'}
            ),
            'time': (
                'time',
                time_coords,
                {'long_name': 'Time'}
            ),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 'long_name': 'Average observation site latitude within grid cell'}
            ),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 'long_name': 'Average observation site longitude within grid cell'}
            )
        },
        attrs={
            'title': 'Monthly Geophysical NO2 (Grouped by GCHP Grid Cell)',
            'description': 'GCHP geophysical NO2 extracted at grouped observation sites',
            'source': 'GCHP model output',
            'spatial_matching': 'Grouped by GCHP grid cell',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Save to file
    geo_file_out = os.path.join(output_dir, 'NO2_geophysical_monthly_grouped.nc')
    encoding = {'NO2_geophysical': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}}
    ds_geo.to_netcdf(geo_file_out, encoding=encoding)
    print(f"Saved grouped geophysical file: {geo_file_out}")
    
    return ds_geo

def create_bias_netcdf_grouped(ds_obs, ds_geo):
    """Create bias NetCDF for grouped data."""
    print("Creating grouped bias NetCDF...")
    
    # Calculate bias
    bias_corrected = ds_obs['NO2_cf'] - ds_geo['NO2_geophysical']
    bias_raw = ds_obs['NO2_raw'] - ds_geo['NO2_geophysical']
    
    # Create xarray Dataset
    ds_bias = xr.Dataset(
        data_vars={
            'NO2_bias_cf': (
                ('sites', 'time'),
                bias_corrected.values,
                {
                    'units': 'ppb',
                    'long_name': 'NO2 bias (grouped corrected observations - geophysical)',
                    'description': 'Difference between grouped corrected observations and geophysical model',
                    '_FillValue': np.nan
                }
            ),
            'NO2_bias_raw': (
                ('sites', 'time'),
                bias_raw.values,
                {
                    'units': 'ppb',
                    'long_name': 'NO2 bias (grouped raw observations - geophysical)',
                    'description': 'Difference between grouped raw observations and geophysical model',
                    '_FillValue': np.nan
                }
            )
        },
        coords=ds_obs.coords,
        attrs={
            'title': 'Monthly NO2 Bias (Grouped Observations - Geophysical)',
            'description': 'Bias between grouped ground observations and geophysical model',
            'calculation': 'grouped_observation - geophysical',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Save to file
    bias_file_out = os.path.join(output_dir, 'NO2_bias_monthly_grouped.nc')
    encoding = {
        'NO2_bias_cf': {'zlib': True, 'complevel': 4, 'dtype': 'float32'},
        'NO2_bias_raw': {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
    }
    ds_bias.to_netcdf(bias_file_out, encoding=encoding)
    print(f"Saved grouped bias file: {bias_file_out}")
    
    return ds_bias

def print_summary_statistics_grouped(ds_obs, ds_geo, ds_bias, grouped_sites):
    """Print summary statistics for grouped datasets."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (GROUPED BY GRID CELL)")
    print("="*60)
    
    # Data availability
    obs_valid = ~np.isnan(ds_obs['NO2_cf'].values)
    geo_valid = ~np.isnan(ds_geo['NO2_geophysical'].values)
    bias_valid = ~np.isnan(ds_bias['NO2_bias_cf'].values)
    
    total_points = ds_obs['NO2_cf'].size
    
    print(f"Total possible data points: {total_points:,}")
    print(f"Valid observations: {np.sum(obs_valid):,} ({100*np.mean(obs_valid):.1f}%)")
    print(f"Valid geophysical: {np.sum(geo_valid):,} ({100*np.mean(geo_valid):.1f}%)")
    print(f"Valid bias: {np.sum(bias_valid):,} ({100*np.mean(bias_valid):.1f}%)")
    
    # Grouping statistics
    print(f"\nGrouping statistics:")
    print(f"  Grouped sites: {len(grouped_sites)}")
    print(f"  Total original sites: {grouped_sites['num_original_sites'].sum()}")
    print(f"  Average sites per group: {grouped_sites['num_original_sites'].mean():.2f}")
    print(f"  Max sites in one group: {grouped_sites['num_original_sites'].max()}")
    
    # Value statistics
    if np.sum(obs_valid) > 0:
        obs_vals = ds_obs['NO2_cf'].values[obs_valid]
        print(f"\nObservations (corrected, grouped): {np.mean(obs_vals):.2f} ± {np.std(obs_vals):.2f} ppb")
        print(f"  Range: {np.min(obs_vals):.2f} to {np.max(obs_vals):.2f} ppb")
    
    if np.sum(geo_valid) > 0:
        geo_vals = ds_geo['NO2_geophysical'].values[geo_valid]
        print(f"Geophysical: {np.mean(geo_vals):.2f} ± {np.std(geo_vals):.2f} ppb")
        print(f"  Range: {np.min(geo_vals):.2f} to {np.max(geo_vals):.2f} ppb")
    
    if np.sum(bias_valid) > 0:
        bias_vals = ds_bias['NO2_bias_cf'].values[bias_valid]
        print(f"Bias (corrected, grouped): {np.mean(bias_vals):.2f} ± {np.std(bias_vals):.2f} ppb")
        print(f"  Range: {np.min(bias_vals):.2f} to {np.max(bias_vals):.2f} ppb")
    
    # Temporal coverage
    print(f"\nTemporal coverage:")
    print(f"  Start: {ds_obs.time.values[0]}")
    print(f"  End: {ds_obs.time.values[-1]}")
    print(f"  Time steps: {len(ds_obs.time)}")
    
    # Spatial coverage  
    print(f"\nSpatial coverage:")
    print(f"  Grouped sites: {len(ds_obs.sites)}")
    print(f"  Latitude range: {ds_obs.latitude.min().values:.2f} to {ds_obs.latitude.max().values:.2f}")
    print(f"  Longitude range: {ds_obs.longitude.min().values:.2f} to {ds_obs.longitude.max().values:.2f}")

def main():
    """Main execution function."""
    print("Starting NO2 analysis file generation with grid cell grouping...")
    print("Using site grouping by GCHP grid cell with averaging")
    print("="*60)
    
    # Step 1: Load observations
    years=list(range(2005, 2024))
    obs_df, site_info = load_observations(years)
    
    # Step 2: Load GCHP data (2019 only)
    gchp_data, gchp_coords = load_gchp_data_for_years(years)
    
    if gchp_coords is None:
        print("ERROR: No GCHP data found!")
        return
    
    # Step 3: Match sites to GCHP grid and group by grid cell
    grouped_sites, site_info_matched = match_sites_to_gchp_and_group(site_info, gchp_coords)
    
    if len(grouped_sites) == 0:
        print("ERROR: No sites matched to GCHP grid!")
        return
    
    # Step 4: Create grouped observation NetCDF (with corrections and averaging)
    ds_obs, time_coords = create_observation_netcdf_grouped(
        obs_df, grouped_sites, site_info_matched, gchp_data
    )
    
    # Step 5: Create grouped geophysical NetCDF
    ds_geo = create_geophysical_netcdf_grouped(grouped_sites, gchp_data, time_coords)
    
    # Step 6: Create grouped bias NetCDF
    ds_bias = create_bias_netcdf_grouped(ds_obs, ds_geo)
    
    # Step 7: Print summary
    print_summary_statistics_grouped(ds_obs, ds_geo, ds_bias, grouped_sites)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"  1. {output_dir}/NO2_observation_monthly_grouped.nc")
    print(f"  2. {output_dir}/NO2_geophysical_monthly_grouped.nc") 
    print(f"  3. {output_dir}/NO2_bias_monthly_grouped.nc")
    print("\nNote: Sites in same GCHP grid cell have been averaged together")
    print("Note: Testing with 2019 data only")
    
if __name__ == "__main__":
    main()