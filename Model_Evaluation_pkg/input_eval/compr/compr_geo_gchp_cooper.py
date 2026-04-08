import argparse
import numpy as np
import pandas as pd
import xarray as xr
import os
from functools import lru_cache
import gc

# Parameters
res = '1x1km'
species = 'NO2'
year = 2023          # overridden by --year argument at runtime
cooper_regions = ['northamerica', 'southamerica', 'oceania', 'europe', 'asia', 'africa']
region = 'global'
obs_version = 'v6'

InDir = '/my-projects2/1.project/cooper/'
GCHP_Dir = f'/my-projects2/1.project/GeoNO2-v5/{year}/'
OutDir = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
Obs_Dir = '/my-projects2/1.project/NO2_ground_compiled/monthly/global/'

def get_nearest_grid_index(site_lon, site_lat, grid_lon_min, grid_lat_min, resolution=0.01):
    """Get nearest grid indices using direct calculation"""
    index_lon = np.round((site_lon - grid_lon_min) / resolution).astype(np.int32)
    index_lat = np.round((site_lat - grid_lat_min) / resolution).astype(np.int32)
    return index_lon, index_lat

def apply_corrections_vectorized(base, obs, alkylnitrates, hno3, pan, alpha=0.15, beta=0.95):
    """Apply correction factors to observations"""
    denom = base + alkylnitrates + alpha * hno3 + beta * pan
    denom_safe = np.where(denom == 0, 1, denom)
    cf = obs * (base / denom_safe)
    return cf

@lru_cache(maxsize=128)
def get_cooper_region_bounds():
    """Cache Cooper region bounds to avoid repeated file I/O"""
    bounds = {}
    for region in cooper_regions:
        cooper_file = os.path.join(InDir, f'TROPOMI-inferred_surface_no2_{region}_2019_annual_mean.nc')
        if not os.path.isfile(cooper_file):
            continue
        try:
            cooper_ds = xr.open_dataset(cooper_file, engine='netcdf4').squeeze()
            bounds[region] = {
                'lat_bounds': (cooper_ds.LAT_CENTER.min().item(), cooper_ds.LAT_CENTER.max().item()),
                'lon_bounds': (cooper_ds.LON_CENTER.min().item(), cooper_ds.LON_CENTER.max().item())
            }
        except:
            continue
    return bounds

def determine_cooper_region_vectorized(lats, lons):
    """Vectorized region determination"""
    region_bounds = get_cooper_region_bounds()
    regions = np.full(len(lats), 'unknown', dtype='U15')
    
    for region, bounds in region_bounds.items():
        lat_min, lat_max = bounds['lat_bounds']
        lon_min, lon_max = bounds['lon_bounds']
        
        mask = ((lats >= lat_min) & (lats <= lat_max) & 
                (lons >= lon_min) & (lons <= lon_max))
        regions[mask] = region
    
    return regions

def load_and_prepare_data():
    """Load and prepare observation data with memory optimization"""
    print("Loading observation data...")
    
    # Use chunking for large files
    chunk_size = 100000
    obs_chunks = []

    for chunk in pd.read_csv(
        os.path.join(Obs_Dir, 'combined_global_no2_2005-2023_v6_filtered.csv'),
        chunksize=chunk_size,
        low_memory=False,
        dtype={'no2': float, 'year': int}
    ):
        year_chunk = chunk[chunk['year'] == year]
        if len(year_chunk) > 0:
            obs_chunks.append(year_chunk)
    
    obs_year = pd.concat(obs_chunks, ignore_index=True) if obs_chunks else pd.DataFrame()
    print(f"Loaded {len(obs_year)} monthly observations for {year}")
    
    return obs_year

def load_gchp_coordinates():
    """Load only GCHP coordinates, not full data"""
    print("Loading GCHP coordinates...")
    # Find first available file
    for month in range(1, 13):
        month_str = f"{month:02d}"
        gchp_file = os.path.join(GCHP_Dir, f'{res}.GeoNO2.{year}{month_str}.MonMean.nc')
        if os.path.exists(gchp_file):
            ds = xr.open_dataset(gchp_file, engine='netcdf4').squeeze()
            gchp_lat = np.array(ds.lat, dtype=np.float32)
            gchp_lon = np.array(ds.lon, dtype=np.float32)
            ds.close()
            
            lat_min, lat_max = gchp_lat.min(), gchp_lat.max()
            lon_min, lon_max = gchp_lon.min(), gchp_lon.max()
            
            return {
                'lat': gchp_lat,
                'lon': gchp_lon,
                'lat_bounds': (lat_min, lat_max),
                'lon_bounds': (lon_min, lon_max)
            }
    
    raise FileNotFoundError("No GCHP files found!")

def process_monthly_data_optimized(obs_data, gchp_coords):
    """Optimized monthly data processing"""
    print("Processing monthly data...")
    
    # Pre-calculate grid indices for all observations
    site_lats = obs_data['lat'].values
    site_lons = obs_data['lon'].values
    
    lat_min = gchp_coords['lat_bounds'][0]
    lon_min = gchp_coords['lon_bounds'][0]
    gchp_lat = gchp_coords['lat']
    gchp_lon = gchp_coords['lon']
    
    index_lon, index_lat = get_nearest_grid_index(site_lons, site_lats, lon_min, lat_min)
    index_lat = np.clip(index_lat, 0, len(gchp_lat) - 1)
    index_lon = np.clip(index_lon, 0, len(gchp_lon) - 1)
    
    # Add indices to dataframe
    obs_data_indexed = obs_data.copy()
    obs_data_indexed['gchp_lat_index'] = index_lat
    obs_data_indexed['gchp_lon_index'] = index_lon
    
    # Group by pixel-month and aggregate
    print("Grouping observations by pixel-month...")
    monthly_grouped = (obs_data_indexed
                      .groupby(['gchp_lat_index', 'gchp_lon_index', 'mon'])
                      .agg({
                          'lat': 'mean',
                          'lon': 'mean',
                          'no2': 'mean',
                          'year': 'first'  # Same for all
                      })
                      .reset_index())
    
    # Count sites per pixel-month
    site_counts = (obs_data_indexed
                   .groupby(['gchp_lat_index', 'gchp_lon_index', 'mon'])
                   .size()
                   .reset_index(name='num_sites'))
    
    monthly_grouped = monthly_grouped.merge(site_counts, on=['gchp_lat_index', 'gchp_lon_index', 'mon'])
    
    # Vectorized region determination
    print("Determining Cooper regions...")
    regions = determine_cooper_region_vectorized(monthly_grouped['lat'].values, monthly_grouped['lon'].values)
    monthly_grouped['region'] = regions
    
    # Process GCHP data month by month to save memory
    print("Extracting GCHP values...")
    monthly_rows = []
    
    # Group by month to minimize file I/O
    for month in monthly_grouped['mon'].unique():
        month_data_df = monthly_grouped[monthly_grouped['mon'] == month].copy()
        
        # Load only this month's GCHP data
        month_str = f"{int(month):02d}"
        gchp_file = os.path.join(GCHP_Dir, f'{res}.GeoNO2.{year}{month_str}.MonMean.nc')
  
        if not os.path.exists(gchp_file):
            print(f"[WARNING] Missing GCHP file for month {month}")
            continue

        
        print(f"Processing month {month}...")
        
        with xr.open_dataset(gchp_file, engine='netcdf4') as ds:
            gchp_no2_data = ds['gchp_NO2'].values
            geo_no2_tot_data = ds['filled_GeoNO2_tot'].values
            geo_no2_trop_data = ds['filled_GeoNO2_trop'].values
            filled_SatColNO2_trop_gcshape_data = ds['filled_SatColNO2_trop_gcshape'].values
            filled_SatColNO2_tot_gcshape_data = ds['filled_SatColNO2_tot_gcshape'].values
            filled_SatColNO2_trop_data = ds['filled_SatColNO2_trop'].values
            filled_SatColNO2_tot_data = ds['filled_SatColNO2_tot'].values
            gchp_NO2col_tot_data = ds['gchp_NO2col_tot'].values
            gchp_NO2col_trop_data = ds['gchp_NO2col_trop'].values
            gchp_eta_tot_data = ds['gchp_eta_tot'].values
            gchp_eta_trop_data = ds['gchp_eta_trop'].values
        
        # Vectorized extraction
        lat_indices = month_data_df['gchp_lat_index'].values
        lon_indices = month_data_df['gchp_lon_index'].values
        
        gchp_values = gchp_no2_data[lat_indices, lon_indices]
        geo_no2_tot_values = geo_no2_tot_data[lat_indices, lon_indices]
        geo_no2_trop_values = geo_no2_trop_data[lat_indices, lon_indices]
        filled_SatColNO2_trop_gcshape_values = filled_SatColNO2_trop_gcshape_data[lat_indices, lon_indices]
        filled_SatColNO2_tot_gcshape_values = filled_SatColNO2_tot_gcshape_data[lat_indices, lon_indices]
        filled_SatColNO2_trop_values = filled_SatColNO2_trop_data[lat_indices, lon_indices]
        filled_SatColNO2_tot_values = filled_SatColNO2_tot_data[lat_indices, lon_indices]
        gchp_NO2col_tot_values = gchp_NO2col_tot_data[lat_indices, lon_indices]
        gchp_NO2col_trop_values = gchp_NO2col_trop_data[lat_indices, lon_indices]
        gchp_eta_tot_values = gchp_eta_tot_data[lat_indices, lon_indices]
        gchp_eta_trop_values = gchp_eta_trop_data[lat_indices, lon_indices]
        
        # Create batch of rows
        month_rows = {
            'region': month_data_df['region'].values,
            'lat': month_data_df['lat'].values,
            'lon': month_data_df['lon'].values,
            'year': month_data_df['year'].values,
            'month': month_data_df['mon'].values,
            'obs_no2': month_data_df['no2'].values,
            'gchp_no2': gchp_values,
            'geophysical_no2_tot': geo_no2_tot_values,
            'geophysical_no2_trop': geo_no2_trop_values,
            'filled_SatColNO2_trop_gcshape': filled_SatColNO2_trop_gcshape_values,
            'filled_SatColNO2_tot_gcshape': filled_SatColNO2_tot_gcshape_values,
            'filled_SatColNO2_trop': filled_SatColNO2_trop_values,
            'filled_SatColNO2_tot': filled_SatColNO2_tot_values,
            'gchp_NO2col_tot': gchp_NO2col_tot_values,
            'gchp_NO2col_trop': gchp_NO2col_trop_values,
            'gchp_eta_tot': gchp_eta_tot_values,
            'gchp_eta_trop': gchp_eta_trop_values,
            'num_sites': month_data_df['num_sites'].values
        }
        
        monthly_rows.append(pd.DataFrame(month_rows))
        
        # Clean up memory
        del gchp_no2_data, geo_no2_tot_data, geo_no2_trop_data, filled_SatColNO2_trop_gcshape_data, filled_SatColNO2_tot_gcshape_data, filled_SatColNO2_trop_data, filled_SatColNO2_tot_data, gap_SatColNO2_tot_data, gap_SatColNO2_trop_data, gap_SatColNO2_tot_gcshape_data, gap_SatColNO2_trop_gcshape_data, gchp_NO2col_tot_data, gchp_NO2col_trop_data, gchp_eta_tot_data, gchp_eta_trop_data
        gc.collect()
    
    return pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()

def get_grid_resolution(lat_array, lon_array):
    """Calculate grid resolution from coordinate arrays"""
    # Flatten arrays in case they're 2D
    lat_flat = np.array(lat_array).flatten()
    lon_flat = np.array(lon_array).flatten()
    
    # Handle edge cases
    if len(lat_flat) < 2 or len(lon_flat) < 2:
        print(f"Warning: Insufficient coordinate points for resolution calculation")
        print(f"  lat points: {len(lat_flat)}, lon points: {len(lon_flat)}")
        return 0.01, 0.01  # fallback to default resolution
    
    # Calculate differences on sorted arrays
    lat_diffs = np.abs(np.diff(np.sort(lat_flat)))
    lon_diffs = np.abs(np.diff(np.sort(lon_flat)))
    
    # Remove zero differences (duplicate coordinates)
    lat_diffs = lat_diffs[lat_diffs > 1e-10]  # Use small epsilon instead of exact zero
    lon_diffs = lon_diffs[lon_diffs > 1e-10]
    
    if len(lat_diffs) == 0 or len(lon_diffs) == 0:
        print(f"Warning: No valid coordinate differences found")
        return 0.01, 0.01  # fallback to default resolution
    
    lat_res = lat_diffs.min()
    lon_res = lon_diffs.min()
    
    return lat_res, lon_res

def load_cooper_data_efficient():
    """Load Cooper data more efficiently with resolution detection"""
    print("Loading Cooper data...")
    cooper_data = {}
    
    for region in cooper_regions:
        cooper_file = os.path.join(InDir, f'TROPOMI-inferred_surface_no2_{region}_{year}_annual_mean.nc')
        if not os.path.isfile(cooper_file):
            print(f"[WARNING] Cooper file for region '{region}' not found.")
            continue
        
        with xr.open_dataset(cooper_file, engine='netcdf4') as cooper_ds:
            lat_coords = np.array(cooper_ds.LAT_CENTER, dtype=np.float32)
            lon_coords = np.array(cooper_ds.LON_CENTER, dtype=np.float32)
            
            # Calculate actual grid resolution
            lat_res, lon_res = get_grid_resolution(lat_coords, lon_coords)
            
            cooper_data[region] = {
                'lat': lat_coords,
                'lon': lon_coords,
                'cooper_NO2': np.array(cooper_ds['surface_no2_ppb']),
                'lat_bounds': (cooper_ds.LAT_CENTER.min().item(), cooper_ds.LAT_CENTER.max().item()),
                'lon_bounds': (cooper_ds.LON_CENTER.min().item(), cooper_ds.LON_CENTER.max().item()),
                'lat_resolution': lat_res,
                'lon_resolution': lon_res
            }
        
        print(f"Loaded Cooper data for region: {region}")
        print(f"  Grid resolution: {lat_res:.4f}° lat x {lon_res:.4f}° lon")
    
    return cooper_data

def process_annual_data_optimized(obs_data, cooper_data):
    """Optimized annual data processing"""
    print("Processing annual data...")
    
    # Pre-aggregate annual observations
    annual_grouped = (obs_data
                     .groupby(['lat', 'lon'])
                     .agg({
                         'no2': 'mean'
                     })
                     .reset_index()
                     .rename(columns={'no2': 'annual_avg_NO2'}))
    
    site_counts = (obs_data
                   .groupby(['lat', 'lon'])
                   .size()
                   .reset_index(name='num_months'))
    
    annual_grouped = annual_grouped.merge(site_counts, on=['lat', 'lon'])
    
    # Vectorized region assignment
    regions = determine_cooper_region_vectorized(
        annual_grouped['lat'].values, 
        annual_grouped['lon'].values
    )
    annual_grouped['region'] = regions
    
    annual_rows = []
    
    for region, region_data in cooper_data.items():
        print(f"Processing annual data for region: {region}")
        print(f"  Cooper data shape: {region_data['cooper_NO2'].shape}")
        print(f"  Cooper data range: {np.nanmin(region_data['cooper_NO2']):.3f} to {np.nanmax(region_data['cooper_NO2']):.3f}")
        print(f"  Grid resolution: {region_data['lat_resolution']:.6f}° lat x {region_data['lon_resolution']:.6f}° lon")
        
        # Filter sites for this region
        region_sites = annual_grouped[annual_grouped['region'] == region].copy()
        
        if len(region_sites) == 0:
            print(f"  No sites found for region {region}")
            continue
        
        print(f"  Found {len(region_sites)} sites in region")
        
        # Get Cooper grid info
        cooper_lat = region_data['lat'].flatten()
        cooper_lon = region_data['lon'].flatten()
        
        print(f"  Cooper grid size: {len(cooper_lat)} lat x {len(cooper_lon)} lon")
        print(f"  Cooper lat range: {cooper_lat.min():.3f} to {cooper_lat.max():.3f}")
        print(f"  Cooper lon range: {cooper_lon.min():.3f} to {cooper_lon.max():.3f}")
        
        # Site coordinates
        site_lats = region_sites['lat'].values
        site_lons = region_sites['lon'].values
        
        print(f"  Site lat range: {site_lats.min():.3f} to {site_lats.max():.3f}")
        print(f"  Site lon range: {site_lons.min():.3f} to {site_lons.max():.3f}")
        
        # Find nearest grid points using broadcasting (more robust)
        lat_distances = np.abs(cooper_lat[:, np.newaxis] - site_lats)
        lon_distances = np.abs(cooper_lon[:, np.newaxis] - site_lons)
        
        lat_indices = np.argmin(lat_distances, axis=0)
        lon_indices = np.argmin(lon_distances, axis=0)
        
        print(f"  Index ranges: lat {lat_indices.min()}-{lat_indices.max()}, lon {lon_indices.min()}-{lon_indices.max()}")
        
        region_sites['cooper_lat_index'] = lat_indices
        region_sites['cooper_lon_index'] = lon_indices
        
        # Group by Cooper pixel
        cooper_grouped = (region_sites
                         .groupby(['cooper_lat_index', 'cooper_lon_index'])
                         .agg({
                             'lat': 'mean',
                             'lon': 'mean',
                             'annual_avg_NO2': 'mean',
                             'num_months': 'sum'
                         })
                         .reset_index())
        
        cooper_site_counts = (region_sites
                             .groupby(['cooper_lat_index', 'cooper_lon_index'])
                             .size()
                             .reset_index(name='num_sites'))
        
        cooper_grouped = cooper_grouped.merge(cooper_site_counts, on=['cooper_lat_index', 'cooper_lon_index'])
        
        # Extract Cooper values
        lat_indices_final = cooper_grouped['cooper_lat_index'].values
        lon_indices_final = cooper_grouped['cooper_lon_index'].values
        
        print(f"  Final extraction - {len(lat_indices_final)} pixels")
        print(f"  Final index ranges: lat {lat_indices_final.min()}-{lat_indices_final.max()}, lon {lon_indices_final.min()}-{lon_indices_final.max()}")
        
        # Extract values based on Cooper data structure
        cooper_no2_data = region_data['cooper_NO2']
        if cooper_no2_data.ndim == 2:
            print("  Using 2D indexing")
            cooper_values = cooper_no2_data[lat_indices_final, lon_indices_final]
        else:
            print("  Using 1D indexing")
            if len(cooper_lat) > len(cooper_lon):
                # More latitude points - data is likely organized as [lat, lon]
                flat_indices = lat_indices_final * len(cooper_lon) + lon_indices_final
            else:
                # More longitude points - data is likely organized as [lon, lat] 
                flat_indices = lon_indices_final * len(cooper_lat) + lat_indices_final
            
            flat_indices = np.clip(flat_indices, 0, len(cooper_no2_data) - 1)
            cooper_values = cooper_no2_data[flat_indices]
        
        # Check for valid values
        valid_mask = ~np.isnan(cooper_values)
        print(f"  Extracted values: {np.sum(valid_mask)} valid out of {len(cooper_values)}")
        
        if np.sum(valid_mask) > 0:
            print(f"  Valid Cooper values range: {cooper_values[valid_mask].min():.3f} to {cooper_values[valid_mask].max():.3f}")
        
        # Replace NaN with flag value
        cooper_values = np.where(valid_mask, cooper_values, -999.0)
        
        # Create batch of rows
        region_rows = pd.DataFrame({
            'region': region,
            'lat': cooper_grouped['lat'].values,
            'lon': cooper_grouped['lon'].values,
            'year': year,
            'obs_no2': cooper_grouped['annual_avg_NO2'].values,
            'cooper_no2': cooper_values,
            'num_sites': cooper_grouped['num_sites'].values
        })
        
        annual_rows.append(region_rows)
    
    return pd.concat(annual_rows, ignore_index=True) if annual_rows else pd.DataFrame()

def main():
    """Optimized main processing function"""
    global year, GCHP_Dir, OutDir

    parser = argparse.ArgumentParser(description='NO2 data processing and evaluation')
    parser.add_argument('--year', type=int, help='Processing year (default: %(default)s)')
    args = parser.parse_args()

    year = args.year
    GCHP_Dir = f'/my-projects2/1.project/GeoNO2-v5/TROPOMI/{year}/'
    OutDir    = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
    os.makedirs(OutDir, exist_ok=True)

    print(f"Starting optimized NO2 data processing for year {year}...")
    
    # Load data efficiently
    obs_data = load_and_prepare_data()
    gchp_coords = load_gchp_coordinates()
    cooper_data = load_cooper_data_efficient()
    
    # Process monthly data
    monthly_df = process_monthly_data_optimized(obs_data, gchp_coords)
    monthly_outfile = os.path.join(OutDir, f'{species}_monthly_{year}_obs{obs_version}.csv')
    monthly_df.to_csv(monthly_outfile, index=False)
    print(f"Saved monthly data: {monthly_outfile} ({len(monthly_df)} rows)")
    
    # Process annual data
    annual_df = process_annual_data_optimized(obs_data, cooper_data)
    
    # Create annual GCHP/Geophysical data from monthly
    if len(monthly_df) > 0:
        annual_from_monthly = (monthly_df
                              .groupby(['region', 'lat', 'lon'])
                              .agg({
                                  'obs_no2': 'mean',
                                  'gchp_no2': 'mean',
                                  'geophysical_no2_tot': 'mean',
                                  'geophysical_no2_trop': 'mean',
                                  'gap_SatColNO2_tot': 'mean',
                                  'gap_SatColNO2_trop': 'mean',
                                  'gap_SatColNO2_tot_gcshape': 'mean',
                                  'gap_SatColNO2_trop_gcshape': 'mean',
                                  'filled_SatColNO2_trop_gcshape': 'mean',
                                  'filled_SatColNO2_tot_gcshape': 'mean',
                                  'filled_SatColNO2_trop': 'mean',  
                                  'filled_SatColNO2_tot': 'mean',
                                  'gchp_NO2col_tot': 'mean',
                                  'gchp_NO2col_trop': 'mean',
                                  'gchp_eta_tot': 'mean',
                                  'gchp_eta_trop': 'mean',
                                  'num_sites': 'sum'
                              })
                              .reset_index())
        annual_from_monthly['year'] = year
        
        # Reorder columns
        annual_from_monthly = annual_from_monthly[['region', 'lat', 'lon', 'year', 'obs_no2', 'gchp_no2', 'geophysical_no2_tot', 'geophysical_no2_trop', 'gap_SatColNO2_tot', 'gap_SatColNO2_trop', 'gap_SatColNO2_tot_gcshape', 'gap_SatColNO2_trop_gcshape', 'filled_SatColNO2_trop_gcshape', 'filled_SatColNO2_tot_gcshape', 'filled_SatColNO2_trop', 'filled_SatColNO2_tot', 'gchp_NO2col_tot', 'gchp_NO2col_trop', 'gchp_eta_tot', 'gchp_eta_trop', 'num_sites']]
        
        annual_gchp_geo_outfile = os.path.join(OutDir, f'{species}_annual_{year}_obs{obs_version}.csv')
        annual_from_monthly.to_csv(annual_gchp_geo_outfile, index=False)
        print(f"Saved annual GCHP/Geophysical/SatColNO2 data: {annual_gchp_geo_outfile} ({len(annual_from_monthly)} rows)")
    
    # Save Cooper annual data
    if len(annual_df) > 0:
        annual_outfile = os.path.join(OutDir, f'{species}_annual_cooper_{year}_obs{obs_version}.csv')
        annual_df.to_csv(annual_outfile, index=False)
        print(f"Saved annual Cooper data: {annual_outfile} ({len(annual_df)} rows)")
    
    print("\nOptimized processing completed successfully!")

if __name__ == "__main__":
    main()