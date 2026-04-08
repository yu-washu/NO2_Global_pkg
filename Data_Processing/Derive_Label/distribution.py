import numpy as np
import pandas as pd
import xarray as xr
import os
from functools import lru_cache
import gc

# Your existing code
OutDir = '/my-projects2/1.project/NO2_DL_global/TrainingDatasets/Global_NO2_BenchMark/'
Obs_Dir = '/my-projects2/1.project//NO2_ground_complied/global/'

def load_and_prepare_data():
    """Load and prepare observation data with memory optimization"""
    print("Loading observation data...")
    
    # Use chunking for large files
    chunk_size = 100000
    obs_chunks = []
    
    for chunk in pd.read_csv(
        os.path.join(Obs_Dir, 'no2_monthly_observations.csv'),
        chunksize=chunk_size,
        low_memory=False,
        dtype={'no2_ppb': float, 'year': int}
    ):
        obs_chunks.append(chunk)
    
    obs = pd.concat(obs_chunks, ignore_index=True) if obs_chunks else pd.DataFrame()
    print(f"Loaded {len(obs)} observations")
    
    return obs

def load_continent_mask():
    """Load the continent mask from netCDF file"""
    print("Loading continent mask...")
    mask_file = '/my-projects2/supportData/Global_Masks/continent_mask_6regions.nc'
    
    # Load the mask with xarray
    ds = xr.open_dataset(mask_file)
    
    # Extract coordinates and mask
    mask_lons = ds.lon.values
    mask_lats = ds.lat.values
    continent_mask = ds.continent_mask.values
    
    print(f"Mask shape: {continent_mask.shape}")
    print(f"Lat range: {mask_lats.min():.4f} to {mask_lats.max():.4f}")
    print(f"Lon range: {mask_lons.min():.4f} to {mask_lons.max():.4f}")
    
    ds.close()
    return mask_lons, mask_lats, continent_mask

def find_nearest_indices(obs_coords, mask_coords):
    """Find nearest indices in mask coordinates for observation coordinates"""
    # Use numpy's searchsorted for efficient nearest neighbor search
    indices = np.searchsorted(mask_coords, obs_coords)
    
    # Handle edge cases
    indices = np.clip(indices, 0, len(mask_coords) - 1)
    
    # Check if the point before is closer
    indices_before = np.clip(indices - 1, 0, len(mask_coords) - 1)
    
    # Calculate distances
    dist_after = np.abs(mask_coords[indices] - obs_coords)
    dist_before = np.abs(mask_coords[indices_before] - obs_coords)
    
    # Use the closer index
    closer_before = dist_before < dist_after
    indices = np.where(closer_before, indices_before, indices)
    
    return indices

def map_observations_to_regions(obs, mask_lons, mask_lats, continent_mask):
    """Map observation coordinates to continent regions"""
    print("Mapping observations to regions...")
    
    # Region mapping from the netCDF metadata
    region_mapping = {
        0: 'Asia',
        1: 'North America', 
        2: 'Europe',
        3: 'Africa',
        4: 'South America',
        5: 'Oceania-Australia'
    }
    
    # Find nearest indices for lat and lon
    lat_indices = find_nearest_indices(obs['lat'].values, mask_lats)
    lon_indices = find_nearest_indices(obs['lon'].values, mask_lons)
    
    # Extract region codes from the mask
    region_codes = continent_mask[lat_indices, lon_indices]
    
    # Map region codes to region names
    obs['region'] = pd.Series(region_codes).map(region_mapping)
    
    # Handle NaN values (points not in any region)
    obs['region'] = obs['region'].fillna('Unknown')
    
    print("Region mapping completed!")
    print("Region distribution:")
    print(obs['region'].value_counts())
    
    return obs

def create_no2_summary_table(obs_with_regions):
    """
    Create comprehensive summary statistics table for NO2 monitors by region
    
    Parameters:
    obs_with_regions: DataFrame with columns ['lat', 'lon', 'year', 'mon', 'no2_ppb', 'country', 'region']
    
    Returns:
    summary_df: DataFrame with regional statistics
    """
    
    print("Creating NO2 monitor summary table...")
    
    # Remove any rows with missing NO2 values
    clean_data = obs_with_regions.dropna(subset=['no2_ppb'])
    
    # Get year range for the dataset
    year_min = clean_data['year'].min()
    year_max = clean_data['year'].max()
    year_range = f"{year_min}-{year_max}"
    
    print(f"Data spans from {year_min} to {year_max}")
    
    # Initialize results list
    summary_results = []
    
    # Calculate statistics for each region
    regions = clean_data['region'].unique()
    regions = sorted([r for r in regions if r != 'Unknown'])  # Sort regions, exclude Unknown if present
    
    for region in regions:
        region_data = clean_data[clean_data['region'] == region]
        
        # Monthly averages (n) - total number of monthly observations
        daily_averages = len(region_data)
        
        # Unique monitors (n) - count unique lat/lon combinations
        unique_monitors = region_data[['lat', 'lon']].drop_duplicates().shape[0]
        
        # NO2 statistics
        no2_values = region_data['no2_ppb']
        
        min_no2 = no2_values.min()
        max_no2 = no2_values.max()
        mean_no2 = no2_values.mean()
        std_no2 = no2_values.std()
        
        # Percentiles
        percentiles = np.percentile(no2_values, [25, 50, 75, 90])
        p25, p50, p75, p90 = percentiles
        
        # Year range for this region
        region_year_min = region_data['year'].min()
        region_year_max = region_data['year'].max()
        region_year_range = f"{region_year_min}-{region_year_max}"
        
        summary_results.append({
            'Region': region,
            'Year Range': region_year_range,
            'Monthly averages (n)': f"{daily_averages:,}",
            'Monitors (n)': unique_monitors,
            'Min NO₂ (ppb)': float(f'{min_no2:.3g}'),
            'Max NO₂ (ppb)': float(f'{max_no2:.3g}'),
            'Mean NO₂ (ppb)': float(f'{mean_no2:.3g}'),
            'SD NO₂ (ppb)': float(f'{std_no2:.3g}'),
            '25th %': float(f'{p25:.3g}'),
            '50th %': float(f'{p50:.3g}'),
            '75th %': float(f'{p75:.3g}'),
            '90th %': float(f'{p90:.3g}')
        })
    
    # Calculate global statistics
    daily_averages_global = len(clean_data)
    unique_monitors_global = clean_data[['lat', 'lon']].drop_duplicates().shape[0]
    
    no2_global = clean_data['no2_ppb']
    percentiles_global = np.percentile(no2_global, [25, 50, 75, 90])
    
    summary_results.append({
        'Region': 'Global',
        'Year Range': year_range,
        'Monthly averages (n)': f"{daily_averages_global:,}",
        'Monitors (n)': unique_monitors_global,
        'Min NO₂ (ppb)': int(round(no2_global.min())),
        'Max NO₂ (ppb)': int(round(no2_global.max())),
        'Mean NO₂ (ppb)': int(round(no2_global.mean())),
        'SD NO₂ (ppb)': int(round(no2_global.std())),
        '25th %': int(round(percentiles_global[0])),
        '50th %': int(round(percentiles_global[1])),
        '75th %': int(round(percentiles_global[2])),
        '90th %': int(round(percentiles_global[3]))
    })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_results)
    
    print(f"\nNO2 Monitor Summary Table ({year_range}):")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    
    return summary_df, year_range

def display_formatted_table(summary_df, year_range):
    """
    Display the summary table in a nicely formatted way
    """
    print(f"\n\nNO₂ Monitor Summary Statistics ({year_range})")
    print("=" * 120)
    
    # Create formatted display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(summary_df.to_string(index=False))
    
    # Reset display options
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')

def create_additional_insights(obs_with_regions):
    """
    Provide additional insights about the data
    """
    print("\n\nAdditional Data Insights:")
    print("-" * 50)
    
    clean_data = obs_with_regions.dropna(subset=['no2_ppb'])
    
    # Temporal coverage
    years = sorted(clean_data['year'].unique())
    months = sorted(clean_data['mon'].unique())
    print(f"Years covered: {min(years)} to {max(years)} ({len(years)} years)")
    print(f"Months covered: {min(months)} to {max(months)}")
    
    # Geographic coverage
    lat_range = (clean_data['lat'].min(), clean_data['lat'].max())
    lon_range = (clean_data['lon'].min(), clean_data['lon'].max())
    print(f"Latitude range: {lat_range[0]:.2f}° to {lat_range[1]:.2f}°")
    print(f"Longitude range: {lon_range[0]:.2f}° to {lon_range[1]:.2f}°")
    
    # Countries represented
    countries = clean_data['country'].nunique()
    print(f"Countries represented: {countries}")
    
    # Data quality
    total_obs = len(obs_with_regions)
    valid_obs = len(clean_data)
    print(f"Total observations: {total_obs:,}")
    print(f"Valid NO₂ observations: {valid_obs:,} ({100*valid_obs/total_obs:.1f}%)")

# Main execution
if __name__ == "__main__":
    # Load observation data
    obs = load_and_prepare_data()
    
    # Load continent mask
    mask_lons, mask_lats, continent_mask = load_continent_mask()
    
    # Map observations to regions
    obs_with_regions = map_observations_to_regions(obs, mask_lons, mask_lats, continent_mask)
    
    # Display sample results
    print("\nSample of data with regions:")
    print(obs_with_regions[['lat', 'lon', 'year', 'mon', 'no2_ppb', 'country', 'region']].head(10))
    
    # Clean up memory
    del mask_lons, mask_lats, continent_mask
    gc.collect()
    
    print("\nRegion column successfully added to observations dataframe!")
    
    output_file = os.path.join(Obs_Dir, 'no2_monthly_observations_with_regions.csv')
    # obs_with_regions.to_csv(output_file, index=False)
    print(f"Updated data saved to: {output_file}")
    
    available_years = sorted(obs_with_regions['year'].dropna().unique())
    print(f"Creating summary tables for years: {available_years}")

    # Create summary table for each year
    for year in available_years:
        print(f"\nCreating summary for year {year}...")
        
        # Filter data for current year
        obs_year = obs_with_regions[obs_with_regions['year'] == year].copy()
        
        # Create summary table for this year
        summary_table_year, year_range = create_no2_summary_table(obs_year)
        
        # Save yearly summary
        yearly_summary_file = OutDir + f'NO2_monitor_summary_{year}.csv'
        summary_table_year.to_csv(yearly_summary_file, index=False)
        print(f"Year {year} summary saved to: {yearly_summary_file}")
    
    summary_table, year_range = create_no2_summary_table(obs_with_regions)
    
    # Display formatted table
    display_formatted_table(summary_table, year_range)
    
    # Show additional insights
    create_additional_insights(obs_with_regions)
    
    output_file = OutDir+ 'NO2_monitor_summary.csv'
    summary_table.to_csv(output_file, index=False)
    print(f"\nSummary table saved to: {output_file}")
    
    print("\n" + "="*50)
    print("Summary table generation completed!")