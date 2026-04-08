#!/usr/bin/env python
import pandas as pd
import numpy as np

obs_file = '/my-projects2/1.project/NO2_ground_complied/global/no2_monthly_observations.csv'
df = pd.read_csv(obs_file)

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

#!/usr/bin/env python
import pandas as pd
import numpy as np
import xarray as xr

# Load your observations data
obs_file = '/my-projects2/1.project/NO2_ground_complied/global/no2_monthly_observations.csv'
df = pd.read_csv(obs_file)

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

def find_nearest_indices(target_coords, reference_coords):
    """Find the nearest indices in reference coordinates for each target coordinate"""
    indices = []
    for coord in target_coords:
        # Find the index of the nearest coordinate
        nearest_idx = np.argmin(np.abs(reference_coords - coord))
        indices.append(nearest_idx)
    return np.array(indices)

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
    
    # Handle NaN values (points not in any region, e.g., over ocean)
    obs['region'] = obs['region'].fillna('Unknown')
    
    print("Region mapping completed!")
    print("Region distribution:")
    print(obs['region'].value_counts())
    
    return obs

def main():
    """Main function to process the data"""
    print("Starting region assignment process...")
    
    # Load continent mask
    mask_lons, mask_lats, continent_mask = load_continent_mask()
    
    # Map observations to regions
    df_with_regions = map_observations_to_regions(df.copy(), mask_lons, mask_lats, continent_mask)
    
    # Save the result
    output_file = obs_file.replace('.csv', '_with_regions.csv')
    df_with_regions.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Display some statistics
    print(f"\nTotal observations: {len(df_with_regions)}")
    print(f"Observations with regions: {len(df_with_regions[df_with_regions['region'] != 'Unknown'])}")
    print(f"Unknown regions: {len(df_with_regions[df_with_regions['region'] == 'Unknown'])}")
    
    return df_with_regions

if __name__ == "__main__":
    # Execute the main function
    df_with_regions = main()
    
    # Optional: Display first few rows to verify
    print("\nFirst few rows of the result:")
    print(df_with_regions[['lat', 'lon', 'region']].head(10))

#(24.45 / 46)
factors = [0.45, 0.35, 0.25]
# factors_name = ['45', '35', '25']
# for i, factor in enumerate(factors):

df_modified = df.copy()
mask = (df_modified['lat'] == 'China') | (df_modified['country'] == 'Japan')

# Apply the division to the filtered rows
df_modified.loc[mask, 'no2_ppb'] = df_modified.loc[mask, 'no2_ppb'] / (24.45 / 46)

outfile = f'/my-projects2/1.project/NO2_ground_complied/global/no2_monthly_observations_asia_ugm3.csv'
df_modified.to_csv(outfile, index=False)