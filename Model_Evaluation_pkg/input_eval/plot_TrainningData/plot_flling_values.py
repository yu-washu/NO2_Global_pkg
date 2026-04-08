#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
import os
import gc

# Define file paths
files = {
    '2023_01': '/my-projects2/1.project/GeoNO2/2023/1x1km.GeoNO2.202301.MonMean.nc',
    '2023_07': '/my-projects2/1.project/GeoNO2/2023/1x1km.GeoNO2.202307.MonMean.nc',
    '2005_01': '/my-projects2/1.project/GeoNO2/2005/1x1km.GeoNO2.200501.MonMean.nc',
    '2005_07': '/my-projects2/1.project/GeoNO2/2005/1x1km.GeoNO2.200507.MonMean.nc',
}

# First pass: get lon/lat and determine global vmin/vmax
print('First pass: determining data range...')
lon = None
lat = None
vmin = np.inf
vmax = -np.inf

for key, fpath in files.items():
    if not os.path.exists(fpath):
        print(f'  WARNING: File not found: {fpath}')
        continue
    
    ds = nc.Dataset(fpath, 'r')
    
    # Load lon/lat (only once)
    if lon is None:
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
    
    # Get min/max for both variables
    for var_name in ['filled_GeoNO2', 'gap_GeoNO2']:
        var = ds.variables[var_name]
        var_min = np.nanmin(var[:])
        var_max = np.nanmax(var[:])
        vmin = min(vmin, var_min)
        vmax = max(vmax, var_max)
        print(f'  {key} {var_name}: min={var_min:.6e}, max={var_max:.6e}')
    
    ds.close()

print(f'\nGlobal data range: vmin={vmin:.6e}, vmax={vmax:.6e}')

# Create figure with 4 rows × 2 columns
fig = plt.figure(figsize=(16, 20))

# Define the layout: 4 rows (2023 gap, 2023 filled, 2005 gap, 2005 filled) × 2 columns (Jan, July)
plot_config = [
    # Row 1: 2023 gap
    {'file_key': '2023_01', 'var_name': 'gap_GeoNO2', 'title': '2023 Jan - gap_GeoNO2'},
    {'file_key': '2023_07', 'var_name': 'gap_GeoNO2', 'title': '2023 July - gap_GeoNO2'},
    # Row 2: 2023 filled
    {'file_key': '2023_01', 'var_name': 'filled_GeoNO2', 'title': '2023 Jan - filled_GeoNO2'},
    {'file_key': '2023_07', 'var_name': 'filled_GeoNO2', 'title': '2023 July - filled_GeoNO2'},
    # Row 3: 2005 gap
    {'file_key': '2005_01', 'var_name': 'gap_GeoNO2', 'title': '2005 Jan - gap_GeoNO2'},
    {'file_key': '2005_07', 'var_name': 'gap_GeoNO2', 'title': '2005 July - gap_GeoNO2'},
    # Row 4: 2005 filled
    {'file_key': '2005_01', 'var_name': 'filled_GeoNO2', 'title': '2005 Jan - filled_GeoNO2'},
    {'file_key': '2005_07', 'var_name': 'filled_GeoNO2', 'title': '2005 July - filled_GeoNO2'},
]

# Set up colormap and normalization
cmap = matplotlib.colormaps['RdYlBu_r']
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# Create subplots
print('\nCreating subplots...')
im = None
for idx, config in enumerate(plot_config):
    print(f"  Plotting subplot {idx+1}/8: {config['title']}")
    
    # Create subplot with cartopy projection
    ax = fig.add_subplot(4, 2, idx + 1, projection=ccrs.PlateCarree())
    
    # Load data for this specific subplot
    fpath = files[config['file_key']]
    if not os.path.exists(fpath):
        print(f"    WARNING: File not found: {fpath}")
        continue
    
    ds = nc.Dataset(fpath, 'r')
    data = ds.variables[config['var_name']][:]
    ds.close()
    
    # Mask invalid values
    data_masked = np.ma.masked_invalid(data)
    
    # Print statistics
    print(f"    Min: {np.nanmin(data):.6e}, Max: {np.nanmax(data):.6e}, Mean: {np.nanmean(data):.6e}")
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Plot the data using pcolormesh at full resolution
    im = ax.pcolormesh(lon, lat, data_masked, cmap=cmap, norm=norm,
                       transform=ccrs.PlateCarree(), shading='auto', rasterized=True)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color='gray',
                      alpha=0.3, linestyle='--')
    
    # Add title
    ax.set_title(config['title'], fontsize=12, pad=10)
    
    # Free memory
    del data, data_masked
    gc.collect()
    
    print(f"    Completed subplot {idx+1}/8")

# Add a shared colorbar at the bottom
if im is not None:
    print('\nAdding colorbar...')
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
    cb = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cb.set_label(r'$\rm{NO_{2}}$ mixing ratio (ppb)', fontsize=14)

# Adjust layout to prevent overlap
plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.05, hspace=0.15, wspace=0.1)

# Save the figure
print('\nSaving figure...')
output_file = '/my-projects2/1.project/NO2_DL_global/input_variables/plots/filled_gap_NO2_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'\n{"="*60}')
print(f'Plot saved to: {output_file}')
print(f'{"="*60}')

plt.close(fig)
gc.collect()
