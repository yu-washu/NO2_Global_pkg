#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from matplotlib.colors import ListedColormap

# Define input directory and file paths
InDir = '/my-projects2/1.project/NO2_DL_global/input_variables/'

variables = {'CEDS_NOx': InDir + 'CEDS_Anthro_Emissions_01_input/2019/NO-em-anthro_CMIP_v2025-04_CEDS_Total_001x001_Global_201901.npy',
    # 'Urban_Builtup': InDir + 'LandCover_buffer_forEachPixels_input/Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-6.5-forEachPixel_001x001_Global_2019.npy',
    # 'Forests': InDir + 'LandCover_input/forests/forests-MCD12C1_LandCover_001x001_Global_2019.npy',
    # 'Shrublands': InDir + 'LandCover_input/shrublands/shrublands-MCD12C1_LandCover_001x001_Global_2019.npy',
    # 'Croplands': InDir + 'LandCover_input/croplands/croplands-MCD12C1_LandCover_001x001_Global_2019.npy',
    # 'Water_Bodies': InDir + 'LandCover_input/Water-Bodies/Water-Bodies-MCD12C1_LandCover_001x001_Global_2019.npy',
    # 'Log_Major_Roads': InDir + 'OpenStreetMap_RoadDensity_input/2025/OpenStreetMap-Global-major_roads_log-RoadDensityMap_2025.npy',
    # 'Log_Major_Roads_New': InDir + 'OpenStreetMap_RoadDensity_input/2025/OpenStreetMap-Global-major_roads_new_log-RoadDensityMap_2025.npy',
    # 'Log_Minor_Roads': InDir + 'OpenStreetMap_RoadDensity_input/2025/OpenStreetMap-Global-minor_roads_log-RoadDensityMap_2025.npy',
    # 'Log_Minor_Roads_New': InDir + 'OpenStreetMap_RoadDensity_input/2025/OpenStreetMap-Global-minor_roads_new_log-RoadDensityMap_2025.npy'
}

# Load lon/lat
lon = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global.npy')
lat = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global.npy')

def plot_map(variable_data, variable_name, fpath_out, var_key):
    """
    Plot a single variable on a map
    
    Parameters:
    -----------
    variable_data : numpy array
        The data to plot
    variable_name : str
        Name of the variable for labeling
    fpath_out : str
        Output file path
    var_key : str
        Variable key for special handling
    """
    
    # Mask invalid values (NaN, inf)
    data_masked = np.ma.masked_invalid(variable_data)
    
    # Choose colormap and normalization based on variable type
    if var_key == 'Urban_Builtup':
        # Binary variable: 0 and 1
        cmap = ListedColormap(['white', 'green'])
        vmin, vmax = 0, 1
        norm = matplotlib.colors.BoundaryNorm([0, 0.5, 1], cmap.N)
        
    elif var_key in ['Log_Major_Roads', 'Log_Minor_Roads','Log_Major_Roads_New', 'Log_Minor_Roads_New']:
        cmap = matplotlib.colormaps['RdYlBu_r']
        vmin, vmax = 0, 10
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    elif var_key == 'CEDS_NOx':
        cmap = matplotlib.colormaps['RdYlBu_r']
        vmin = 0
        vmax = 1e-9
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
               
    elif var_key == 'Forests':
        cmap = matplotlib.colormaps['RdYlBu_r']
        vmin = 0
        vmax = 100
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        
    elif var_key == 'Water_Bodies':
        cmap = matplotlib.colormaps['RdYlBu_r']
        vmin = 0
        vmax = 100
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)     

    elif var_key == 'Shrublands':
        cmap = matplotlib.colormaps['RdYlBu_r']
        vmin = 0
        vmax = 100
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)  
                  
    else:
        # Other variables: vmin=min, vmax=0.8*max
        cmap = matplotlib.colormaps['RdYlBu_r']
        vmin = 0
        vmax = 0.5 * np.nanmax(variable_data)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Set up the figure with Cartopy projection
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines, state lines, and country borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    # Plot the data with detailed colormap
    im = ax.pcolormesh(lon, lat, data_masked, cmap=cmap, norm=norm, 
                       transform=ccrs.PlateCarree(), shading='auto')
    
    # Create a custom horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.10, 0.07, 0.8, 0.03])
    cb = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    
    # Set the title for the colorbar
    cb.set_label(variable_name, fontsize=14)
    
    # Add gridlines
    ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', 
                 alpha=0.5, linestyle='--')
    
    # Add title
    ax.set_title(f'{variable_name} - Global Distribution', fontsize=16, pad=20)
    
    # Print statistics
    print(f'\n{variable_name} Statistics:')
    print(f'  Min: {np.nanmin(variable_data):.4f}')
    print(f'  Max: {np.nanmax(variable_data):.4f}')
    print(f'  Mean: {np.nanmean(variable_data):.4f}')
    print(f'  Plot vmin: {vmin:.4f}')
    print(f'  Plot vmax: {vmax:.4f}')
    
    # Save the figure
    plt.savefig(fpath_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved to {fpath_out}')

# Plot all variables
for var_name, var_path in variables.items():
    print(f'\nProcessing {var_name}...')
    
    # Check if file exists
    if not os.path.exists(var_path):
        print(f'  WARNING: File not found: {var_path}')
        continue
    
    # Load data
    try:
        data = np.load(var_path)
        
        # Create output filename
        output_file = os.path.join(InDir, f'{var_name}_global_map.png')
        
        # Plot
        plot_map(data, var_name.replace('_', ' '), output_file, var_name)
        
    except Exception as e:
        print(f'  ERROR loading/plotting {var_name}: {str(e)}')
        continue

print('\n' + '='*50)
print('All plots completed!')
print(f'Plots saved in: {InDir}/')
print('='*50)