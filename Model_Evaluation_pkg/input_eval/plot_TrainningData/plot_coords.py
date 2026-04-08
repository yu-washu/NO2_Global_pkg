#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib

# Load data
Dir = '/my-projects2/1.project/NO2_DL_global/input_variables/Geographical_Variables_input/Spherical_Coordinates/'
lon = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global.npy')
lat = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global.npy')
S1 = np.load(Dir + 'Spherical_Coordinates_1.npy')
S2 = np.load(Dir + 'Spherical_Coordinates_2.npy')
S3 = np.load(Dir + 'Spherical_Coordinates_3.npy')

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 6))
cmap = matplotlib.colormaps['RdYlBu_r']

# List of data arrays and titles
data_list = [S1, S2, S3]
titles = ['Spherical Coordinate 1', 'Spherical Coordinate 2', 'Spherical Coordinate 3']

# Create subplots
for i, (arr, title) in enumerate(zip(data_list, titles), 1):
    ax = fig.add_subplot(1, 3, i, projection=ccrs.PlateCarree())
    
    # Plot data
    mesh = ax.pcolormesh(lon, lat, arr,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap,
                        vmin=0, 
                        vmax=np.nanmax(arr)*0.5)
    
    # Add coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Set extent
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    
    # Add colorbar for each subplot
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                       pad=0.05, fraction=0.046, aspect=30)
    cbar.set_label(f'S{i}', fontsize=10)
    
    # Set title
    ax.set_title(title, fontsize=12)

plt.tight_layout()

# Save figure
out_png = os.path.join(Dir, 'Spherical_Coordinates.png')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Figure saved: {out_png}")

plt.show()