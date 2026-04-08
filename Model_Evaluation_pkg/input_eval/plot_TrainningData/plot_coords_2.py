import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import os

# Generate spherical coordinates
lat = np.linspace(-59.995, 69.995, 13000)
lon = np.linspace(-179.995, 179.995, 36000)

# Convert to radians
lat_rad = np.radians(lat)
lon_rad = np.radians(lon)

# Create 2D mesh
lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)

# Convert to 3D Cartesian coordinates
x = np.cos(lat_grid) * np.cos(lon_grid)
y = np.cos(lat_grid) * np.sin(lon_grid)
z = np.sin(lat_grid)

# Convert lon and lat back to degrees for plotting
lon_deg = np.degrees(lon_rad)
lat_deg = np.degrees(lat_rad)

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 6))
cmap = matplotlib.colormaps['RdYlBu_r']

# List of data arrays and titles
data_list = [x, y, z]
titles = ['X Coordinate (cos(lat)×cos(lon))', 
          'Y Coordinate (cos(lat)×sin(lon))', 
          'Z Coordinate (sin(lat))']

# Create subplots
for i, (arr, title) in enumerate(zip(data_list, titles), 1):
    ax = fig.add_subplot(1, 3, i, projection=ccrs.PlateCarree())
    
    # Plot data
    mesh = ax.pcolormesh(lon_deg, lat_deg, arr,
                        transform=ccrs.PlateCarree(), 
                        cmap=cmap,
                        vmin=-1,  # Cartesian coordinates range from -1 to 1
                        vmax=1)
    
    # Add coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Set extent
    ax.set_extent([-180, 180, -60, 70], ccrs.PlateCarree())
    
    # Add colorbar for each subplot
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', 
                       pad=0.05, fraction=0.046, aspect=30)
    cbar.set_label(f'{"XYZ"[i-1]} value', fontsize=10)
    
    # Set title
    ax.set_title(title, fontsize=12)

plt.tight_layout()
Dir = '/my-projects2/1.project/NO2_DL_global/input_variables/Geographical_Variables_input/Spherical_Coordinates_2/'
out_png = os.path.join(Dir, 'Spherical_Coordinates_2.png')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Figure saved: {out_png}")