import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import regionmask
import geopandas as gpd
from cartopy.io import shapereader as shpreader
from functools import lru_cache
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

@lru_cache(maxsize=4)
def _land_regions(resolution="10m"):
    """Cache land regions from Natural Earth data"""
    shp = shpreader.natural_earth(resolution=resolution, category="physical", name="land")
    geoms = list(shpreader.Reader(shp).geometries())
    return regionmask.Regions(outlines=geoms, names=None, numbers=None)

def _mask_ocean_from_coastline(lon_2d, lat_2d, resolution="10m"):
    """
    Mask ocean areas using high-resolution coastline data.
    Returns True for ocean, False for land.
    """
    land = _land_regions(resolution)
    mask = land.mask(lon_2d, lat_2d)
    ocean = np.isnan(mask)
    return ocean

def plot_map(fpath_in, fpath_out):
    with xr.open_dataset(fpath_in, engine='netcdf4') as ds:
        lon = ds.lon.values
        lat = ds.lat.values
        Distance = ds['Distance'].values
        print('Distance.min(), Distance.max():', Distance.min(), Distance.max())

        lon_2d, lat_2d = np.meshgrid(lon, lat)
        # Build a land mask on your lon/lat grid
        ocean_mask = _mask_ocean_from_coastline(lon_2d, lat_2d, resolution="10m")
        Distance_masked = np.where(ocean_mask, np.nan, Distance)
    
    # Yellow-Orange-Red colormap
    cmap = plt.cm.RdYlBu_r
    cmap.set_bad(color='none', alpha=0) 
    
    # Set up the figure with Cartopy projection
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines, state lines, and country borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='lightgray', zorder=0)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

    im = ax.pcolormesh(lon, lat, Distance_masked, cmap=cmap, 
                       transform=ccrs.PlateCarree(), 
                       vmin=0, vmax=1.5, zorder=1)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', 
                      alpha=0.5, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False    
    
    cbar_ax = inset_axes(ax, width="1%", height="25%", loc='lower left',
                         bbox_to_anchor=(0.02, 0.1, 1, 1),
                         bbox_transform=ax.transAxes, borderpad=0)    
    cb = plt.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both')
    
    # Set the title for the colorbar
    cb.set_label('Distance (km)', fontsize=15, fontweight='bold')
    
    # Set colorbar ticks
    cb.set_ticks(np.linspace(0, 5000, 3))
    cb.ax.tick_params(labelsize=15)
    
    plt.tight_layout()
    plt.savefig(fpath_out, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f'plot saved to {fpath_out}')

# Main execution
year = 2019
version = 'v0.1.1'
Dir = f'/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/Results/Pixels2sites_distances/{year}/'

fpath_in = f'{Dir}NO2_nearby_site_distances_forEachPixel_mean-mode_20Number_{year}Annual.nc'
fpath_out = f'{Dir}NO2_nearby_site_distances_forEachPixel_mean-mode_20Number_{year}Annual_map.png'

if os.path.exists(fpath_in):
    print(f"\nPlotting {id}...")
    plot_map(fpath_in, fpath_out)