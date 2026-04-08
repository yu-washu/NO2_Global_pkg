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

def average_season(dir, version, year, season_name, months):
    """
    Average data for a specific season
    
    Parameters:
    -----------
    dir : str
        Directory path
    version : str
        Version name
    year : int
        Year
    season_name : str
        Name of season (MAM, JJA, SON, DJF)
    months : list
        List of month numbers for the season
    """
    monthly_data = []
    for mon in months:
        fpath = f'{dir}rRMSE_Map_NO2_{version}_{year}{mon}_BenchMark.nc'
        if os.path.exists(fpath):
            ds = xr.open_dataset(fpath, engine='netcdf4').squeeze()
            monthly_data.append(ds)
        else:
            print(f"Warning: File not found: {fpath}")
    
    if len(monthly_data) == 0:
        print(f"No data found for season {season_name}")
        return None
    
    data = xr.concat(monthly_data, dim='time').mean(dim='time')
    outpath = os.path.join(f'{dir}rRMSE_Map_NO2_{version}_{year}{season_name}_v2.nc')
    data.to_netcdf(outpath)
    print(f"Saved seasonal average: {outpath}")
    return outpath

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
        
        # Get rRMSE variable (adjust variable name if needed)
        if 'rRMSE' in ds.variables:
            rRMSE = ds['rRMSE'].values
        elif 'NO2' in ds.variables:
            rRMSE = ds['NO2'].values
        else:
            var_name = list(ds.data_vars.keys())[0]
            rRMSE = ds[var_name].values
            print(f"Using variable: {var_name}")
        
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        # Build a land mask on your lon/lat grid
        ocean_mask = _mask_ocean_from_coastline(lon_2d, lat_2d, resolution="10m")
        rRMSE_masked = np.where(ocean_mask, np.nan, rRMSE)
    
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
    
    # Plot the data with colormap (vmin=0, vmax=0.8)
    im = ax.pcolormesh(lon, lat, rRMSE_masked, cmap=cmap, 
                       transform=ccrs.PlateCarree(), 
                       vmin=0.2, vmax=0.8, zorder=1)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', 
                      alpha=0.5, linestyle='--')
    # gl.top_labels = False
    # gl.right_labels = False    
    
    cbar_ax = inset_axes(ax, width="1%", height="30%", loc='lower left',
                         bbox_to_anchor=(0.02, 0.12, 1, 1),
                         bbox_transform=ax.transAxes, borderpad=0)
    cb = plt.colorbar(im, cax=cbar_ax, orientation='vertical', extend='both')
    
    # Set the title for the colorbar
    cb.set_label('NO₂ Uncertainty', fontsize=16, fontweight='bold')
    
    # Set colorbar ticks
    cb.set_ticks(np.linspace(0.2, 0.8, 3))
    cb.ax.tick_params(labelsize=16)
       
    plt.tight_layout()
    plt.savefig(fpath_out, dpi=500, bbox_inches='tight')
    plt.close(fig)
    print(f'plot saved to {fpath_out}')

# Main execution
year = 2019
version = 'LightGBM_1006'
Dir = f'/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/Uncertainty_Results/rRMSE_Map/{year}/'

# Define seasons
seasons = {
    'MAM': ['Mar', 'Apr', 'May'],      # March, April, May
    'JJA': ['Jun', 'Jul', 'Aug'],      # June, July, August
    'SON': ['Sep', 'Oct', 'Nov'],    # September, October, November
    'DJF': ['Dec', 'Jan', 'Feb']      # December, January, February
}

# First, calculate seasonal averages
print("Calculating seasonal averages...")
for season_name, months in seasons.items():
    print(f"\nProcessing season: {season_name}")
    # average_season(Dir, version, year, season_name, months)
    fpath_in = f'{Dir}rRMSE_Map_NO2_{version}_{year}{season_name}_v2.nc'
    fpath_out = f'{Dir}NO2_{version}_{year}{season_name}_rRMSE_map_v2.png'
    
    if os.path.exists(fpath_in):
        print(f"\nPlotting {id}...")
        plot_map(fpath_in, fpath_out)

# Then, plot all maps (including seasons)
print("\n\nPlotting maps...")

ids = ['Annual', 
       'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
       'MAM', 'JJA', 'SON', 'DJF']

for id in ids:
    fpath_in = f'{Dir}rRMSE_Map_NO2_{version}_{year}{id}_BenchMark.nc'
    fpath_out = f'{Dir}NO2_{version}_{year}{id}_rRMSE_map.png'
    
    if os.path.exists(fpath_in):
        print(f"\nPlotting {id}...")
        plot_map(fpath_in, fpath_out)
    else:
        print(f"\nSkipping {id} - file not found: {fpath_in}")

print("\n\nAll maps completed!")