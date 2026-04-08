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
import matplotlib.ticker as tick
from matplotlib.colors import LinearSegmentedColormap

def average_year(dir, id, version, special_name, year):
    monthly_data = []
    for mon in range(1, 13):
        fpath = f'{dir}NO2_{version}_{year}{mon:02d}{special_name}{id}.nc'
        ds = xr.open_dataset(fpath, engine='netcdf4').squeeze()
        monthly_data.append(ds)

    data = xr.concat(monthly_data, dim='time').mean(dim='time')
    outpath = os.path.join(f'{dir}NO2_{version}_{year}{special_name}{id}_AnnualMean.nc')
    data.to_netcdf(outpath)

@lru_cache(maxsize=4)
def _land_regions(resolution="50m"):
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

def plot_map(fpath_in, fpath_out, extent=None, vmin=None, vmax=None):
    with xr.open_dataset(fpath_in, engine='netcdf4') as ds:
        lon = ds.lon.values
        lat = ds.lat.values
        NO2 = ds['NO2'].values
        
        # Preprocessing matching the requested style
        # NO2[np.where(NO2 < 0)] = 0
        # NO2 = np.nan_to_num(NO2, nan=5.0, posinf=3.0, neginf=2.0)
    
    # Get max NO2 value (ignoring NaN)
    no2_max = np.nanmax(NO2)
    
    cmap = plt.cm.get_cmap('plasma')
    cmap.set_bad(alpha=0)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Default to global extent if not specified
    if extent is None:
        extent = [-170, 180, -60, 70]
    print('extent:', extent)

    # Setup figure with equirectangular (PlateCarree) projection
    # Use central_longitude from the middle of the regional extent
    central_lon = (extent[0] + extent[1]) / 2.0
    proj = ccrs.PlateCarree(central_longitude=central_lon)

    # Compute figure size from extent so the figure matches the map's natural shape
    lon_range = extent[1] - extent[0]
    lat_range = extent[3] - extent[2]
    fig_width = 12
    fig_height = fig_width * (lat_range / lon_range) * 2
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Features matching the requested style
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeature.COASTLINE, linewidth=0.15)
    ax.add_feature(cfeature.LAKES, linewidth=0.1, facecolor='white')
    ax.add_feature(cfeature.BORDERS, linewidth=0.15)

    # Plot data
    pcm = plt.pcolormesh(lon, lat, NO2, transform=ccrs.PlateCarree(),
                         cmap=cmap, norm=norm)

    # Title / Annotation
    # Extract date/info from filename
    filename = os.path.basename(fpath_in)
    parts = filename.replace('.nc','').split('_')
    # Heuristic to find date part (e.g., 201901 or 2019)
    date_str = ""
    for p in parts:
        if p.isdigit() and len(p) >= 4:
            if len(p) == 6:
                date_str = f"{p[:4]} {p[4:]}"
            else:
                date_str = p
            break
            
    SPEC_NAME = r'NO$_{2}$'
    # Text position calculation matching the requested style
    # extent is [x0, x1, y0, y1] = [-180, 180, -60, 70]
    # x = extent[1] + 0.01 * width
    # y = extent[2] + 0.05 * height (Note: requested code had extent[2] as y0 in its own extent list logic, but here extent[2] is -60)
    # Using the extent values directly:
    x_pos = extent[1] + 0.01 * abs(extent[1] - extent[0])
    y_pos = extent[2] + 0.05 * abs(extent[3] - extent[2])
    
    # ax.text(x_pos, y_pos, '{} {}'.format(SPEC_NAME, date_str), style='italic', fontsize=10, fontweight='bold')

    # Colorbar — inset inside the map (bottom-right corner), short height
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="1.2%", height="35%", loc='lower left',
                     bbox_to_anchor=(0.1, 0.2, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(pcm, cax=cax, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(r'NO$_{2}$ (ppb)', size=15, labelpad=2)
    
    # Show a subset of tick labels so it's not too crowded
    # tick_step = max(1, n_levels // 6)
    # cbar.set_ticks(boundaries[::tick_step])
    # cbar.set_ticklabels([f'{b:.1f}' for b in boundaries[::tick_step]])
    
    # Display max value on top of colorbar
    cbar.ax.set_title(f'max\n{no2_max:.1f} ppb', fontsize=15, pad=5)

    plt.savefig(fpath_out, format='png', dpi=1000, transparent=True, bbox_inches='tight')
    plt.close()
    print(f'plot saved to {fpath_out}')

if __name__ == '__main__':
    year = 2023
    version = 'v4.1.0'
    special_name = '_Geov5133'
    base_dir = f'/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}'
    fig_dir = f'{base_dir}/Figures/figures-Estimation/'
    os.makedirs(fig_dir, exist_ok=True)

    # Two estimation types to plot
    estimation_types = [
        {
            'dir': f'{base_dir}/ForcedSlopeUnity_Map_Estimation/{year}/',
            'id': '_ForcedSlopeUnity',
            'label': 'ForcedSlopeUnity',
        },
        {
            'dir': f'{base_dir}/Map_Estimation/{year}/',
            'id': '',
            'label': 'Raw',
        },
    ]

    for est in estimation_types:
        est_dir = est['dir']
        id = est['id']
        label = est['label']

        print(f'\n=== Processing {label} estimation ===')
        print(f'  Directory: {est_dir}')

        # 1. Compute annual mean from 12 monthly files
        print(f'  Computing annual mean...')
        # average_year(est_dir, id, version, special_name, year)

        # 2. Plot annual mean
        fpath_in_year = f'{est_dir}NO2_{version}_{year}{special_name}{id}_AnnualMean.nc'
        fpath_out_year = f'{fig_dir}NO2_{version}{special_name}{id}_{year}_AnnualMean_map.png'
        plot_map(fpath_in_year, fpath_out_year, vmin=0, vmax=10)

        # 3. Plot January and July
        for mon in [1, 7]:
            fpath_in = f'{est_dir}NO2_{version}_{year}{mon:02d}{special_name}{id}.nc'
            fpath_out = f'{fig_dir}NO2_{version}{special_name}{id}_{year}{mon:02d}_map.png'
            plot_map(fpath_in, fpath_out, vmin=0, vmax=10)