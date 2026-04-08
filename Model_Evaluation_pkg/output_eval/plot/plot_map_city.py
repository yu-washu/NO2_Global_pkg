import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# %matplotlib inline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from cartopy.io import shapereader as shpreader
import regionmask
import matplotlib.ticker as tick
from functools import lru_cache

# ---------------- Config ----------------
year=2023
version = 'v2'
special_name = '_cf_v6_filtered'
nc_path = (
    f"/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/ForcedSlopeUnity_Map_Estimation/{year}/NO2_{version}_{year}{special_name}_ForcedSlopeUnity_AnnualMean.nc"
)

cities = {
    "Paris, France":       {"coords": (2.3522, 48.8566),   "zoom": 0.5},
    "Chicago, US":         {"coords": (-87.6298, 41.8781), "zoom": 0.6},
    "New York, US":        {"coords": (-74.0060, 40.7128), "zoom": 0.8},
    "Los Angeles, US":     {"coords": (-118.2437, 34.0522),"zoom": 0.6},
    "Dallas, US":          {"coords": (-96.7970, 32.7767), "zoom": 0.6},
    "Mexico City, Mexico": {"coords": (-99.1332, 19.4326), "zoom": 0.8},
    "Santiago, Chile":     {"coords": (-70.6693, -33.4489),"zoom": 0.8},
    "Delhi, India":        {"coords": (77.1025, 28.7041),  "zoom": 0.8},
    "Beijing, China":      {"coords": (116.4074, 39.9042), "zoom": 0.8},
    "Johannesburg, South Africa": {"coords": (28.0473, -26), "zoom": 0.8},
    "London, UK":          {"coords": (-0.1276, 51.5074),  "zoom": 1.0},
    "Melbourne, Australia": {"coords": (144.9631, -37.8136), "zoom": 0.6},
    "Sydney, Australia":   {"coords": (151.2093, -33.8688), "zoom": 0.6},
    "Auckland, New Zealand": {"coords": (174.7633, -36.8485), "zoom": 0.5},
    "Yangtze River Delta, China": {"coords": (120.4737, 31.2304), "zoom": 2.0},  # Centered on Shanghai
    "São Paulo, Brazil":   {"coords": (-46.6333, -23.5505), "zoom": 0.8},
    "Amsterdam, Netherlands": {"coords": (4.897070, 52.377956), "zoom": 0.8},
    "Antwerp, Belgium":    {"coords": (4.4025, 51.2194),   "zoom": 0.7},
    "Brussels, Belgium":   {"coords": (4.3517, 50.8503),   "zoom": 0.4},
    "Tokyo, Japan":        {"coords": (139.6917, 35.6762), "zoom": 0.8},
    "Cairo, Egypt":        {"coords": (31.2357, 30.0444),  "zoom": 0.8},
    "Tehran, Iran":        {"coords": (51.3890, 35.6892),  "zoom": 0.6},
    "Dubai, UAE":          {"coords": (55.2708, 25.2048),  "zoom": 0.8},
    "Milan, Italy":        {"coords": (9.1900, 45.4642),   "zoom": 0.8},
}
countris= {
    "Italy": {"coords": (12.5674, 41.8719), "zoom": 6},
    "Europe": {"coords": (5, 50), "zoom": 20},
    "United States West": {"coords": (-115, 37), "zoom": 15},
    "United States East": {"coords": (-82, 37), "zoom": 15},
    "United Kingdom": {"coords": (-3.5360, 55.0781), "zoom": 6},
    "Australia": {"coords": (133.7751, -25.2744), "zoom": 15},
    "New Zealand": {"coords": (172, -41), "zoom": 7},
    "China": {"coords": (115, 35.8617), "zoom": 20},
    "India": {"coords": (80, 20), "zoom": 15},
    "South Africa": {"coords": (24.5, -30), "zoom": 8},
    "Egypt": {"coords": (31.2357, 30.0444), "zoom": 15},
}

cmap = plt.cm.plasma

city_vlims = {
    "Los Angeles, US":            (0, 20),
    "New York, US":               (0, 15),
    "Delhi, India":               (8, 16),
    "Yangtze River Delta, China": (0, 17),
    "Milan, Italy":               (0, 15),
}

# ---------------- Helpers ----------------
def normalize_city_lon_to_ds(city_lon, ds_lon0):
    if ds_lon0 >= 0 and city_lon < 0:
        return city_lon + 360.0
    if ds_lon0 < 0 and city_lon > 180:
        return city_lon - 360.0
    return city_lon

def slice_with_order(coord, vmin, vmax):
    asc = bool(coord[1] > coord[0])
    return slice(vmin, vmax) if asc else slice(vmax, vmin)

def draw_scalebar(ax,
                total_km=20,          # shows 0–20 km
                tick_km=10,           # middle tick (10 km)
                pad_x_frac=0.03,      # left margin (fraction of map width)
                pad_y_frac=0.04,      # bottom margin (fraction of map height)
                color="black",      # subtle gray
                lw_main=1.6,
                lw_tick=1.2,
                tick_h_km=1.5,        # tick height in km
                font_size=8):
    """
    Mexico City style scalebar:
    - thin gray baseline with ticks at 0, 10, 20
    - labels '0  10  20 km' UNDER the bar
    - anchored bottom-left with small padding
    """
    geod = Geod(ellps="WGS84")
    xmin, xmax, ymin, ymax = ax.get_extent(crs=ccrs.PlateCarree())

    # anchor near bottom-left
    y = ymin + pad_y_frac * (ymax - ymin)
    x0 = xmin + pad_x_frac * (xmax - xmin)

    # end points along an east–west geodesic
    L  = total_km * 1000.0
    Lm = tick_km  * 1000.0
    x10, y10, _ = geod.fwd(x0, y,  90, Lm)  # 10 km east
    x20, y20, _ = geod.fwd(x0, y,  90, L)   # 20 km east

    # main baseline 0–20 km
    ax.plot([x0, x20], [y, y], transform=ccrs.PlateCarree(),
            color=color, lw=lw_main)

    # ticks at 0, 10, 20 (short verticals)
    def vtick(xlon, xlat):
        xu, yu, _ = geod.fwd(xlon, xlat,   0, tick_h_km*1000.0)   # north
        xd, yd, _ = geod.fwd(xlon, xlat, 180, tick_h_km*1000.0)   # south
        ax.plot([xlon, xlon], [yd, yu], transform=ccrs.PlateCarree(),
                color=color, lw=lw_tick)

    vtick(x0,  y)
    vtick(x10, y)
    vtick(x20, y)

    # small lower "bracket" stub at the left (like your reference)
    # draws a short line just below the baseline
    _, y_low, _ = geod.fwd(x0, y, 180, tick_h_km*1000.0)  # a bit below
    x_stub, y_stub, _ = geod.fwd(x0, y_low, 90, 6000.0)   # ~6 km stub
    # ax.plot([x0, x_stub], [y_low, y_low], transform=ccrs.PlateCarree(),
    #         color=color, lw=lw_tick)

    # labels UNDER the bar
    ax.text(x0,  y-0.02, "0", transform=ccrs.PlateCarree(),
            va="top", ha="center", fontsize=font_size, color=color)
    ax.text(x10, y-0.02, f"{tick_km}", transform=ccrs.PlateCarree(),
            va="top", ha="center", fontsize=font_size, color=color)
    ax.text(x20, y-0.02, f"{total_km} km", transform=ccrs.PlateCarree(),
            va="top", ha="center", fontsize=font_size, color=color)

def load_city_window(nc_path, lon0, lat0, half_deg):
    """Open file without Dask, slice to a small lon/lat box, return lon, lat, NO2 arrays and extent."""
    with xr.open_dataset(nc_path, engine="netcdf4") as ds:
        ds_lon0 = float(ds.lon.isel(lon=0))
        lon_c = normalize_city_lon_to_ds(lon0, ds_lon0)

        lon_min, lon_max = lon_c - half_deg, lon_c + half_deg
        lat_min, lat_max = lat0    - half_deg, lat0    + half_deg

        lon_slice = slice_with_order(ds.lon.values, lon_min, lon_max)
        lat_slice = slice_with_order(ds.lat.values, lat_min, lat_max)

        # Select the small window; this reads ONLY that subset from disk
        ds_sub = ds.sel(lon=lon_slice, lat=lat_slice)

        lon_sub = ds_sub.lon.values
        lat_sub = ds_sub.lat.values
        NO2_sub = ds_sub["NO2"].values  # safe: it's small

    extent = [lon_min, lon_max, lat_min, lat_max]
    return lon_sub, lat_sub, NO2_sub, extent, lon_c

@lru_cache(maxsize=4)
def _land_regions(resolution="10m"):
    shp = shpreader.natural_earth(resolution=resolution, category="physical", name="land")
    geoms = list(shpreader.Reader(shp).geometries())  # MultiPolygons/Polygons
    # Build a regionmask Regions object from coastline-based land polygons
    return regionmask.Regions(outlines=geoms, names=None, numbers=None)

def _mask_ocean_from_coastline(lon_2d, lat_2d, resolution="10m"):
    land = _land_regions(resolution)
    # mask has integers (0,1,...) over land polygons and NaN over ocean
    mask = land.mask(lon_2d, lat_2d)
    ocean = np.isnan(mask)
    return ocean  # True on ocean, False on land

def plot_city_map(city_name, lon_sub, lat_sub, NO2_sub, extent, lon_center):
    vmin, vmax = city_vlims.get(city_name, (0, 8))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(6.2, 5.8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 2D lon/lat grid
    lon_2d, lat_2d = np.meshgrid(lon_sub, lat_sub)
    NO2_sub[np.where(NO2_sub < 0)] = 0
    NO2_sub = np.nan_to_num(NO2_sub, nan=5.0, posinf=3.0, neginf=2.0)

    im = ax.pcolormesh(lon_sub, lat_sub, NO2_sub, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree(), rasterized=True)

    # colorbar
    cax = fig.add_axes([0.06, -0.01, 0.88, 0.03])
    cb = plt.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    mid = (vmin + vmax) / 2
    cb.set_ticks([vmin, mid, vmax])
    cb.ax.tick_params(labelsize=36)

    # Draw borders & coastline for reference
    borders = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_0_boundary_lines_land',
        scale='10m', facecolor='none'
    )
    # ax.add_feature(borders, linewidth=1.5, edgecolor='black', zorder=200)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0, edgecolor='black', zorder=200)
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeature.LAKES, linewidth=0.1, facecolor='white', edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0)
    
    # draw_scalebar(ax, total_km=100, tick_km=50)
    plt.tight_layout()
    FigDir = f"/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/ForcedSlopeUnity_Map_Estimation/{year}/"
    fig.savefig(f'{FigDir}{city_name}.png', dpi=500, bbox_inches='tight')
    plt.show()

# ---------------- Use it (one city at a time) ----------------
selected_cities = [
    "Los Angeles, US",
    "New York, US",
    "Milan, Italy",
    "Yangtze River Delta, China",
    "Delhi, India",
]
for city in selected_cities:
    lon0, lat0 = cities[city]["coords"]
    half_deg = cities[city]["zoom"]

    lon_sub, lat_sub, NO2_sub, extent, lon_center = load_city_window(nc_path, lon0, lat0, half_deg)
    plot_city_map(city, lon_sub, lat_sub, NO2_sub, extent, lon_center)