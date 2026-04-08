#!/usr/bin/env python3
import os
import argparse
import datetime
import numpy as np
import xarray as xr
import sparselt.esmf
import sparselt.xr
import xesmf as xe
import dask
import psutil
import gc


# — Configure dask for bounded memory usage —
dask.config.set({'array.chunk-size': '500MB'})

# — User settings & constants —
IN_ROOT        = '/fs2/yuanjian.z/archive/c180/2xneedleleaf_Regional_GFAS_0.85xDust_luo_2023_ceds202504_usaoc202106/OutputDir'
WEIGHT_FILE    = '/my-projects2/supportData/gridinfo/c180_to_1800x3600_weights.nc'
OUT_TESS_ROOT  = '/my-projects2/1.project/gchp/forTessellation'
OUT_GEO_ROOT   = '/my-projects2/1.project/gchp/forObservation-Geophysical'
OUT_CNN_ROOT   = '/my-projects2/1.project/NO2_DL_global/input_variables/GCHP_input'
LOCAL_HOURS    = [13, 14, 15]   # desired local solar hours (satellite overpass ~13:30 LST)
KEEP_DIMS      = {'lev','nf','Ydim','Xdim'}
DIM_ORDER      = ('lev','nf','Ydim','Xdim','time')
CRES_01        = '01x01'
NA             = 6.022e23    # molecules/mol
MwAir          = 28.97       # g/mol
# Chunking for optional fallback
CHUNK_LAT      = 1000        
CHUNK_LON      = 1000        

def print_memory_usage(stage=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{stage}] RSS = {mb:.1f} MB", flush=True)

# — Bilinear-interpolation helpers — 
def get_BilinearInterpolate_Index(fine_Lat, fine_Lon, fine_Lat_map, fine_Lon_map, coarse_Lat, coarse_Lon):
    """
    Compute floor/ceil indices and dx,dy offsets for bilinear interpolation
    from coarse grid (coarse_Lat, coarse_Lon) to fine grid (fine_Lat_map, fine_Lon_map).
    """
    delta_x = coarse_Lat[1] - coarse_Lat[0]
    delta_y = coarse_Lon[1] - coarse_Lon[0]
    min_x   = coarse_Lat[0]
    min_y   = coarse_Lon[0]

    lat_floor = np.floor((fine_Lat - min_x) / delta_x).astype(int)
    lat_ceil  = np.ceil ((fine_Lat - min_x) / delta_x).astype(int)
    lon_floor = np.floor((fine_Lon - min_y) / delta_y).astype(int)
    lon_ceil  = np.ceil ((fine_Lon - min_y) / delta_y).astype(int)

    nlat = len(fine_Lat)
    nlon = len(fine_Lon)
    dx = np.zeros((nlat, nlon), dtype=np.float32)
    dy = np.zeros((nlat, nlon), dtype=np.float32)

    for i in range(nlat):
        dy[i, :] = fine_Lon_map[i, :] - coarse_Lon[lon_floor]
    for j in range(nlon):
        dx[:, j] = fine_Lat_map[:, j] - coarse_Lat[lat_floor]

    return lat_floor, lat_ceil, lon_floor, lon_ceil, dx, dy, delta_x, delta_y

def bilinear_interp_vectorized(coarse, lat_floor, lat_ceil, lon_floor, lon_ceil, dx, dy, delta_x, delta_y):
    """
    Fully vectorized bilinear interpolation from coarse (1800x3600) to fine (18000x36000).
    All four corner lookups are done with numpy fancy indexing — no Python loop.
    """
    Cxfyf = coarse[lat_floor - 1][:, lon_floor - 1]
    Cxfyc = coarse[lat_floor - 1][:, lon_ceil  - 1]
    Cxcyf = coarse[lat_ceil  - 1][:, lon_floor - 1]
    Cxcyc = coarse[lat_ceil  - 1][:, lon_ceil  - 1]
    rx = dx / delta_x
    ry = dy / delta_y
    Cx1 = Cxcyf * rx + Cxfyf * (1 - rx)
    Cx2 = Cxcyc * rx + Cxfyc * (1 - rx)
    return (Cx2 * ry + Cx1 * (1 - ry)).astype(np.float32)

def init_worker():
    global transform
    global lon_01, lat_01, lon_km, lat_km
    global lat_floor, lat_ceil, lon_floor, lon_ceil, dx, dy, delta_x, delta_y

    print_memory_usage("init start")
    # load ESMF weights once
    transform = sparselt.esmf.load_weights(
        WEIGHT_FILE,
        input_dims =[('nf','Ydim','Xdim'), (6,180,180)],
        output_dims=[('lat','lon'), (1800,3600)]
    )
    # define coarse (0.1°) and fine (1 km) grids
    lon_01 = np.round(np.linspace(-179.995, 179.995, 3600), 5)
    lat_01 = np.round(np.linspace( -89.995,  89.995, 1800), 5)
    lon_km = np.round(np.linspace(-179.995, 179.995, 36000), 5)
    lat_km = np.round(np.linspace( -89.995,  89.995, 18000), 5)

    # 2D mesh of fine grid
    lon_km_2d, lat_km_2d = np.meshgrid(lon_km, lat_km)

    # precompute bilinear indices & offsets
    lat_floor, lat_ceil, lon_floor, lon_ceil, dx, dy, delta_x, delta_y = \
        get_BilinearInterpolate_Index(
            fine_Lat=lat_km,
            fine_Lon=lon_km,
            fine_Lat_map=lat_km_2d,
            fine_Lon_map=lon_km_2d,
            coarse_Lat=lat_01,
            coarse_Lon=lon_01
        )
    print_memory_usage("init done")

def process_day(year, mon, day):
    # prepare output dirs
    src_dir = os.path.join(IN_ROOT, str(year))
    tessD = os.path.join(OUT_TESS_ROOT, str(year),'daily'); os.makedirs(tessD, exist_ok=True)
    geoD  = os.path.join(OUT_GEO_ROOT,  str(year),'daily'); os.makedirs(geoD,  exist_ok=True)
    cnnD  = os.path.join(OUT_CNN_ROOT,  str(year)); os.makedirs(cnnD,  exist_ok=True)

    print_memory_usage("start process_day")

    # ── A) local-time 3-h avg → 0.1° ──
    # UTC offset per 0.1° longitude: round(lon/15), range -12..+12
    UTC_OFF = np.round(lon_01 / 15).astype(int)   # shape (3600,)

    # Raw UTC hours (no mod): 13-12=1 .. 15-(-12)=27
    # raw_h <= 23 → same day; raw_h >= 24 → next calendar day (raw_h - 24)
    needed_raw = sorted({lh - int(off)
                         for off in np.unique(UTC_OFF)
                         for lh in LOCAL_HOURS})

    base_date = datetime.date(year, mon, day)
    next_date = base_date + datetime.timedelta(days=1)

    # Stream one hour at a time: accumulate running sum, free each regridded dataset immediately.
    # This keeps peak memory ~1 regridded dataset instead of 27 simultaneously.
    utc_off_unique = np.unique(UTC_OFF)
    off_masks = {off: (UTC_OFF == off) for off in utc_off_unique}
    h_to_offsets = {raw_h: [off for off in utc_off_unique if (raw_h + int(off)) in LOCAL_HOURS]
                    for raw_h in needed_raw}

    local_sum  = None
    ref_dims   = None
    ref_coords = None

    for raw_h in needed_raw:
        offsets = h_to_offsets[raw_h]
        if not offsets:
            continue
        if raw_h <= 23:
            file_date, h = base_date, raw_h
        else:
            file_date, h = next_date, raw_h - 24
        fp = os.path.join(IN_ROOT, str(file_date.year),
                          f"GEOSChem.ACAGNO2Hourly."
                          f"{file_date.year}{file_date.month:02d}{file_date.day:02d}_{h:02d}00z.nc4")
        if not os.path.exists(fp):
            fp = os.path.join(src_dir,
                              f"GEOSChem.ACAGNO2Hourly.{year}{mon:02d}{day:02d}_2300z.nc4")
        ds = xr.open_dataset(fp).squeeze()
        ds = ds.drop_dims([d for d in ds.dims if d not in KEEP_DIMS], errors='ignore')
        for v in ds.data_vars:
            if ds[v].dtype == 'float64':
                ds[v] = ds[v].astype('float32')
        ds = ds.transpose(*DIM_ORDER, missing_dims='ignore')
        ds_regrid = sparselt.xr.apply(transform, ds).assign_coords(lon=lon_01, lat=lat_01)
        ds.close()
        if local_sum is None:
            local_sum  = {var: np.zeros(ds_regrid[var].shape, dtype=np.float32)
                          for var in ds_regrid.data_vars}
            ref_dims   = {var: ds_regrid[var].dims for var in ds_regrid.data_vars}
            ref_coords = ds_regrid.coords
        for var in ds_regrid.data_vars:
            arr = ds_regrid[var].values
            for off in offsets:
                local_sum[var][..., off_masks[off]] += arr[..., off_masks[off]]
        del ds_regrid
        gc.collect()
        print_memory_usage(f"after regrid hour {raw_h}")

    # Each longitude zone received exactly len(LOCAL_HOURS) contributions.
    local_avg = {var: (ref_dims[var], local_sum[var] / len(LOCAL_HOURS)) for var in local_sum}
    d01 = xr.Dataset(local_avg, coords=ref_coords)
    del local_sum
    gc.collect()
    print_memory_usage("after 0.1° regrid")

    # ── A.1) save four vars at 0.1° ──
    out01  = d01[['SpeciesConcVV_NO2','Met_PMIDDRY','Met_BXHEIGHT','Met_AIRDEN']]
    path01 = os.path.join(tessD, f'{CRES_01}.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    out01.to_netcdf(path01, encoding={v:{'zlib':True,'complevel':4} for v in out01.data_vars})
    print_memory_usage("after save 0.1°")

    # ── B) compute NO2col @0.1° & vectorized bilinear regrid to 1 km ──
    AirDen = d01['Met_AIRDEN'] * 1e3
    BoxH   = d01['Met_BXHEIGHT']
    conc   = d01['SpeciesConcVV_NO2'] * AirDen / MwAir
    no2col = (conc * BoxH).sum('lev') * 1e-4 * NA
    no2col.name = 'NO2col'

    coarse   = no2col.values.astype(np.float32)
    fine_NO2 = bilinear_interp_vectorized(
        coarse, lat_floor, lat_ceil, lon_floor, lon_ceil, dx, dy, delta_x, delta_y
    )
    da_col  = xr.DataArray(fine_NO2, coords={'lat':lat_km,'lon':lon_km}, dims=['lat','lon'], name='NO2col')
    path1km = os.path.join(geoD, f'1x1km.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    da_col.to_netcdf(path1km, encoding={'NO2col':{'zlib':True,'complevel':4}})
    print_memory_usage("after save 1km NO2col")

    # ── C) daily geophysical regrid: 0.1° then 1 km ──
    fpD  = os.path.join(src_dir, f'GEOSChem.ACAGGasDaily.{year}{mon:02d}{day:02d}_0000z.nc4')
    dD   = xr.open_dataset(fpD).squeeze()
    dD   = dD.drop_dims([d for d in dD.dims if d not in KEEP_DIMS], errors='ignore')
    d01D = sparselt.xr.apply(transform, dD).assign_coords(lon=lon_01, lat=lat_01)
    print_memory_usage("after 0.1° geophy")
    # build surface-level variables
    geo = xr.Dataset({
        'gchp_NO2':           d01D['SpeciesConcVV_NO2'].isel(lev=0)*1e9,
        'gchp_PAN':           d01D['SpeciesConcVV_PAN'].isel(lev=0)*1e9,
        'gchp_HNO3':          d01D['SpeciesConcVV_HNO3'].isel(lev=0)*1e9,
        'gchp_alkylnitrates': (d01D['SpeciesConcVV_BUTN']+d01D['SpeciesConcVV_NPRNO3']).isel(lev=0)*1e9,
        'gchp_NH3':           d01D['SpeciesConcVV_NH3'].isel(lev=0)*1e9,
        'gchp_O3':            d01D['SpeciesConcVV_O3'].isel(lev=0)*1e9,
        'gchp_OH':            d01D['SpeciesConcVV_OH'].isel(lev=0)*1e9,
        'gchp_NO':            d01D['SpeciesConcVV_NO'].isel(lev=0)*1e9,
        'gchp_NO3':           d01D['SpeciesConcVV_NO3'].isel(lev=0)*1e9,
        'gchp_N2O5':          d01D['SpeciesConcVV_N2O5'].isel(lev=0)*1e9,
        'gchp_HO2':           d01D['SpeciesConcVV_HO2'].isel(lev=0)*1e9,
        'gchp_H2O2':          d01D['SpeciesConcVV_H2O2'].isel(lev=0)*1e9,
        'gchp_CO':            d01D['SpeciesConcVV_CO'].isel(lev=0)*1e9,
    })
    np_vars = ['gchp_NH3','gchp_O3','gchp_OH','gchp_NO',
               'gchp_NO3','gchp_N2O5','gchp_HO2','gchp_H2O2','gchp_CO']
    others = {}
    nlat_fine = len(lat_km)
    nlon_fine = len(lon_km)
    for var in geo.data_vars:
        coarse2D = geo[var].values.astype(np.float32)
        fine2D   = bilinear_interp_vectorized(
            coarse2D, lat_floor, lat_ceil, lon_floor, lon_ceil, dx, dy, delta_x, delta_y
        )
        if var in np_vars:
            fname = f"{var}_001x001_Global_map_{year}{mon:02d}{day:02d}.npy"
            # np.save(os.path.join(cnnD, fname), fine2D)
        else:
            others[var] = (['lat','lon'], fine2D)
        del coarse2D, fine2D
        gc.collect()
    if others:
        ds_geo = xr.Dataset(others, coords={'lat': lat_km, 'lon': lon_km})
        gpath  = os.path.join(geoD, f'1x1km.DailyVars.{year}{mon:02d}{day:02d}.nc4')
        ds_geo.to_netcdf(gpath, encoding={v:{'zlib':True,'complevel':4} for v in ds_geo.data_vars})
    print_memory_usage("after daily geophy interpolation")

# ── D) plotting saved .npy fields ──
def plot_multiple_geophy(YEAR, MONTH, DAY, vars_to_plot):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    base_dir = os.path.join(OUT_CNN_ROOT, str(YEAR))
    lon = lon_km
    lat = lat_km

    n = len(vars_to_plot)
    if n <= 3:
        nrows, ncols = 1, n
    else:
        nrows, ncols = min(3, n), min(3, n)

    fig = plt.figure(figsize=(ncols*4, nrows*3))
    cmap = plt.cm.get_cmap('RdYlBu_r')

    for i, var in enumerate(vars_to_plot):
        ax  = fig.add_subplot(nrows, ncols, i+1, projection=ccrs.PlateCarree())
        fn  = f"{var}_001x001_Global_map_{YEAR}{MONTH}{DAY}.npy"
        arr = np.load(os.path.join(base_dir, fn))
        mesh = ax.pcolormesh(lon, lat, arr,
                             transform=ccrs.PlateCarree(), cmap=cmap,
                             vmin=0, vmax=np.nanmax(arr)*0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent([-180,180,-70,70], ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
        cbar.set_label(var)
        ax.set_title(var, fontsize=10)

    plt.suptitle(f'Geophysical Fields {YEAR}-{MONTH}-{DAY}', fontsize=14, y=0.98)
    plt.tight_layout()
    out_png = os.path.join(OUT_CNN_ROOT, str(YEAR), f'Geophy_{YEAR}_{MONTH}_{DAY}.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, required=True)
    parser.add_argument('--day',  type=int, required=True)
    args = parser.parse_args()

    init_worker()
    process_day(args.year, args.mon, args.day)
    # plot_multiple_geophy(args.year, f"{args.mon:02d}", f"{args.day:02d}", 
    #                      vars_to_plot = ['gchp_NH3','gchp_O3','gchp_OH'])
