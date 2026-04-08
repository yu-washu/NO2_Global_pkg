#!/usr/bin/env python3
"""
regen_NO2col.py — Regenerate 1 km NO2col_tot + NO2col_trop from existing 0.1° tess files.

Reads:  OUT_TESS_ROOT/{year}/daily/01x01.Hours.13-15.{date}.nc4
Writes: OUT_GEO_ROOT/{year}/daily/1x1km.Hours.13-15.{date}.nc4
          → variables: NO2col_tot, NO2col_trop

Use this when the 0.1° tess files are already present and you only need to
recompute/overwrite the combined 1 km column file (e.g. after changing the
save format to bundle both columns in one file).
"""
import os
import argparse
import numpy as np
import xarray as xr
import psutil
import gc

# ── paths (mirror main_nearest_neighbour.py) ──
OUT_TESS_ROOT = '/my-projects2/1.project/gchp/forTessellation'
OUT_GEO_ROOT  = '/my-projects2/1.project/gchp/forObservation-Geophysical'

# ── grid constants ──
NA     = 6.022e23   # molecules/mol
MwAir  = 28.97      # g/mol
lon_km = np.round(np.linspace(-179.995, 179.995, 36000), 5)
lat_km = np.round(np.linspace( -89.995,  89.995, 18000), 5)
lon_01 = np.round(np.linspace(-179.995, 179.995,  3600), 5)
lat_01 = np.round(np.linspace( -89.995,  89.995,  1800), 5)

# Nearest-neighbour index: each fine point → coarse cell
_delta_lat = lat_01[1] - lat_01[0]
_delta_lon = lon_01[1] - lon_01[0]
lat_floor = np.clip(
    np.floor((lat_km - lat_01[0]) / _delta_lat).astype(int) + 1, 1, len(lat_01)
)
lon_floor = np.clip(
    np.floor((lon_km - lon_01[0]) / _delta_lon).astype(int) + 1, 1, len(lon_01)
)


def print_memory_usage(stage=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{stage}] RSS = {mb:.1f} MB", flush=True)


def process_day(year, mon, day):
    tessD = os.path.join(OUT_TESS_ROOT, str(year), 'daily')
    geoD  = os.path.join(OUT_GEO_ROOT,  str(year), 'daily')
    os.makedirs(geoD, exist_ok=True)

    path_tess = os.path.join(tessD, f'01x01.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    if not os.path.exists(path_tess):
        print(f"Missing tess file: {path_tess}", flush=True)
        return

    print_memory_usage("start")

    with xr.open_dataset(path_tess) as ds:
        AirDen = ds['Met_AIRDEN'] * 1e3
        BoxH   = ds['Met_BXHEIGHT']
        conc   = ds['SpeciesConcVV_NO2'] * AirDen / MwAir

        # Total column
        no2col      = (conc * BoxH).sum('lev') * 1e-4 * NA
        coarse      = no2col.values.astype(np.float32)
        fine_NO2    = coarse[lat_floor - 1][:, lon_floor - 1]

        # Tropospheric column
        trop_mask   = ds['Met_PMIDDRY'] >= ds['Met_TROPPT']
        no2col_trop = (conc * BoxH).where(trop_mask).sum('lev') * 1e-4 * NA
        coarse_trop = no2col_trop.values.astype(np.float32)
        fine_trop   = coarse_trop[lat_floor - 1][:, lon_floor - 1]

    gc.collect()
    print_memory_usage("after column computation")

    ds_col = xr.Dataset(
        {'NO2col_tot':  (['lat', 'lon'], fine_NO2),
         'NO2col_trop': (['lat', 'lon'], fine_trop)},
        coords={'lat': lat_km, 'lon': lon_km}
    )
    path1km = os.path.join(geoD, f'1x1km.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    ds_col.to_netcdf(path1km, encoding={v: {'zlib': True, 'complevel': 4} for v in ds_col.data_vars})
    print_memory_usage(f"saved {path1km}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, required=True)
    parser.add_argument('--day',  type=int, required=True)
    args = parser.parse_args()
    process_day(args.year, args.mon, args.day)
