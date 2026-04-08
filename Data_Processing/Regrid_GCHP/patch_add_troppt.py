#!/usr/bin/env python3
"""
Patch existing 01x01 tessellation files to add Met_TROPPT.
Skips all GCHP regridding — only loads TROPPT from GEOS-IT and appends it.

Usage:
    python patch_add_troppt.py --year 2019 --mon 1
    python patch_add_troppt.py --year 2019  # entire year
"""
import os
import argparse
import calendar
import numpy as np
import xarray as xr
import sparselt.esmf
import sparselt.xr

MET_ROOT    = '/ExtData/GEOS_C180/GEOS_IT'
WEIGHT_FILE = '/my-projects2/supportData/gridinfo/c180_to_1800x3600_weights.nc'
TESS_ROOT   = '/my-projects2/1.project/gchp/forTessellation'
LOCAL_HOURS = [13, 14, 15]
KEEP_DIMS   = {'lev', 'nf', 'Ydim', 'Xdim'}
DIM_ORDER   = ('lev', 'nf', 'Ydim', 'Xdim', 'time')

lon_01 = np.round(np.linspace(-179.995, 179.995, 3600), 5)
lat_01 = np.round(np.linspace( -89.995,  89.995, 1800), 5)


def load_transform():
    return sparselt.esmf.load_weights(
        WEIGHT_FILE,
        input_dims =[('nf', 'Ydim', 'Xdim'), (6, 180, 180)],
        output_dims=[('lat', 'lon'), (1800, 3600)],
    )


def patch_day(year, mon, day, transform):
    path01 = os.path.join(TESS_ROOT, str(year), 'daily',
                          f'01x01.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    if not os.path.exists(path01):
        print(f"  SKIP (not found): {path01}")
        return

    ds = xr.open_dataset(path01)
    if 'Met_TROPPT' in ds.data_vars:
        ds.close()
        print(f"  SKIP (already has Met_TROPPT): {year}{mon:02d}{day:02d}")
        return
    ds.close()

    # Load TROPPT from GEOS-IT
    fp_met = os.path.join(MET_ROOT, str(year), f'{mon:02d}',
                          f'GEOSIT.{year}{mon:02d}{day:02d}.A1.C180.nc')
    if not os.path.exists(fp_met):
        print(f"  SKIP (met not found): {fp_met}")
        return

    MET_KEEP  = KEEP_DIMS | {'time'}
    ds_met    = xr.open_dataset(fp_met).squeeze()
    ds_met_3h = ds_met.isel(time=LOCAL_HOURS)[['TROPPT']]
    ds_met_3h = ds_met_3h.drop_dims(
        [d for d in ds_met_3h.dims if d not in MET_KEEP], errors='ignore'
    )
    ds_met.close()
    avg3_met = ds_met_3h.mean('time').transpose(*DIM_ORDER, missing_dims='ignore')
    troppt_01 = sparselt.xr.apply(transform, avg3_met).assign_coords(lon=lon_01, lat=lat_01)

    # Append Met_TROPPT to existing file
    ds = xr.open_dataset(path01)
    ds['Met_TROPPT'] = troppt_01['TROPPT']
    tmp = path01 + '.tmp'
    ds.to_netcdf(tmp, encoding={v: {'zlib': True, 'complevel': 4} for v in ds.data_vars})
    ds.close()
    os.replace(tmp, path01)
    print(f"  patched: {year}{mon:02d}{day:02d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, default=None)
    parser.add_argument('--day',  type=int, default=None)
    args = parser.parse_args()

    print("Loading sparselt weights...")
    transform = load_transform()
    print("Weights loaded.")

    if args.day:
        # single day — fastest for array job submission
        patch_day(args.year, args.mon, args.day, transform)
    elif args.mon:
        ndays = calendar.monthrange(args.year, args.mon)[1]
        for day in range(1, ndays + 1):
            patch_day(args.year, args.mon, day, transform)
    else:
        for mon in range(1, 13):
            ndays = calendar.monthrange(args.year, mon)[1]
            for day in range(1, ndays + 1):
                patch_day(args.year, mon, day, transform)


if __name__ == '__main__':
    main()
