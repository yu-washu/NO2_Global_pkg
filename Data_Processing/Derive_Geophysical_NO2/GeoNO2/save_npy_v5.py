#!/usr/bin/env python3
"""
save_npy_v5.py — extract filled variables from geono2_v5 monthly NetCDF files
and save as numpy arrays.

Variables extracted (nc var → npy name):
  filled_GeoNO2_tot          → GeoNO2_tot
  filled_GeoNO2_trop         → GeoNO2_trop
  filled_SatColNO2_trop_gcshape → SatColNO2_gcshape
  filled_SatColNO2_trop      → SatColNO2

Input:  /my-projects2/1.project/GeoNO2-v5/{instrument}/{year}/
            1x1km.GeoNO2.{instrument}.{year}{month:02d}.MonMean.nc
Output: /my-projects2/1.project/NO2_DL_global/input_variables/GeoNO2-v5_input/{instrument}/{year}/
            {var_name}_001x001_Global_map_{year}{month:02d}.npy
"""
import os
import sys
import argparse
import glob
import numpy as np
import xarray as xr
from datetime import datetime

# nc variable → output npy stem name
VARS = {
    'filled_GeoNO2_tot':          'GeoNO2_tot',
    'filled_GeoNO2_trop':         'GeoNO2_trop',
    'filled_SatColNO2_trop_gcshape': 'SatColNO2_gcshape',
    'filled_SatColNO2_trop':      'SatColNO2',
}


def get_paths(instrument, year):
    input_dir  = f'/my-projects2/1.project/GeoNO2-v5/{instrument}/{year}/'
    output_dir = (f'/my-projects2/1.project/NO2_DL_global/input_variables'
                  f'/GeoNO2-v5_input/{instrument}/{year}/')
    return input_dir, output_dir


def extract_month(year, month, instrument):
    input_dir, output_dir = get_paths(instrument, year)
    nc_file = os.path.join(input_dir,
                           f'1x1km.GeoNO2.{instrument}.{year}{month:02d}.MonMean.nc')

    if not os.path.exists(nc_file):
        print(f'  ✗ Not found: {nc_file}')
        return False

    print(f'  Processing {year}-{month:02d} [{instrument}]...', end=' ', flush=True)

    try:
        os.makedirs(output_dir, exist_ok=True)

        with xr.open_dataset(nc_file, engine='netcdf4') as ds:
            nc_size = os.path.getsize(nc_file) / 1024**2
            results = {}
            for nc_var, out_name in VARS.items():
                if nc_var not in ds:
                    print(f'\n    [WARN] {nc_var} not in file, skipping')
                    continue
                results[out_name] = ds[nc_var].values.astype('float32')

        for out_name, arr in results.items():
            npy_file = os.path.join(output_dir,
                                    f'{out_name}_001x001_Global_map_{year}{month:02d}.npy')
            np.save(npy_file, arr)

        # report sizes for first saved file as proxy
        sample_npy = os.path.join(output_dir,
            f'{list(results.keys())[0]}_001x001_Global_map_{year}{month:02d}.npy')
        npy_size = os.path.getsize(sample_npy) / 1024**2
        print(f'✓  ({nc_size:.0f} MB nc → {npy_size:.0f} MB/var, {len(results)} vars)')

        for out_name, arr in results.items():
            valid = int(np.sum(~np.isnan(arr)))
            total = arr.size
            print(f'    {out_name}: shape={arr.shape}  valid={valid:,}/{total:,} '
                  f'({100*valid/total:.1f}%)  '
                  f'range=[{np.nanmin(arr):.2e}, {np.nanmax(arr):.2e}]')
        return True

    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback; traceback.print_exc()
        return False


def find_available(instrument, year):
    input_dir, _ = get_paths(instrument, year)
    pattern = os.path.join(input_dir, f'1x1km.GeoNO2.{instrument}.{year}??.MonMean.nc')
    files = sorted(glob.glob(pattern))
    months = []
    print(f'\n=== Available files for {instrument} {year} ===')
    if not files:
        print(f'  None found in {input_dir}')
        return months
    for f in files:
        try:
            ym = os.path.basename(f).split('.')[3]   # e.g. 202301
            m  = int(ym[-2:])
            months.append(m)
            print(f'  {m:02d}: {os.path.basename(f)}  ({os.path.getsize(f)/1024**2:.0f} MB)')
        except (IndexError, ValueError):
            print(f'  ?: {os.path.basename(f)}')
    return months


def main():
    parser = argparse.ArgumentParser(
        description='Save geono2_v5 filled variables as npy arrays')
    parser.add_argument('year',  type=int)
    parser.add_argument('--month', type=int, choices=range(1, 13))
    parser.add_argument('--instrument', choices=['TROPOMI', 'OMI', 'both'],
                        default='TROPOMI')
    args = parser.parse_args()

    instruments = ['TROPOMI', 'OMI'] if args.instrument == 'both' else [args.instrument]

    print(f'save_npy_v5  year={args.year}  instrument={args.instrument}')
    print(f'Start: {datetime.now():%Y-%m-%d %H:%M:%S}')

    total_ok = 0
    for inst in instruments:
        available = find_available(inst, args.year)
        if not available:
            continue

        months = [args.month] if args.month else list(range(1, 13))
        ok = 0
        print(f'\n=== Extracting {inst} {args.year} ===')
        for m in months:
            if m not in available:
                print(f'  {m:02d}: not available, skipping')
                continue
            if extract_month(args.year, m, inst):
                ok += 1
        print(f'  {ok}/{len(months)} months extracted for {inst}')
        total_ok += ok

    print(f'\nDone. {datetime.now():%Y-%m-%d %H:%M:%S}')
    sys.exit(0 if total_ok > 0 else 1)


if __name__ == '__main__':
    main()
