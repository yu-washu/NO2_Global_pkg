#!/usr/bin/env python3
import os
import sys
import shutil
import tempfile
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import psutil
from Tess_func import (
    load_and_save_to_nc
)

def print_status(stage):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[{stage}] - Memory: {mem:.2f} GB")

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# spatial grid
max_lat, min_lat, max_lon, min_lon = 70, -60, 180, -180
latres, lonres = 0.01, 0.01
out_lat_edges = np.arange(min_lat, max_lat + latres, latres)
out_lon_edges = np.arange(min_lon, max_lon + lonres, lonres)
tlat = out_lat_edges[:-1] + latres / 2
tlon = out_lon_edges[:-1] + lonres / 2

# QC thresholds
CloudFraction_max, sza_max, QAlim = 0.1, 75, 0.75
qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
    int(CloudFraction_max * 100),
    sza_max,
    int(QAlim * 100)
)

# directories
tess_temp_dir      = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/temp/'
out_dir            = '/my-projects2/1.project/NO2_col/TROPOMI/daily/'

os.makedirs(tess_temp_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

tess_var_str = [
    'tropomi_NO2_total',
    'tropomi_NO2_gcshape_total'
]

# ─── PER-DAY WORKER ──────────────────────────────────────────────────────────────
def process_single_day(year, month, day):
    label = f"{year:04d}{month:02d}{day:02d}"
    scratch = f"{tess_temp_dir}tess_{label}"

    flabel   = f"{label}_{qcstr}"
    tess_in  = os.path.join(scratch, f"tessellate_input_{flabel}.dat")
    tess_out = os.path.join(scratch, f"tessellate_output_{flabel}.dat")
    setup_f  = os.path.join(scratch, f"tessellate_setup_{flabel}.dat")
    nc_out   = os.path.join(out_dir,  f"Tropomi_Regrid_{flabel}.nc")


    load_and_save_to_nc(tess_out, nc_out, len(tess_var_str), tlat, tlon)
    print_status("Daily Tessellation saved")
    


# ─── DISPATCH ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, required=True)
    parser.add_argument('--day',  type=int, required=True)
    args = parser.parse_args()

    process_single_day(args.year, args.mon, args.day)
