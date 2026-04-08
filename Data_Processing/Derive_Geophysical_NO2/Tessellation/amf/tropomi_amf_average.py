#!/usr/bin/env python3
"""
Monthly average of TROPOMI AMF_tot and AMF_gcshape from daily tessellated files
produced by tess_TROPOMI_AMFtot.py.

Saves a compressed NetCDF and produces a 2-panel map.

Usage:
    python tropomi_amf_average.py <year> --month <1-12>               # monthly average
    python tropomi_amf_average.py <year> --yearly-only                # yearly average from monthlies
    python tropomi_amf_average.py <year> --month 7 --no-plot          # skip plotting
    python tropomi_amf_average.py <year> --month 7 --plot-only        # only plot existing monthly .nc (compute if missing)
    python tropomi_amf_average.py <year> --yearly-only --plot-only    # only plot existing yearly .nc (compute if missing)
"""
import os
import sys
import gc
import time
import calendar
import argparse
import warnings
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import psutil

# ─── CONFIG ──────────────────────────────────────────────────────────────────
sza_max, QAlim = 80, 0.75
qcstr = 'SZA{}-QA{}'.format(sza_max, int(QAlim * 100))

# ─── MEMORY HELPERS ──────────────────────────────────────────────────────────
def check_memory():
    mem = psutil.virtual_memory()
    return {'used_gb': mem.used / 1e9, 'available_gb': mem.available / 1e9,
            'percent': mem.percent}

def log_memory(step):
    m = check_memory()
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"  [{ts}] {step}: {m['used_gb']:.1f} GB used, "
          f"{m['available_gb']:.1f} GB free ({m['percent']:.1f}%)", flush=True)

def force_cleanup():
    gc.collect()
    time.sleep(0.05)

# ─── DAILY → MONTHLY ─────────────────────────────────────────────────────────
def average_daily_to_monthly(year, month):
    base_dir    = f'/my-projects2/1.project/NO2col-v2/TROPOMI/{year}'
    daily_dir   = os.path.join(base_dir, 'daily_AMFtot')
    monthly_dir = os.path.join(base_dir, 'monthly_AMFtot')
    os.makedirs(monthly_dir, exist_ok=True)

    days_in_month = calendar.monthrange(year, month)[1]
    print(f"Processing {year}-{month:02d} ({days_in_month} days)", flush=True)
    print(f"PID: {os.getpid()}  Start: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    log_memory("Initial")

    VARS = ['amf_trop', 'amf_trop_gcshape', 'amf_tot', 'amf_tot_gcshape']

    # Running sum/count accumulators — allocated on first valid day
    acc_sum   = {}  # var -> float64 array
    acc_count = {}  # var -> int32 array
    ref_lat = ref_lon = None
    valid_days = 0

    for day in range(1, days_in_month + 1):
        fname = f"Tropomi_AMFtot_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc"
        fpath = os.path.join(daily_dir, fname)

        if not os.path.exists(fpath):
            print(f"  [WARN] Missing: {fname}", flush=True)
            continue

        try:
            # Open without dask to avoid building a massive task graph
            with xr.open_dataset(fpath) as ds:
                if ref_lat is None:
                    ref_lat = ds['lat'].values.copy()
                    ref_lon = ds['lon'].values.copy()
                    for v in VARS:
                        shape = ds[v].shape  # (lat, lon)
                        acc_sum[v]   = np.zeros(shape, dtype=np.float64)
                        acc_count[v] = np.zeros(shape, dtype=np.int32)

                for v in VARS:
                    data = ds[v].values          # numpy array, one I/O read
                    valid = np.isfinite(data)
                    acc_sum[v][valid]   += data[valid]
                    acc_count[v][valid] += 1

            valid_days += 1
            print(f"  ✓ Day {day:02d} loaded", flush=True)

        except Exception as e:
            print(f"  [ERROR] Day {day:02d}: {e}", flush=True)

    if valid_days == 0:
        print(f"[ERROR] No daily AMF files found for {year}-{month:02d}", flush=True)
        return False

    log_memory(f"Loaded {valid_days} days")
    print(f"  Averaging {valid_days}/{days_in_month} days...", flush=True)

    try:
        t0 = time.time()
        mean_arrays = {}
        for v in VARS:
            mean = np.where(acc_count[v] > 0,
                            acc_sum[v] / acc_count[v],
                            np.nan).astype(np.float32)
            mean_arrays[v] = mean
        del acc_sum, acc_count
        force_cleanup()
        print(f"  Mean computed in {time.time()-t0:.1f}s", flush=True)
        log_memory("After mean")

        monthly_mean = xr.Dataset(
            {v: (['lat', 'lon'], mean_arrays[v]) for v in VARS},
            coords={'lat': ref_lat, 'lon': ref_lon},
        )
        monthly_mean.attrs.update({
            'title': f'Monthly mean TROPOMI AMF for {year}-{month:02d}',
            'source': 'tess_TROPOMI_AMFtot.py daily output',
            'created': datetime.now().isoformat(),
            'days_averaged': valid_days,
            'total_days_in_month': days_in_month,
            'quality_control': qcstr,
        })

        out_fname = f"Tropomi_AMFtot_{year}{month:02d}_Monthly_{qcstr}.nc"
        out_path  = os.path.join(monthly_dir, out_fname)

        chunk_size = 500
        encoding = {v: {'zlib': True, 'complevel': 4, 'shuffle': True,
                        'chunksizes': (chunk_size, chunk_size)}
                    for v in monthly_mean.data_vars}

        print(f"  Writing {out_path} ...", flush=True)
        t0 = time.time()
        monthly_mean.to_netcdf(out_path, encoding=encoding)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  ✓ Written in {time.time()-t0:.1f}s ({size_mb:.1f} MB)", flush=True)

        return out_path

    except Exception as e:
        print(f"  [ERROR] Monthly average failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None


# ─── MONTHLY → YEARLY ────────────────────────────────────────────────────────
def average_monthly_to_yearly(year):
    base_dir    = f'/my-projects2/1.project/NO2col-v2/TROPOMI/{year}'
    monthly_dir = os.path.join(base_dir, 'monthly_AMFtot')
    yearly_dir  = os.path.join(base_dir, 'yearly_AMFtot')
    os.makedirs(yearly_dir, exist_ok=True)

    print(f"Creating yearly AMF average for {year}", flush=True)
    log_memory("Initial")

    VARS = ['amf_trop', 'amf_trop_gcshape', 'amf_tot', 'amf_tot_gcshape']
    acc_sum   = {}
    acc_count = {}
    ref_lat = ref_lon = None
    months_found = []

    for month in range(1, 13):
        fname = f"Tropomi_AMFtot_{year}{month:02d}_Monthly_{qcstr}.nc"
        fpath = os.path.join(monthly_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing monthly: {fname}", flush=True)
            continue
        try:
            with xr.open_dataset(fpath) as ds:
                if ref_lat is None:
                    ref_lat = ds['lat'].values.copy()
                    ref_lon = ds['lon'].values.copy()
                    for v in VARS:
                        acc_sum[v]   = np.zeros(ds[v].shape, dtype=np.float64)
                        acc_count[v] = np.zeros(ds[v].shape, dtype=np.int32)
                for v in VARS:
                    data  = ds[v].values
                    valid = np.isfinite(data)
                    acc_sum[v][valid]   += data[valid]
                    acc_count[v][valid] += 1
            months_found.append(month)
            print(f"  ✓ Month {month:02d} loaded", flush=True)
        except Exception as e:
            print(f"  [ERROR] Month {month:02d}: {e}", flush=True)

    if not months_found:
        print(f"[ERROR] No monthly AMF files found for {year}", flush=True)
        return None

    try:
        mean_arrays = {}
        for v in VARS:
            mean_arrays[v] = np.where(acc_count[v] > 0,
                                      acc_sum[v] / acc_count[v],
                                      np.nan).astype(np.float32)
        del acc_sum, acc_count
        force_cleanup()

        yearly_mean = xr.Dataset(
            {v: (['lat', 'lon'], mean_arrays[v]) for v in VARS},
            coords={'lat': ref_lat, 'lon': ref_lon},
        )
        yearly_mean.attrs.update({
            'title': f'Yearly mean TROPOMI AMF for {year}',
            'source': 'Monthly AMF averages',
            'created': datetime.now().isoformat(),
            'months_included': str(months_found),
        })

        out_fname = f"Tropomi_AMFtot_{year}_Yearly_{qcstr}.nc"
        out_path  = os.path.join(yearly_dir, out_fname)

        encoding = {v: {'zlib': True, 'complevel': 2, 'shuffle': True}
                    for v in yearly_mean.data_vars}
        yearly_mean.to_netcdf(out_path, encoding=encoding)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  ✓ Yearly file written: {out_path} ({size_mb:.1f} MB)", flush=True)
        return out_path

    except Exception as e:
        print(f"  [ERROR] Yearly average failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None


# ─── PLOT ────────────────────────────────────────────────────────────────────
def plot_amf(nc_path, title, out_png):
    """2-panel cartopy map: AMF_tot (top) and AMF_gcshape (bottom)."""
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"[ERROR] Cannot open {nc_path} for plotting: {e}", flush=True)
        return

    lat = ds['lat'].values
    lon = ds['lon'].values


    panels = [
        ('amf_trop',     'Native amf_trop AMFtot',              'AMF (dimensionless)', 'RdYlBu_r',  1.0, 4.0),
        ('amf_trop_gcshape', 'amf_trop_gcshape', 'AMF (dimensionless)', 'RdYlBu_r', 1.0, 4.0),
        ('amf_tot', 'amf_tot',              'AMF (dimensionless)', 'RdYlBu_r',  1.0, 5.0),
        ('amf_tot_gcshape', 'amf_tot_gcshape', 'AMF (dimensionless)', 'RdYlBu_r', 1.0, 5.0),
    ]

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 20),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for ax, (var, label, unit, cmap, vmin, vmax) in zip(axes, panels):
        if var not in ds:
            print(f"  [WARN] Variable {var} not in dataset, skipping", flush=True)
            ax.set_visible(False)
            continue

        data = ds[var].values
        # Use 2–98th percentile for robust colour limits if the default range looks bad
        valid = data[~np.isnan(data)]
        if valid.size > 0:
            p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
            vmin_use = max(vmin, p2)
            vmax_use = min(vmax, p98)
            if vmin_use >= vmax_use:          # fall back if percentiles collapse
                vmin_use, vmax_use = vmin, vmax
        else:
            vmin_use, vmax_use = vmin, vmax

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

        mesh = ax.pcolormesh(
            lon, lat, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='auto', rasterized=True
        )
        cb = plt.colorbar(mesh, ax=ax, orientation='vertical',
                          pad=0.02, fraction=0.015)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)

        # Annotate with basic statistics
        if valid.size > 0:
            stats = f"mean={np.nanmean(data):.2f}  median={np.nanmedian(data):.2f}"
            ax.set_title(f"{label}\n{stats}", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)', fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {out_png}", flush=True)
    ds.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Monthly/yearly average of TROPOMI AMF_tot and AMF_gcshape'
    )
    parser.add_argument('year', type=int, help='Year (e.g. 2019)')
    parser.add_argument('--month', type=int, choices=range(1, 13),
                        metavar='MONTH', help='Process specific month (1-12)')
    parser.add_argument('--yearly-only', action='store_true',
                        help='Create yearly average from existing monthly files')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip PNG plot generation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate PNG plot')
    args = parser.parse_args()

    warnings.filterwarnings('ignore', message='.*chunks separate.*')
    sys.stdout.reconfigure(line_buffering=True)

    print(f"=== TROPOMI AMF AVERAGING ===", flush=True)
    print(f"Year={args.year}  Hostname={os.getenv('HOSTNAME','?')}  "
          f"JobID={os.getenv('LSB_JOBID','local')}", flush=True)
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)

    t_start = datetime.now()
    year    = args.year

    if args.yearly_only:
        base_dir    = f'/my-projects2/1.project/NO2col-v2/TROPOMI/{year}'
        yearly_dir  = os.path.join(base_dir, 'yearly_AMFtot')
        os.makedirs(yearly_dir, exist_ok=True)
        yearly_fname = f"Tropomi_AMFtot_{year}_Yearly_{qcstr}.nc"
        yearly_path  = os.path.join(yearly_dir, yearly_fname)

        if args.plot_only:
            if not os.path.exists(yearly_path):
                print(f"[INFO] Yearly file {yearly_fname} not found, computing it first.", flush=True)
                nc_path = average_monthly_to_yearly(year)
            else:
                nc_path = yearly_path
            if nc_path:
                out_png = nc_path.replace('.nc', '.png')
                plot_amf(nc_path, f"TROPOMI AMF — {year} Annual Mean", out_png)
        else:
            nc_path = average_monthly_to_yearly(year)
            if nc_path and not args.no_plot:
                out_png = nc_path.replace('.nc', '.png')
                plot_amf(nc_path, f"TROPOMI AMF — {year} Annual Mean", out_png)

    elif args.month:
        base_dir    = f'/my-projects2/1.project/NO2col-v2/TROPOMI/{year}'
        monthly_dir = os.path.join(base_dir, 'monthly_AMFtot')
        os.makedirs(monthly_dir, exist_ok=True)
        monthly_fname = f"Tropomi_AMFtot_{year}{args.month:02d}_Monthly_{qcstr}.nc"
        monthly_path  = os.path.join(monthly_dir, monthly_fname)

        if args.plot_only:
            if not os.path.exists(monthly_path):
                print(f"[INFO] Monthly file {monthly_fname} not found, computing it first.", flush=True)
                nc_path = average_daily_to_monthly(year, args.month)
            else:
                nc_path = monthly_path
            if nc_path:
                out_png = nc_path.replace('.nc', '.png')
                plot_amf(nc_path, f"TROPOMI AMF — {year}-{args.month:02d} Monthly Mean", out_png)
            return
        else:
            nc_path = average_daily_to_monthly(year, args.month)
            if nc_path and not args.no_plot:
                out_png = nc_path.replace('.nc', '.png')
                plot_amf(nc_path,
                        f"TROPOMI AMF — {year}-{args.month:02d} Monthly Mean",
                        out_png)

    else:
        print("Specify --month N  or  --yearly-only", flush=True)
        sys.exit(1)

    elapsed = datetime.now() - t_start
    print(f"\nDone in {elapsed}.", flush=True)


if __name__ == '__main__':
    main()
