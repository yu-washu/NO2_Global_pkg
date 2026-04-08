import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta
import calendar
import argparse
import sys
import gc
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Configuration
res = '1x1km'
region = 'global'

sza_max, QAlim = 80, 0.75
qcstr = 'SZA{}-QA{}'.format(
    sza_max,
    int(QAlim * 100)
)

def get_days_in_month(year, month):
    """Get number of days in a given month"""
    return calendar.monthrange(year, month)[1]

def slice_latitude(ds, lat_min=-60, lat_max=70):
    """
    Slice dataset to specified latitude range with memory optimization
    """
    lat_coord = None
    for coord in ds.coords:
        if coord.lower() in ['lat', 'latitude', 'y']:
            lat_coord = coord
            break

    if lat_coord is None:
        print("[WARN] No latitude coordinate found, returning original dataset")
        return ds

    lat_slice = ds.sel({lat_coord: slice(lat_min, lat_max)})

    lat_slice.attrs.update({
        'latitude_slice': f'{lat_min} to {lat_max} degrees',
        'original_lat_range': f'{float(ds[lat_coord].min().values):.2f} to {float(ds[lat_coord].max().values):.2f}'
    })

    return lat_slice

# ---------------------------------------------------------------------------
# Monthly GCHP surface NO2 (derived from daily files — same as v3)
# ---------------------------------------------------------------------------

def process_single_day_gchp(year, month, day):
    """
    Load GCHP surface NO2 and related species for a single day.
    Returns a dict with float32 arrays, or None if any file is missing.
    """
    gchp_dir = f'/my-projects2/1.project/gchp/forObservation-Geophysical/{year}/'

    gchp_daily_path  = gchp_dir + f'daily/{res}.DailyVars.{year}{month:02d}{day:02d}.nc4'
    gchp_3hours_path = gchp_dir + f'daily/{res}.Hours.13-15.{year}{month:02d}{day:02d}.nc4'

    if not all(os.path.exists(f) for f in [gchp_daily_path, gchp_3hours_path]):
        return None

    try:
        with xr.open_dataset(gchp_daily_path, engine='netcdf4') as gchp_daily:
            gchp_daily = slice_latitude(gchp_daily.squeeze(), -60, 70)
            gchp_no2        = gchp_daily['gchp_NO2'].values.astype('float32')
            gchp_hno3       = gchp_daily['gchp_HNO3'].values.astype('float32')
            gchp_pan        = gchp_daily['gchp_PAN'].values.astype('float32')

            if 'gchp_alkylnitrates' in gchp_daily:
                gchp_alkylnitrates = gchp_daily['gchp_alkylnitrates'].values.astype('float32')
            elif 'gchp_alklnitrates' in gchp_daily:
                gchp_alkylnitrates = gchp_daily['gchp_alklnitrates'].values.astype('float32')
            else:
                gchp_alkylnitrates = np.zeros_like(gchp_no2, dtype='float32')

        with xr.open_dataset(gchp_3hours_path, engine='netcdf4') as gchp_3hours:
            gchp_3hours = slice_latitude(gchp_3hours.squeeze(), -60, 70)
            lat_coords  = gchp_3hours['lat'].values
            lon_coords  = gchp_3hours['lon'].values

        return {
            'gchp_NO2':          gchp_no2,
            'gchp_HNO3':         gchp_hno3,
            'gchp_PAN':          gchp_pan,
            'gchp_alkylnitrates': gchp_alkylnitrates,
            'lat':               lat_coords,
            'lon':               lon_coords,
        }

    except Exception as e:
        print(f"    Error processing day {day:02d}: {str(e)}")
        return None


def derive_monthly_gchp(year, month):
    """
    Average GCHP surface species over the days of a month.
    Returns (monthly_means dict, lat, lon) or (None, None, None).
    """
    print(f"  Deriving monthly GCHP surface NO2 for {year}-{month:02d}...")

    days_in_month = get_days_in_month(year, month)
    monthly_sums  = {}
    valid_counts  = {}
    valid_day_count = 0
    lat_coords = lon_coords = None

    for day in range(1, days_in_month + 1):
        print(f"    Day {day:02d}...", end=' ')
        result = process_single_day_gchp(year, month, day)

        if result is None:
            print("missing")
            continue

        if valid_day_count == 0:
            lat_coords = result['lat']
            lon_coords = result['lon']
            for var in ['gchp_NO2', 'gchp_HNO3', 'gchp_PAN', 'gchp_alkylnitrates']:
                monthly_sums[var]  = np.zeros_like(result[var], dtype='float64')
                valid_counts[var]  = np.zeros_like(result[var], dtype='int32')

        for var in monthly_sums:
            d = result[var]
            mask = ~np.isnan(d)
            monthly_sums[var]  = np.where(mask, monthly_sums[var] + d.astype('float64'), monthly_sums[var])
            valid_counts[var]  = np.where(mask, valid_counts[var] + 1, valid_counts[var])

        valid_day_count += 1
        print("✓")

        del result
        gc.collect()

    if valid_day_count == 0:
        print(f"  ✗ No valid GCHP daily data for {year}-{month:02d}")
        return None, None, None

    print(f"  Averaged {valid_day_count} days")

    min_obs = 5
    monthly_means = {}
    for var, s in monthly_sums.items():
        cnt = valid_counts[var]
        monthly_means[var] = np.where(cnt >= min_obs,
                                      (s / cnt).astype('float32'),
                                      np.nan)

    return monthly_means, lat_coords, lon_coords


# ---------------------------------------------------------------------------
# Core GeoNO2 calculation (v4 — tropospheric column)
# ---------------------------------------------------------------------------

def compute_geono2_trop(year, month):
    """
    Compute surface GeoNO2 using tropospheric column scaling:

        GeoNO2 = NO2_trop_filled × (gchp_surface_NO2 / GCHP_TropNO2col)

    NO2_trop_filled and GCHP_TropNO2col are read from the pre-filled monthly
    file produced by TropNO2col-v2.  gchp_surface_NO2 is derived here from
    GCHP daily files.
    """
    print(f"\n=== Processing {year}-{month:02d} (v4 — tropospheric column scaling) ===")

    # ---- 1. Load pre-filled TROPOMI trop NO2 and GCHP trop column ----
    trop_dir  = f'/my-projects2/1.project/NO2_col/TropNO2col-v2/{year}/'
    trop_file = f'{trop_dir}{res}.TROPOMI.TropNO2col_filled.{year}{month:02d}.MonMean.nc'

    if not os.path.exists(trop_file):
        print(f"  ✗ Pre-filled trop NO2 file not found: {trop_file}")
        return False

    print(f"  Loading: {trop_file}")
    with xr.open_dataset(trop_file, engine='netcdf4') as ds_trop:
        NO2_trop_filled  = ds_trop['NO2_trop_filled'].values.astype('float32')
        GCHP_TropNO2col  = ds_trop['GCHP_TropNO2col'].values.astype('float32')
        lat_trop         = ds_trop['lat'].values
        lon_trop         = ds_trop['lon'].values

    print(f"  TROPOMI trop NO2 shape: {NO2_trop_filled.shape}")
    print(f"  GCHP trop col shape:    {GCHP_TropNO2col.shape}")

    valid_trop = int(np.sum(~np.isnan(NO2_trop_filled)))
    total_pix  = NO2_trop_filled.size
    print(f"  NO2_trop_filled valid: {valid_trop:,}/{total_pix:,} ({100*valid_trop/total_pix:.1f}%)")

    # ---- 2. Derive monthly mean GCHP surface NO2 from daily files ----
    gchp_means, lat_gchp, lon_gchp = derive_monthly_gchp(year, month)

    if gchp_means is None:
        print("  ✗ Cannot compute GeoNO2: no GCHP daily data")
        return False

    gchp_no2 = gchp_means['gchp_NO2']

    # ---- 3. Surface-to-trop-column ratio ----
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = np.where(GCHP_TropNO2col > 0,
                       gchp_no2 / GCHP_TropNO2col,
                       np.nan).astype('float32')

    # ---- 4. GeoNO2 = TROPOMI trop col × eta ----
    GeoNO2 = np.where(~np.isnan(NO2_trop_filled) & ~np.isnan(eta),
                      NO2_trop_filled * eta,
                      np.nan).astype('float32')

    valid_geo = int(np.sum(~np.isnan(GeoNO2)))
    print(f"  GeoNO2 valid: {valid_geo:,}/{total_pix:,} ({100*valid_geo/total_pix:.1f}%)")
    print(f"  GeoNO2 range: {np.nanmin(GeoNO2):.2e} – {np.nanmax(GeoNO2):.2e}")

    # ---- 5. Build output dataset ----
    out_ds = xr.Dataset({
        'GeoNO2':            (['lat', 'lon'], GeoNO2),
        'NO2_trop_filled':   (['lat', 'lon'], NO2_trop_filled),
        'GCHP_TropNO2col':   (['lat', 'lon'], GCHP_TropNO2col),
        'gchp_NO2':          (['lat', 'lon'], gchp_no2),
        'gchp_HNO3':         (['lat', 'lon'], gchp_means['gchp_HNO3']),
        'gchp_PAN':          (['lat', 'lon'], gchp_means['gchp_PAN']),
        'gchp_alkylnitrates':(['lat', 'lon'], gchp_means['gchp_alkylnitrates']),
    }, coords={
        'lat':  lat_trop,
        'lon':  lon_trop,
        'time': datetime(year, month, 15),
    })

    out_ds.attrs.update({
        'title':              f'Monthly surface GeoNO2 (v4 — tropospheric column) {year}-{month:02d}',
        'source':             'TROPOMI TropNO2col-v2 (pre-filled) + GCHP c180 surface NO2',
        'method':             'GeoNO2 = NO2_trop_filled × (gchp_NO2 / GCHP_TropNO2col)',
        'version':            'v4',
        'created':            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'quality_control':    qcstr,
        'processed_by_pid':   os.getpid(),
    })

    var_attrs = {
        'GeoNO2':            'Surface GeoNO2 derived from tropospheric column scaling [molec/cm2 → surface unit]',
        'NO2_trop_filled':   'TROPOMI tropospheric NO2 column, gap-filled [molec/cm2]',
        'GCHP_TropNO2col':   'GCHP tropospheric NO2 column [molec/cm2]',
        'gchp_NO2':          'GCHP surface NO2 monthly mean',
        'gchp_HNO3':         'GCHP surface HNO3 monthly mean',
        'gchp_PAN':          'GCHP surface PAN monthly mean',
        'gchp_alkylnitrates':'GCHP surface alkylnitrates monthly mean',
    }
    for var, attr in var_attrs.items():
        if var in out_ds:
            out_ds[var].attrs['long_name'] = attr

    # ---- 6. Save ----
    out_dir  = f'/my-projects2/1.project/GeoNO2-v4/{year}/'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}{res}.GeoNO2.{year}{month:02d}.MonMean.nc'

    encoding = {v: {'zlib': True, 'complevel': 4, 'shuffle': True, 'dtype': 'float32'}
                for v in out_ds.data_vars}

    print(f"  Saving: {out_path}")
    out_ds.to_netcdf(out_path, encoding=encoding)

    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  ✓ Saved ({file_size:.2f} MB)")

    plot_geono2_monthly(out_path, year, month)

    del out_ds, GeoNO2, NO2_trop_filled, GCHP_TropNO2col, gchp_means
    gc.collect()

    return True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_VARS   = ['GeoNO2', 'NO2_trop_filled', 'gchp_NO2']
PLOT_LABELS = {
    'GeoNO2':          'GeoNO2 (trop-col scaling)',
    'NO2_trop_filled': 'TROPOMI trop NO2 (filled)',
    'gchp_NO2':        'GCHP surface NO\u2082',
}

def plot_geono2_monthly(outpath, year, month):
    """Plot GeoNO2, NO2_trop_filled, and gchp_NO2 from the monthly output NetCDF."""
    try:
        with xr.open_dataset(outpath, engine='netcdf4') as ds:
            lats      = ds['lat'].values
            lons      = ds['lon'].values
            available = [v for v in PLOT_VARS if v in ds.data_vars]

        if not available:
            print("  [WARN] None of the plot variables found, skipping plot", flush=True)
            return

        n = len(available)
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n),
                                 subplot_kw={'projection': ccrs.PlateCarree()},
                                 squeeze=False)

        with xr.open_dataset(outpath, engine='netcdf4') as ds:
            for i, var in enumerate(available):
                ax = axes[i, 0]
                v  = ds[var].values.squeeze()

                if v.size == 0 or np.all(np.isnan(v)):
                    ax.set_visible(False)
                    continue

                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
                ax.set_extent([float(lons.min()), float(lons.max()),
                               float(lats.min()), float(lats.max())],
                              crs=ccrs.PlateCarree())

                mesh = ax.pcolormesh(lons, lats, v,
                                     transform=ccrs.PlateCarree(),
                                     cmap='RdYlBu_r', vmin=0, vmax=15)
                cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                                    pad=0.02, fraction=0.03, shrink=0.75)
                cbar.set_label(PLOT_LABELS.get(var, var))
                ax.set_title(f"{year}-{month:02d}  {PLOT_LABELS.get(var, var)}",
                             fontsize=12, pad=6)
                ax.gridlines(draw_labels=True, alpha=0.3, linewidth=0.4)

        fig.subplots_adjust(top=0.96, bottom=0.12, left=0.04, right=0.96, hspace=0.4)
        png_path = outpath.replace('.nc', '.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot: {png_path}", flush=True)

    except Exception as e:
        print(f"  [WARN] Plot failed: {str(e)}", flush=True)


# ---------------------------------------------------------------------------
# Annual average
# ---------------------------------------------------------------------------

def process_yearly_average(year):
    """Create annual average from existing monthly files."""
    print(f"\n=== Creating Annual Average for {year} (v4) ===")

    monthly_dir = f'/my-projects2/1.project/GeoNO2-v4/{year}/'

    if not os.path.exists(monthly_dir):
        print(f"✗ Monthly directory not found: {monthly_dir}")
        return False

    monthly_sums  = {}
    valid_month_count = 0
    lat_coords = lon_coords = None
    months_found = []

    for month in range(1, 13):
        monthly_file = f'{monthly_dir}{res}.GeoNO2.{year}{month:02d}.MonMean.nc'

        if not os.path.exists(monthly_file):
            continue

        try:
            print(f"  Month {month:02d}...", end=' ')
            with xr.open_dataset(monthly_file, engine='netcdf4') as ds_month:
                if valid_month_count == 0:
                    lat_coords = ds_month['lat'].values
                    lon_coords = ds_month['lon'].values
                    for var in ds_month.data_vars:
                        monthly_sums[var] = np.zeros_like(ds_month[var].values, dtype='float64')

                for var in ds_month.data_vars:
                    data = ds_month[var].values
                    mask = ~np.isnan(data)
                    monthly_sums[var] = np.where(mask,
                                                 monthly_sums[var] + data.astype('float64'),
                                                 monthly_sums[var])

            valid_month_count += 1
            months_found.append(month)
            print("✓")

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue

        gc.collect()

    if valid_month_count == 0:
        print(f"✗ No valid monthly files found for {year}")
        return False

    annual_means = {var: (s / valid_month_count).astype('float32')
                    for var, s in monthly_sums.items()}

    annual_ds = xr.Dataset(
        {var: (['lat', 'lon'], data) for var, data in annual_means.items()},
        coords={'lat': lat_coords, 'lon': lon_coords, 'time': datetime(year, 7, 1)}
    )
    annual_ds.attrs.update({
        'title':            f'Annual average GeoNO2 (v4 — tropospheric column) {year}',
        'source':           'TROPOMI TropNO2col-v2 + GCHP c180',
        'months_averaged':  valid_month_count,
        'months_included':  ', '.join([f'{m:02d}' for m in months_found]),
        'created':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })

    out_path = f'{monthly_dir}{res}.GeoNO2.{year}.AnnualMean.nc'
    encoding = {v: {'zlib': True, 'complevel': 4, 'shuffle': True, 'dtype': 'float32'}
                for v in annual_ds.data_vars}
    annual_ds.to_netcdf(out_path, encoding=encoding)

    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✓ Annual average saved ({file_size:.2f} MB): {out_path}")

    del monthly_sums, annual_means, annual_ds
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Top-level wrappers
# ---------------------------------------------------------------------------

def process_single_month(year, month):
    """Process one month."""
    print(f"\n=== Single Month: {year}-{month:02d} ===")
    return compute_geono2_trop(year, month)


def process_year(year):
    """Process all months for a year, then create annual average."""
    print(f"\n=== Processing Year {year} (v4) ===")
    successful = 0

    for month in range(1, 13):
        print(f"\n--- Month {month:02d}/12 ---")
        if compute_geono2_trop(year, month):
            successful += 1
        gc.collect()

    if successful > 0:
        print("\n--- Creating Annual Average ---")
        process_yearly_average(year)

    print(f"\n=== Year {year} Summary: {successful}/12 months processed ===")
    return successful > 0


def plot_only(year, month=None):
    """Re-generate PNG plots from existing monthly NetCDF files."""
    base_dir = f'/my-projects2/1.project/GeoNO2-v4/{year}/'
    months   = [month] if month else range(1, 13)
    plotted  = 0
    for m in months:
        nc = f'{base_dir}{res}.GeoNO2.{year}{m:02d}.MonMean.nc'
        if not os.path.exists(nc):
            print(f"  [WARN] Not found: {nc}", flush=True)
            continue
        print(f"Plotting {year}-{m:02d}...", flush=True)
        plot_geono2_monthly(nc, year, m)
        plotted += 1
    print(f"Done: {plotted} plot(s) generated.")
    return plotted > 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GeoNO2 v4 — tropospheric column scaling (no gap filling)')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2023)')
    parser.add_argument('--month', type=int, choices=range(1, 13),
                        help='Process only a specific month (1-12)')
    parser.add_argument('--yearly-only', action='store_true',
                        help='Only create annual average from existing monthly files')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate PNG plots from existing NetCDF files')

    args = parser.parse_args()
    year = args.year

    print(f"GeoNO2 v4 — tropospheric column scaling")
    print(f"Year: {year}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        start_time = datetime.now()

        if args.plot_only:
            success = plot_only(year, args.month)
        elif args.yearly_only:
            success = process_yearly_average(year)
        elif args.month:
            success = process_single_month(year, args.month)
        else:
            success = process_year(year)

        duration = datetime.now() - start_time

        if success:
            print(f"\n✓ Completed. Total time: {duration}")
            sys.exit(0)
        else:
            print(f"\n✗ Failed for year {year}")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted")
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
