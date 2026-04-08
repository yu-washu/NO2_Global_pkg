"""
Map the number of daily samples (valid days) per pixel for OMI-MINDS NO2.

For a given year (and optionally month), counts how many daily files have
valid (finite) NO2_tot at each grid cell and plots the result.
Optionally saves the count field as netCDF.
"""

import os
import argparse
import calendar
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# Default paths (match omi_average.py)
DEFAULT_BASE_DIR = '/my-projects2/1.project/NO2_col/OMI-MINDS/'
DEFAULT_GRID_LON = '/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global_MAP.npy'
DEFAULT_GRID_LAT = '/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global_MAP.npy'

# QC string for daily files (match average_daily_to_monthly in omi_average.py)
ECF_max, SZA_max, QAFlag, RowAnomalyFlag = 0.2, 75, 0, 0
QCSTR = 'ECF{:03d}-SZA{}-QA{}-RA{}'.format(
    int(ECF_max * 100), SZA_max, int(QAFlag), int(RowAnomalyFlag)
)


def get_sample_count(year, month=None, base_dir=DEFAULT_BASE_DIR, var='NO2_tot'):
    """
    Count how many daily files have valid (finite) data at each pixel.

    Parameters
    ----------
    year : int
    month : int or None
        If None, use all months in the year.
    base_dir : str
    var : str
        Variable to use for validity (e.g. 'NO2_tot').

    Returns
    -------
    count_2d : np.ndarray
        Shape (ny, nx); each element is number of days with valid data.
    valid_days : int
        Number of daily files actually read.
    days_possible : int
        Total number of days in the period.
    """
    daily_dir = os.path.join(base_dir, f"{year}", "daily")
    if not os.path.isdir(daily_dir):
        raise FileNotFoundError(f"Daily directory not found: {daily_dir}")

    if month is not None:
        days_in_period = calendar.monthrange(year, month)[1]
        month_list = [month]
    else:
        days_in_period = 366 if calendar.isleap(year) else 365
        month_list = range(1, 13)

    count_2d = None
    valid_days = 0

    for m in month_list:
        days_in_month = calendar.monthrange(year, m)[1]
        for day in range(1, days_in_month + 1):
            fname = f"OMI-MINDS_Regrid_{year}{m:02d}{day:02d}_{QCSTR}.nc"
            fpath = os.path.join(daily_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                ds = xr.open_dataset(fpath)
                if var not in ds:
                    ds.close()
                    continue
                data = ds[var].values
                ds.close()
                # Squeeze to 2D if there are extra dims
                if data.ndim > 2:
                    data = np.squeeze(data)
                valid = np.isfinite(data)
                if count_2d is None:
                    count_2d = np.zeros(data.shape, dtype=np.int32)
                count_2d += valid.astype(np.int32)
                valid_days += 1
            except Exception as e:
                print(f"  [WARN] Skip {fname}: {e}")
                continue

    if count_2d is None:
        raise ValueError(f"No daily files found for {year}" + (f"-{month:02d}" if month else ""))

    return count_2d, valid_days, days_in_period


def load_grid(grid_lon_path=DEFAULT_GRID_LON, grid_lat_path=DEFAULT_GRID_LAT):
    """Load lon/lat grid for plotting."""
    if not os.path.isfile(grid_lon_path) or not os.path.isfile(grid_lat_path):
        raise FileNotFoundError(
            f"Grid files not found: {grid_lon_path}, {grid_lat_path}"
        )
    x = np.load(grid_lon_path)
    y = np.load(grid_lat_path)
    return x, y


def plot_sample_count(
    count_2d,
    x,
    y,
    title,
    out_png,
    vmin=None,
    vmax=None,
    cmap='RdYlBu_r',
    extend='neither',
):
    """Plot 2D sample count as a map."""
    if count_2d.size == 0 or np.all(count_2d == 0):
        print("[WARN] No valid counts to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = np.nanmax(count_2d) if np.any(np.isfinite(count_2d)) else 1

    # Use a discrete-ish colormap for integer counts
    mesh = ax.pcolormesh(
        x, y, count_2d,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading='auto',
    )
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, extend=extend)
    cbar.set_label('Number of daily samples')
    ax.set_title(title, pad=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description='Map number of daily samples per pixel for OMI-MINDS NO2'
    )
    parser.add_argument('year', type=int, help='Year (e.g. 2019)')
    parser.add_argument('--month', type=int, default=None, metavar='M',
                        choices=range(1, 13), help='Restrict to month 1-12')
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR,
                        help='Base directory for OMI-MINDS daily/yearly')
    parser.add_argument('--grid-lon', type=str, default=DEFAULT_GRID_LON,
                        help='Path to longitude grid .npy')
    parser.add_argument('--grid-lat', type=str, default=DEFAULT_GRID_LAT,
                        help='Path to latitude grid .npy')
    parser.add_argument('--var', type=str, default='NO2_tot',
                        help='Variable used to define valid pixel')
    parser.add_argument('--out-png', type=str, default=None,
                        help='Output PNG path (default: base_dir/year/daily or yearly)')
    parser.add_argument('--out-nc', type=str, default=None,
                        help='Optional: save sample count as netCDF')
    parser.add_argument('--vmin', type=float, default=None)
    parser.add_argument('--vmax', type=float, default=None)
    parser.add_argument('--no-plot', action='store_true', help='Do not create PNG')
    args = parser.parse_args()

    year = args.year
    month = args.month
    period_str = f"{year}-{month:02d}" if month else str(year)
    print(f"Computing daily sample count for {period_str}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    count_2d, valid_days, days_possible = get_sample_count(
        year, month=month, base_dir=args.base_dir, var=args.var
    )
    print(f"  Valid days read: {valid_days} / {days_possible}")
    print(f"  Count range: {int(np.nanmin(count_2d))} - {int(np.nanmax(count_2d))}")

    # Save netCDF if requested
    if args.out_nc:
        # Build minimal dataset with same grid info as daily files if possible
        ds_out = xr.Dataset(
            {
                'sample_count': (('lat', 'lon'), count_2d),
            },
            coords={
                'lat': np.arange(count_2d.shape[0]),
                'lon': np.arange(count_2d.shape[1]),
            },
            attrs={
                'title': f'OMI-MINDS daily sample count per pixel, {period_str}',
                'period': period_str,
                'valid_days_used': valid_days,
                'days_in_period': days_possible,
                'quality_control': QCSTR,
                'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
        )
        ds_out.to_netcdf(args.out_nc)
        print(f"Saved netCDF: {args.out_nc}")

    if args.no_plot:
        return

    x, y = load_grid(args.grid_lon, args.grid_lat)
    if x.shape != count_2d.shape or y.shape != count_2d.shape:
        print(f"[WARN] Grid shape {x.shape} vs count shape {count_2d.shape}; attempting plot anyway")
        # Allow (nlat+1, nlon+1) grid for pcolormesh
        if x.shape[0] == count_2d.shape[0] + 1 and x.shape[1] == count_2d.shape[1] + 1:
            pass
        else:
            print("  Skipping plot due to shape mismatch.")
            return

    title = f"Daily sample count, {period_str} (n={valid_days} days)"
    if args.out_png:
        out_png = args.out_png
    else:
        subdir = os.path.join(args.base_dir, str(year), "yearly" if month is None else "monthly")
        os.makedirs(subdir, exist_ok=True)
        suffix = f"{year}{month:02d}" if month else str(year)
        out_png = os.path.join(subdir, f"OMI-MINDS_sample_count_{suffix}_{QCSTR}.png")

    plot_sample_count(
        count_2d, x, y, title, out_png,
        vmin=args.vmin, vmax=args.vmax,
    )
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
