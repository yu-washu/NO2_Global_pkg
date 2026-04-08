#!/usr/bin/env python3
"""
plot_geono2_vs_satcol_paper.py — Paper figure: GeoNO2 vs SatColNO2.

Layout: 4 rows × 2 columns (map | scatter)
  Row 0: OMI 2005 SatColNO2       Row 1: OMI 2005 GeoNO2
  Row 2: TROPOMI 2023 SatColNO2   Row 3: TROPOMI 2023 GeoNO2

Usage:
  python3 plot_geono2_vs_satcol_paper.py
  python3 plot_geono2_vs_satcol_paper.py --month 7
"""
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
GEONO2_VERSION = 'v5.13'
OBS_VERSION = 'v6'
MONTH = 1

GEONO2_ROOT = '/my-projects2/1.project/GeoNO2-{version}/{year}/'
EVAL_DIR = '/my-projects2/1.project/Evaluation/obs{obs_version}/'
OUTPUT_DIR = '/my-projects2/1.project/Evaluation/obs{obs_version}/plots/'

MAP_VAR_SAT = 'gap_SatColNO2_trop_gcshape'
MAP_VAR_GEO = 'filled_GeoNO2_trop'

SCATTER_SAT = 'gap_SatColNO2_trop_gcshape'
SCATTER_GEO = 'geophysical_no2_trop'
SCATTER_OBS = 'obs_no2'

MOLEC_CM2_PER_DU = 2.6867e16

# All scatter plots use 0–60 range (obs in ppb, SatCol in 0.1 DU, GeoNO2 in ppb)
SCATTER_LIM = 60

# A4 page
A4_WIDTH_IN = 7.5
A4_MAX_HEIGHT_IN = 10.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

# =========================
# Stats utilities
# =========================
def linear_regression(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return -999.0
    xm, ym = x.mean(), y.mean()
    diffx, diffy = x - xm, y - ym
    sst = np.sqrt(np.sum(diffx**2) * np.sum(diffy**2))
    return 0.0 if sst == 0 else (np.sum(diffx * diffy) / sst) ** 2


def regress2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size <= 1:
        return {"slope": np.nan, "intercept": np.nan}
    try:
        Xa = sm.add_constant(x); Ya = sm.add_constant(y)
        ia, sa = sm.OLS(y, Xa).fit().params
        ib, sb = sm.OLS(x, Ya).fit().params
        ib = -ib / sb; sb = 1.0 / sb
        if np.sign(sa) != np.sign(sb):
            return {"slope": np.nan, "intercept": np.nan}
        slope = float(np.sign(sa) * np.sqrt(sa * sb))
        intercept = float(y.mean() - slope * x.mean())
        return {"slope": slope, "intercept": intercept}
    except Exception:
        return {"slope": np.nan, "intercept": np.nan}


# =========================
# Map plotting
# =========================
def plot_map(ax, data, lat, lon, title, vmin=0, vmax=15, cmap='plasma'):
    step = 10
    data_s = data[::step, ::step]
    lat_s = lat[::step]
    lon_s = lon[::step]

    im = ax.pcolormesh(
        lon_s, lat_s, data_s,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading='auto', rasterized=True
    )
    ax.coastlines(linewidth=0.3, color='0.3')
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=11, fontweight='bold')

    n_valid = int(np.isfinite(data_s).sum())
    n_total = data_s.size
    coverage = 100 * n_valid / n_total
    ax.text(0.02, 0.02, f'{coverage:.0f}% coverage',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    return im


def load_map_data(year, month, version=GEONO2_VERSION):
    geo_dir = GEONO2_ROOT.format(version=version, year=year)
    nc_file = os.path.join(geo_dir, f'1x1km.GeoNO2.{year}{month:02d}.MonMean.nc')
    print(f"  Loading map: {nc_file}")

    with xr.open_dataset(nc_file, engine='netcdf4') as ds:
        lat = ds['lat'].values
        lon = ds['lon'].values
        sat = ds[MAP_VAR_SAT].values.astype('float32')
        geo = ds[MAP_VAR_GEO].values.astype('float32')

    # SatCol: molec/cm² → 0.1 DU (multiply by 10 after converting to DU)
    sat_01du = sat / MOLEC_CM2_PER_DU * 10

    return lat, lon, sat_01du, geo


# =========================
# Scatter plotting (hexbin)
# =========================
def plot_scatter(ax, x, y):
    """Hexbin scatter. x=obs (ppb), y=estimate. Both axes 0–SCATTER_LIM."""
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]

    lim = SCATTER_LIM
    im = ax.hexbin(x, y, cmap='RdYlBu_r',
                   norm=mcolors.LogNorm(vmin=1, vmax=100),
                   extent=(0, lim, 0, lim),
                   mincnt=1, gridsize=60, rasterized=True)

    ax.plot([0, lim], [0, lim], color='black', linestyle='--', lw=0.8)

    reg = regress2(x, y)
    b0, b1 = reg['intercept'], reg['slope']
    if np.isfinite(b1):
        ax.plot([0, lim], [b0, b0 + b1 * lim], color='blue', linestyle='-', lw=1)

    r2 = linear_regression(x, y)
    RMSE = np.sqrt(mean_squared_error(x, y))
    n = len(x)
    b0_str = f'+ {abs(b0):.2f}' if b0 >= 0 else f'- {abs(b0):.2f}'
    slope_str = f'{abs(b1):.2f}' if np.isfinite(b1) else '-'
    if np.isfinite(b1) and b1 >= 0:
        eq_str = f'y = {slope_str}x {b0_str}'
    elif np.isfinite(b1):
        eq_str = f'y = -{slope_str}x {b0_str}'
    else:
        eq_str = ''

    # Stats text — positioned in lower-right area, larger line spacing
    fs = 10
    x0 = 0.03 * lim
    y0 = 0.90 * lim  # start lower
    step = 0.10 * lim  # larger spacing
    ax.text(x0, y0 - step * 0, f'$R^2$ = {r2:.2f}', style='italic', fontsize=fs)
    ax.text(x0, y0 - step * 1, f'RMSE = {RMSE:.1f}', style='italic', fontsize=fs)
    ax.text(x0, y0 - step * 2, eq_str, style='italic', fontsize=fs)
    ax.text(x0, y0 - step * 3, f'N = {n:,}', style='italic', fontsize=fs)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=9)

    return im


def load_scatter_data(year, version=GEONO2_VERSION):
    eval_dir = EVAL_DIR.format(obs_version=OBS_VERSION)
    csv_file = os.path.join(eval_dir,
                            f'NO2_annual_{year}_obs{OBS_VERSION}_geono2-{version}.csv')
    print(f"  Loading scatter: {csv_file}")
    df = pd.read_csv(csv_file)

    # SatCol: molec/cm² → 0.1 DU
    for col in [SCATTER_SAT, 'gap_SatColNO2_trop', 'gap_SatColNO2_tot_gcshape', 'gap_SatColNO2_tot']:
        if col in df.columns:
            df[col] = df[col] / MOLEC_CM2_PER_DU * 10

    return df


# =========================
# Main figure
# =========================
def make_figure(month=MONTH):
    fig = plt.figure(figsize=(A4_WIDTH_IN, A4_MAX_HEIGHT_IN))

    # Maps larger, scatters smaller
    gs = fig.add_gridspec(4, 2, width_ratios=[2.3, 1],
                          wspace=0.35, hspace=0.30,
                          left=0.03, right=0.97, top=0.96, bottom=0.03)

    month_name = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec'][month - 1]

    rows = [
        (0, 2005, 'OMI',     'sat'),
        (1, 2005, 'OMI',     'geo'),
        (2, 2023, 'TROPOMI', 'sat'),
        (3, 2023, 'TROPOMI', 'geo'),
    ]

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    pi = 0

    SAT_LABEL = r'SatColNO$_2$'
    GEO_LABEL = r'GeoNO$_2$'

    for row_idx, year, instrument, var_type in rows:
        print(f"\nProcessing {year} ({instrument}) — {var_type}...")

        lat, lon, sat_01du, geo_ppb = load_map_data(year, month)
        df = load_scatter_data(year)

        if var_type == 'sat':
            map_data = sat_01du
            map_title = f'{instrument} {year} {month_name} — ' + SAT_LABEL
            map_vmin, map_vmax = 0, 10  # 0.1 DU
            cbar_label = SAT_LABEL + ' (0.1 DU)'
            scatter_y = df[SCATTER_SAT].values  # y = estimate (0.1 DU)
            scatter_ylabel = SAT_LABEL + ' (0.1 DU)'
        else:
            map_data = geo_ppb
            map_title = f'{instrument} {year} {month_name} — ' + GEO_LABEL
            map_vmin, map_vmax = 0, 15  # ppb
            cbar_label = GEO_LABEL + ' (ppb)'
            scatter_y = df[SCATTER_GEO].values  # y = estimate (ppb)
            scatter_ylabel = GEO_LABEL + ' (ppb)'

        # ---- Map with colorbar on right ----
        ax_map = fig.add_subplot(gs[row_idx, 0], projection=ccrs.Robinson())
        im_map = plot_map(ax_map, map_data, lat, lon, map_title,
                          vmin=map_vmin, vmax=map_vmax, cmap='plasma')

        # Colorbar right of map (using inset)
        # Manual axes: get map position and place cbar to its right
        box = ax_map.get_position()
        cbar_ax = fig.add_axes([box.x1 + 0.005, box.y0 + box.height * 0.15,
                                0.008, box.height * 0.7])
        if var_type == 'sat':
            cb = fig.colorbar(im_map, cax=cbar_ax, orientation='vertical', extend='max',
                              ticks=[0, 5, 10])
        else:
            cb = fig.colorbar(im_map, cax=cbar_ax, orientation='vertical', extend='max')
        cb.ax.tick_params(labelsize=7)

        ax_map.text(-0.02, 1.05, panel_labels[pi], transform=ax_map.transAxes,
                    fontsize=13, fontweight='bold', va='bottom')
        pi += 1

        # ---- Scatter with colorbar on right ----
        ax_sc = fig.add_subplot(gs[row_idx, 1])
        # x = obs (ppb), y = estimate (0.1 DU or ppb). Both axes 0–60.
        im_hex = plot_scatter(ax_sc, df[SCATTER_OBS].values, scatter_y)
        ax_sc.set_xlabel('Measured NO$_2$ (ppb)', fontsize=10)
        ax_sc.set_ylabel(scatter_ylabel, fontsize=10)

        # Hexbin colorbar right of scatter
        divider = make_axes_locatable(ax_sc)
        cbar_ax2 = divider.append_axes("right", size="4%", pad=0.05)
        cb2 = fig.colorbar(im_hex, cax=cbar_ax2, orientation='vertical',
                           ticks=[1, 10, 100])
        cb2.ax.set_yticklabels(['1', '10', r'$10^2$'], fontsize=7)
        cb2.ax.tick_params(labelsize=7)
        cb2.set_label('Number of points', fontsize=7, labelpad=2)

        ax_sc.text(-0.18, 1.05, panel_labels[pi], transform=ax_sc.transAxes,
                   fontsize=13, fontweight='bold', va='bottom')
        pi += 1

    # ---- Save ----
    out_dir = OUTPUT_DIR.format(obs_version=OBS_VERSION)
    os.makedirs(out_dir, exist_ok=True)

    for ext in ['png', 'svg', 'pdf']:
        outpath = os.path.join(out_dir, f'GeoNO2_vs_SatColNO2_paper_{GEONO2_VERSION}.{ext}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {outpath}")

    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Paper figure: GeoNO2 vs SatColNO2 (maps + scatter)')
    parser.add_argument('--month', type=int, default=MONTH, choices=range(1, 13),
                        help='Month for maps (default: 1 = January)')
    args = parser.parse_args()
    make_figure(month=args.month)
