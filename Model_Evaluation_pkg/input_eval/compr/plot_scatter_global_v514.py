import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math
import warnings
import os

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================
species = 'NO2'
obs_version = 'v6'
geono2_version = 'v5.14'
MOLEC_CM2_PER_DU = 2.6867e16   # unit conversion: molec/cm² → DU
data_dir   = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
output_dir = f'/my-projects2/1.project/Evaluation/obs{obs_version}/plots/'

# ---------- A4 page geometry (inches) for Word with "Normal" 1" margins ----------
A4_PORTRAIT_IN = (4, 6)            # width, height in inches
WORD_MARGINS_IN = (1.0, 1.0, 1.0, 1.0)    # left, right, top, bottom inches
CONTENT_PORTRAIT_IN = (
    A4_PORTRAIT_IN[0] - WORD_MARGINS_IN[0] - WORD_MARGINS_IN[1],
    A4_PORTRAIT_IN[1] - WORD_MARGINS_IN[2] - WORD_MARGINS_IN[3]
)

# ---------- Panel/label styling ----------
AX_LIMITS = (0, 75)            # fixed limits for all panels
TICKS = [0, 25, 50, 75]
TITLE_SIZE = 10
STAT_SIZE = 14
LABEL_SIZE = 14
MARKER_SIZE = 10

# =========================
# Matplotlib base style
# =========================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
    "xtick.major.size": 4,
    "ytick.major.size": 4
})

# =========================
# Utilities
# =========================
def linear_regression(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0 or y.size == 0:
        return -999.0
    xm, ym = x.mean(), y.mean()
    diffx, diffy = x - xm, y - ym
    sst = np.sqrt(np.sum(diffx**2) * np.sum(diffy**2))
    return 0.0 if sst == 0 else (np.sum(diffx * diffy) / sst) ** 2

def regress2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size <= 1 or y.size <= 1:
        return {"slope": -999.0, "intercept": -999.0, "r": -999.0, 'r_square': -999.0,
                "std_slope": -999.0, "std_intercept": -999.0, "predict": -999.0}
    try:
        Xa = sm.add_constant(x); Ya = sm.add_constant(y)
        ia, sa = sm.OLS(y, Xa).fit().params
        ib, sb = sm.OLS(x, Ya).fit().params
        ib = -ib / sb; sb = 1.0 / sb
        if np.sign(sa) != np.sign(sb):
            return {"slope": -999.0, "intercept": -999.0, "r": -999.0, 'r_square': -999.0,
                    "std_slope": -999.0, "std_intercept": -999.0, "predict": -999.0}
        slope = float(np.sign(sa) * np.sqrt(sa * sb))
        intercept = float(y.mean() - slope * x.mean())
        r = float(np.sign(sa) * np.sqrt(sa / sb))
        pred = slope * x + intercept
        n = x.size
        diff = y - pred
        Sx2 = np.sum(x * x)
        den = n * Sx2 - (np.sum(x) ** 2)
        s2 = np.sum(diff * diff) / max(n - 2, 1)
        std_slope = float(np.sqrt(n * s2 / den)) if den != 0 else -999.0
        std_intercept = float(np.sqrt(Sx2 * s2 / den)) if den != 0 else -999.0
        return {"slope": slope, "intercept": intercept, "r": r, 'r_square': r*r,
                "std_slope": std_slope, "std_intercept": std_intercept, "predict": pred}
    except Exception:
        return {"slope": -999.0, "intercept": -999.0, "r": -999.0, 'r_square': -999.0,
                "std_slope": -999.0, "std_intercept": -999.0, "predict": -999.0}

def Cal_NRMSE(estimate, obs):
    estimate = np.asarray(estimate, float); obs = np.asarray(obs, float)
    m = np.isfinite(estimate) & np.isfinite(obs)
    estimate, obs = estimate[m], obs[m]
    if estimate.size == 0 or obs.size == 0:
        return -999.0
    rmse = np.sqrt(mean_squared_error(obs, estimate))
    return rmse / np.mean(obs)

# =========================
# Data loader
# =========================
def load_and_process_data():
    annual_gchp_geo_file = os.path.join(data_dir, f'{species}_annual_{year}_obs{obs_version}_geono2-{geono2_version}.csv')

    gchp_geo_df = pd.read_csv(annual_gchp_geo_file) if os.path.exists(annual_gchp_geo_file) else None
    print(f"Loaded annual GCHP/Geophysical data: {len(gchp_geo_df)} rows" if gchp_geo_df is not None
          else f"Annual GCHP/Geo data file not found: {annual_gchp_geo_file}")

    if gchp_geo_df is not None:
        # Convert molec/cm² → DU for satellite column and GCHP column variables
        du_cols = ([col for col, _ in SAT_COLS] +
                   ['gchp_NO2col_tot', 'gchp_NO2col_trop'])
        for col in du_cols:
            if col in gchp_geo_df.columns:
                gchp_geo_df[col] = gchp_geo_df[col] / MOLEC_CM2_PER_DU

    return gchp_geo_df

# =========================
# Plot helpers
# =========================
def _clean(df, x_col, y_col):
    m = (~pd.isna(df[x_col]) & ~pd.isna(df[y_col]) &
         (df[x_col] != -999) & (df[y_col] != -999))
    return df[m].copy()

def _stats_box(x, y):
    reg = regress2(x, y)
    r2 = linear_regression(x, y)
    nrmse = Cal_NRMSE(y, x)
    slope_txt = f"{reg['slope']:.2f}"
    itcp_txt  = f"{reg['intercept']:.2f}"
    if reg['std_slope'] != -999.0 and reg['std_intercept'] != -999.0:
        slope_txt += f"±{reg['std_slope']:.2f}"
        itcp_txt  += f"±{reg['std_intercept']:.2f}"
    eq = f"y = ({slope_txt})x+{abs(reg['intercept']):.2f}" if reg['intercept'] >= 0 else f"y = ({slope_txt})x-{abs(reg['intercept']):.2f}"
    return f"{eq}\nR² = {r2:.2f}\nNRMSE = {nrmse:.2f}\nN = {len(x)}"

def _panel(ax, df, x_col, y_col, ax_limits=None, ticks=None):
    data = _clean(df, x_col, y_col)
    limits = ax_limits if ax_limits is not None else AX_LIMITS
    tick_vals = ticks if ticks is not None else TICKS

    if data.empty:
        ax.set_xlim(limits); ax.set_ylim(limits); ax.grid(True, alpha=0.25)
        ax.set_xticks(tick_vals); ax.set_yticks(tick_vals)
        ax.set_aspect('equal', adjustable='box')
        return

    x = data[x_col].values; y = data[y_col].values
    ax.scatter(x, y, s=MARKER_SIZE, alpha=0.65)

    # 1:1 and RMA
    ax.plot(limits, limits, 'k--', lw=1.5)
    reg = regress2(x, y)
    if reg['slope'] != -999.0:
        xx = np.linspace(limits[0], limits[1], 150)
        ax.plot(xx, reg['slope'] * xx + reg['intercept'], lw=2.5)

    ax.text(0.02, 0.98, _stats_box(x, y),
            transform=ax.transAxes, ha='left', va='top', fontsize=STAT_SIZE)

    ax.set_xlim(limits); ax.set_ylim(limits)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(tick_vals); ax.set_yticks(tick_vals)
    ax.set_aspect('equal', adjustable='box')

def _sat_panel_limits(df, x_col, y_col):
    """Compute symmetric axis limits for a SatColNO2 panel from the data."""
    data = _clean(df, x_col, y_col)
    if data.empty:
        return (0, 1), [0, 0.5, 1]
    combined = np.concatenate([data[x_col].values, data[y_col].values])
    vmax = np.nanpercentile(combined, 99)
    vmax = max(vmax, 1e-6)
    # Round up to a nice number
    magnitude = 10 ** math.floor(math.log10(vmax))
    vmax_nice = math.ceil(vmax / magnitude) * magnitude
    tick_step = vmax_nice / 4
    ticks = [round(tick_step * i, 10) for i in range(5)]
    return (0, vmax_nice), ticks

gchp_vars = [
    ('gchp_no2',        'GCHP NO2 (ppb)'),
    ('gchp_eta_tot',        'GCHP eta_tot (DU)'),
    ('gchp_eta_trop',        'GCHP eta_trop (DU)'),
    ('gchp_NO2col_tot',        'GCHP NO2col_tot (DU)'),
    ('gchp_NO2col_trop',        'GCHP NO2col_trop (DU)'),
]

geophysical_vars = [
    ('geophysical_no2_tot',        'Geophysical NO2_tot (ppb)'),
    ('geophysical_no2_trop',       'Geophysical NO2_trop (ppb)'),
    ('geophysical_no2_trop_TM5',   'Geophysical NO2_trop_TM5 (ppb)'),
]

SAT_COLS = [
    ('gap_SatColNO2_trop_gcshape',   'SatColNO2_trop_gcshape (DU)'),
    ('gap_SatColNO2_tot_gcshape',    'SatColNO2_tot_gcshape (DU)'),
    ('gap_SatColNO2_trop',           'SatColNO2_trop (DU)'),
    ('gap_SatColNO2_tot',            'SatColNO2_tot (DU)'),
]


def create_global_figure(gchp_geo_df):
    """
    Create 1 column x N rows figure for global data.
    Row 1: Geophysical vs Observed
    Row 2: GCHP vs Observed
    Rows 3-6: sat_no2, sat_no2_tot, sat_no2_tot_gcshape, eta vs Observed
    """
    if gchp_geo_df is None:
        print("No GCHP/Geophysical data available to plot.")
        return

    present_sat = [(col, lbl) for col, lbl in SAT_COLS if col in gchp_geo_df.columns]
    n_rows = len(geophysical_vars) + len(gchp_vars) + len(present_sat)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 4 * n_rows))

    # Geophysical rows
    for i, (col, lbl) in enumerate(geophysical_vars):
        _panel(axes[i], gchp_geo_df, 'obs_no2', col)
        axes[i].set_xlabel('Observed NO₂ (ppb)', fontsize=LABEL_SIZE)
        axes[i].set_ylabel(lbl, fontsize=LABEL_SIZE)

    # GCHP rows
    offset = len(geophysical_vars)
    for i, (col, lbl) in enumerate(gchp_vars, start=offset):
        _panel(axes[i], gchp_geo_df, 'obs_no2', col)
        axes[i].set_xlabel('Observed NO₂ (ppb)', fontsize=LABEL_SIZE)
        axes[i].set_ylabel(lbl, fontsize=LABEL_SIZE)

    # Satellite column rows
    offset = len(geophysical_vars) + len(gchp_vars)
    for i, (col, lbl) in enumerate(present_sat, start=offset):
        sat_limits, sat_ticks = _sat_panel_limits(gchp_geo_df, 'obs_no2', col)
        _panel(axes[i], gchp_geo_df, 'obs_no2', col,
               ax_limits=sat_limits, ticks=sat_ticks)
        axes[i].set_xlabel('Observed NO₂ (ppb)', fontsize=LABEL_SIZE)
        axes[i].set_ylabel(lbl, fontsize=LABEL_SIZE)

    # Save
    out_base = os.path.join(output_dir, f'{species}_global_{year}_obs{obs_version}_geono2-{geono2_version}')
    plt.savefig(out_base + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(out_base + '.svg', bbox_inches='tight', pad_inches=0.05)
    plt.show()
    print(f"Saved: {out_base}.png  and  {out_base}.svg")

# =========================
# Summary stats
# =========================
def create_summary_statistics(gchp_geo_df):
    print("\n" + "="*80)
    print("GLOBAL SUMMARY STATISTICS")
    print("="*80)

    if gchp_geo_df is not None:
        for col, lbl in geophysical_vars:
            print("\nGLOBAL GEOPHYSICAL COMPARISON:")
            print(f"{col}")
            print("-" * 40)
            df = _clean(gchp_geo_df, 'obs_no2', col)
            if not df.empty:
                reg_stats = regress2(df['obs_no2'], df[col])
                r2 = linear_regression(df['obs_no2'], df[col])
                nrmse = Cal_NRMSE(df[col], df['obs_no2'])
                print(f"Slope: {reg_stats['slope']:.3f} ± {reg_stats['std_slope']:.3f}")
                print(f"Intercept: {reg_stats['intercept']:.3f} ± {reg_stats['std_intercept']:.3f}")
                print(f"R²: {r2:.3f}")
                print(f"NRMSE: {nrmse:.3f}")
                print(f"N: {len(df)}")

        for col, lbl in gchp_vars:
            print("\nGLOBAL GCHP COMPARISON:")
            print(f"{col}")
            print("-" * 40)
            df = _clean(gchp_geo_df, 'obs_no2', col)
            if not df.empty:
                reg_stats = regress2(df['obs_no2'], df[col])
                r2 = linear_regression(df['obs_no2'], df[col])
                nrmse = Cal_NRMSE(df[col], df['obs_no2'])
                print(f"Slope: {reg_stats['slope']:.3f} ± {reg_stats['std_slope']:.3f}")
                print(f"Intercept: {reg_stats['intercept']:.3f} ± {reg_stats['std_intercept']:.3f}")
                print(f"R²: {r2:.3f}")
                print(f"NRMSE: {nrmse:.3f}")
                print(f"N: {len(df)}")

        for col, lbl in SAT_COLS:
            if col not in gchp_geo_df.columns:
                continue
            print(f"\nGLOBAL {lbl.upper()} COMPARISON:")
            print(f"{col}")
            print("-" * 40)
            df = _clean(gchp_geo_df, 'obs_no2', col)
            if not df.empty:
                reg_stats = regress2(df['obs_no2'], df[col])
                r2 = linear_regression(df['obs_no2'], df[col])
                nrmse = Cal_NRMSE(df[col], df['obs_no2'])
                print(f"Slope: {reg_stats['slope']:.3f} ± {reg_stats['std_slope']:.3f}")
                print(f"Intercept: {reg_stats['intercept']:.3f} ± {reg_stats['std_intercept']:.3f}")
                print(f"R²: {r2:.3f}")
                print(f"NRMSE: {nrmse:.3f}")
                print(f"N: {len(df)}")

# =========================
# Main
# =========================
def main():
    global year, data_dir, output_dir

    parser = argparse.ArgumentParser(description='Global NO2 scatter plot evaluation (GeoNO2 v5.14)')
    parser.add_argument('--year', type=int, help='Evaluation year (default: %(default)s)')
    args = parser.parse_args()

    year       = args.year
    data_dir   = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
    output_dir = f'/my-projects2/1.project/Evaluation/obs{obs_version}/plots/'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting global NO2 evaluation and visualization (GeoNO2 v5.14) for year {year}...")
    gchp_geo_df = load_and_process_data()
    if gchp_geo_df is None:
        print("No data files found. Please run the data processing script first.")
        return
    create_global_figure(gchp_geo_df)
    create_summary_statistics(gchp_geo_df)
    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
