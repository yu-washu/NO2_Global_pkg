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
year = 2019
data_dir = f'/my-projects2/1.project/Evaluation/global/'
output_dir = f'/my-projects2/1.project/Evaluation/global/plots/'
os.makedirs(output_dir, exist_ok=True)

# ---------- A4 page geometry (inches) for Word with "Normal" 1" margins ----------
A4_PORTRAIT_IN = (8.27, 11.69)            # width, height in inches
WORD_MARGINS_IN = (1.0, 1.0, 1.0, 1.0)    # left, right, top, bottom inches
CONTENT_PORTRAIT_IN = (
    A4_PORTRAIT_IN[0] - WORD_MARGINS_IN[0] - WORD_MARGINS_IN[1],
    A4_PORTRAIT_IN[1] - WORD_MARGINS_IN[2] - WORD_MARGINS_IN[3]
)
CONTENT_LANDSCAPE_IN = (CONTENT_PORTRAIT_IN[1], CONTENT_PORTRAIT_IN[0])

# ---------- Panel/label styling tuned for A4 ----------
AX_LIMITS = (0, 75)            # fixed limits for all panels
TICKS = [0, 25, 50, 75]
TITLE_SIZE = 8
STAT_SIZE = 8
SUPLABEL_SIZE = 10
ROW_LABEL_SIZE = 12
MARKER_SIZE = 6

# =========================
# Matplotlib base style: compact and readable
# =========================
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1.0,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5
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
# Data loaders
# =========================
def load_and_process_data():
    monthly_file = os.path.join(data_dir, f'{species}_monthly_{year}_processed_china_ugm3.csv')
    annual_gchp_geo_file = os.path.join(data_dir, f'{species}_annual_gchp_geo_{year}_processed_china_ugm3.csv')
    annual_cooper_file = os.path.join(data_dir, f'{species}_annual_cooper_{year}_processed_china_ugm3.csv')

    monthly_df = pd.read_csv(monthly_file) if os.path.exists(monthly_file) else None
    print(f"Loaded monthly data: {len(monthly_df)} rows" if monthly_df is not None
          else f"Monthly data file not found: {monthly_file}")

    gchp_geo_df = pd.read_csv(annual_gchp_geo_file) if os.path.exists(annual_gchp_geo_file) else None
    print(f"Loaded annual GCHP/Geophysical data: {len(gchp_geo_df)} rows" if gchp_geo_df is not None
          else f"Annual GCHP/Geo data file not found: {annual_gchp_geo_file}")

    cooper_df = pd.read_csv(annual_cooper_file) if os.path.exists(annual_cooper_file) else None
    if cooper_df is not None:
        cooper_df = cooper_df[cooper_df['cooper_no2'] != -999.0]
        print(f"Loaded annual Cooper data: {len(cooper_df)} rows (after filtering -999)")
    else:
        print(f"Annual Cooper data file not found: {annual_cooper_file}")

    return monthly_df, gchp_geo_df, cooper_df

# =========================
# Plot helpers
# =========================
REGION_ORDER  = ['global', 'northamerica', 'asia', 'europe', 'southamerica', 'oceania']
REGION_TITLES = ['Global',  'N. America',  'Asia', 'Europe', 'S. America', 'Oceania']

def _clean(df, x_col, y_col):
    m = (~pd.isna(df[x_col]) & ~pd.isna(df[y_col]) &
         (df[x_col] != -999) & (df[y_col] != -999) &
         (df['region'] != 'unknown'))
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

def _panel(ax, df, x_col, y_col, region, show_stats, title):
    data = _clean(df, x_col, y_col)
    if region != 'global':
        data = data[data['region'] == region]
    if data.empty:
        ax.set_title("No data", fontweight='bold')
        ax.set_xlim(AX_LIMITS); ax.set_ylim(AX_LIMITS); ax.grid(True, alpha=0.25)
        ax.set_xticks(TICKS); ax.set_yticks(TICKS)
        return

    x = data[x_col].values; y = data[y_col].values
    ax.scatter(x, y, s=MARKER_SIZE, alpha=0.65)

    # 1:1 and RMA
    ax.plot(AX_LIMITS, AX_LIMITS, 'k--', lw=1)
    reg = regress2(x, y)
    if reg['slope'] != -999.0:
        xx = np.linspace(AX_LIMITS[0], AX_LIMITS[1], 150)
        ax.plot(xx, reg['slope'] * xx + reg['intercept'], lw=2)

    if show_stats:
        ax.text(0.02, 0.98, _stats_box(x, y),
                transform=ax.transAxes, ha='left', va='top', fontsize=STAT_SIZE)  # no box

    ax.set_title(title, fontweight='bold')
    ax.set_xlim(AX_LIMITS); ax.set_ylim(AX_LIMITS)
    ax.grid(True, alpha=0.25)
    ax.set_xticks(TICKS); ax.set_yticks(TICKS)

def create_method_by_region_figure(gchp_geo_df, cooper_df, a4_orientation='landscape'):
    """
    Rows (top->bottom): Cooper, Geophysical, GCHP (skip a row if missing).
    Cols: Global | N. America | Asia | Europe | S. America | Oceania
    A4 Word-friendly export.
    """
    rows = []
    if cooper_df is not None:
        rows.append((f"Cooper {year}", cooper_df, 'cooper_no2'))
    if gchp_geo_df is not None:
        rows.append((f"Geophysical {year}", gchp_geo_df, 'geophysical_no2'))
        rows.append((f"GCHP {year}",        gchp_geo_df, 'gchp_no2'))
    if not rows:
        print("No datasets available to plot.")
        return

    n_rows = len(rows); n_cols = len(REGION_ORDER)

    # --- A4 content box size ---
    if a4_orientation.lower().startswith('land'):
        fig_w, fig_h = CONTENT_LANDSCAPE_IN  # ~9.69" × ~6.27"
    else:
        fig_w, fig_h = CONTENT_PORTRAIT_IN   # ~6.27" × ~9.69"

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Extra wspace so region titles don't collide
    gs = fig.add_gridspec(
        n_rows, n_cols,
        left=0.08,   # was 0.11
        right=0.998, # was 0.985 → removes the wide right margin
        top=0.94,
        bottom=0.10,
        wspace=0.25,
        hspace=0.20
    )
    axes = gs.subplots()

    # Row labels in the margin (won't shrink panels)
    row_ycenters = [(n_rows - r - 0.5) / n_rows for r in range(n_rows)]

    for r, (row_title, df, ycol) in enumerate(rows):
        for c, region in enumerate(REGION_ORDER):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            _panel(ax, df, 'obs_no2', ycol, region,
                   show_stats=True, title=REGION_TITLES[c])
            # use figure-level labels, so keep per-axis labels empty
            ax.set_xlabel(''); ax.set_ylabel('')

        fig.text(0.035, row_ycenters[r], row_title,
                 rotation=90, va='center', ha='left',
                 fontsize=ROW_LABEL_SIZE, fontweight='bold')

    # ---------- Figure-level centered labels aligned to the grid ----------
    x_center = (gs.left + gs.right) / 2.0
    y_center = (gs.bottom + gs.top) / 2.0

    fig.text(x_center, gs.bottom - 0.058, 'Observed ' + r'$\mathrm{NO_{2}}$ (ppb)',
            ha='center', va='top', fontsize=SUPLABEL_SIZE)

    fig.text(gs.left - 0.012, y_center, 'Estimated ' + r'$\mathrm{NO_{2}}$ (ppb)',
            ha='right', va='center', rotation=90, fontsize=SUPLABEL_SIZE)

    # --- Save: 600dpi PNG + SVG (SVG is best for Word) ---
    out_base = os.path.join(output_dir, f'{species}_method_by_region_{year}_A4_{a4_orientation}_china_ugm3')
    plt.savefig(out_base + '.png', dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(out_base + '.svg', bbox_inches='tight', pad_inches=0.01)
    plt.show()
    print(f"Saved: {out_base}.png  and  {out_base}.svg")

# =========================
# (Optional) Summary stats
# =========================
def create_summary_statistics(monthly_df, gchp_geo_df, cooper_df):
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    if cooper_df is not None:
        print("\nANNUAL COOPER COMPARISON:")
        print("-" * 40)
        df = cooper_df
        valid = (df['region'] != 'unknown') & (df['obs_no2'] != -999) & (df['cooper_no2'] != -999)
        df = df[valid]
        if not df.empty:
            reg_stats = regress2(df['obs_no2'], df['cooper_no2'])
            r2 = linear_regression(df['obs_no2'], df['cooper_no2'])
            nrmse = Cal_NRMSE(df['cooper_no2'], df['obs_no2'])
            print(f"Slope: {reg_stats['slope']:.3f} ± {reg_stats['std_slope']:.3f}")
            print(f"Intercept: {reg_stats['intercept']:.3f} ± {reg_stats['std_intercept']:.3f}")
            print(f"R²: {r2:.3f}")
            print(f"NRMSE: {nrmse:.3f}")
            print(f"N: {len(df)}")

    if gchp_geo_df is not None:
        print("\nANNUAL GEOPHYSICAL COMPARISON:")
        print("-" * 40)
        df = gchp_geo_df
        valid = (df['region'] != 'unknown') & (df['obs_no2'] != -999) & (df['geophysical_no2'] != -999)
        df = df[valid]
        if not df.empty:
            reg_stats = regress2(df['obs_no2'], df['geophysical_no2'])
            r2 = linear_regression(df['obs_no2'], df['geophysical_no2'])
            nrmse = Cal_NRMSE(df['geophysical_no2'], df['obs_no2'])
            print(f"Slope: {reg_stats['slope']:.3f} ± {reg_stats['std_slope']:.3f}")
            print(f"Intercept: {reg_stats['intercept']:.3f} ± {reg_stats['std_intercept']:.3f}")
            print(f"R²: {r2:.3f}")
            print(f"NRMSE: {nrmse:.3f}")
            print(f"N: {len(df)}")

        print("\nANNUAL GCHP COMPARISON:")
        print("-" * 40)
        df = gchp_geo_df
        valid = (df['region'] != 'unknown') & (df['obs_no2'] != -999) & (df['gchp_no2'] != -999)
        df = df[valid]
        if not df.empty:
            reg_stats = regress2(df['obs_no2'], df['gchp_no2'])
            r2 = linear_regression(df['obs_no2'], df['gchp_no2'])
            nrmse = Cal_NRMSE(df['gchp_no2'], df['obs_no2'])
            print(f"Slope: {reg_stats['slope']:.3f} ± {reg_stats['std_slope']:.3f}")
            print(f"Intercept: {reg_stats['intercept']:.3f} ± {reg_stats['std_intercept']:.3f}")
            print(f"R²: {r2:.3f}")
            print(f"NRMSE: {nrmse:.3f}")
            print(f"N: {len(df)}")

# =========================
# Main
# =========================
def main():
    print("Starting NO2 evaluation and visualization (A4-ready)...")
    monthly_df, gchp_geo_df, cooper_df = load_and_process_data()
    if (monthly_df is None) and (gchp_geo_df is None) and (cooper_df is None):
        print("No data files found. Please run the data processing script first.")
        return
    create_method_by_region_figure(gchp_geo_df, cooper_df, a4_orientation='landscape')
    create_summary_statistics(monthly_df, gchp_geo_df, cooper_df)
    print(f"\nAll plots saved to: {output_dir}")
    print("Done!")

if __name__ == "__main__":
    main()
