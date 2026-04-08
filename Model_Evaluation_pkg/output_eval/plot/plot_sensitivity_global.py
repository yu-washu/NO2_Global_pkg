"""
Global-Only Sensitivity Bar Chart
──────────────────────────────────
Reads baseline and exclusion-test CSV files, computes percentage change in
R² and RMSE for the Global region only, and produces a single horizontal
grouped bar chart with both metrics side by side.

Usage:
    python plot_sensitivity_global.py
"""

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# %matplotlib inline
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
DISPLAY_NAMES = {'GeoNO2v513_GC': 'GeoNO2', 'GeoNo2': 'GeoNO2'}

# Variables to skip in sensitivity analysis (not part of the model channels)
SKIP_VARS = {'SatColNO2'}

STAT_DIR = (
    '/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/'
    'NO2/v4.1.0/Results/results-SpatialCV/statistical_indicators'
)

BASELINE_DIR_NAME = 'Absolute-NO2_NO2_v4.1.0_26Channel_5x5_Geov5131-abs'

YEAR = 2023

REGION = 'Global'           # single region for this plot

FIG_DIR = (
    '/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/'
    'NO2/v4.1.0/Figures/figures-Sensitivity/'
)

# ═══════════════════════════════════════════════════════════════════════════════
# PARSER  (reused from plot_sensitivity.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_value(line, key):
    """Return the float that follows *key* (after a comma) in *line*."""
    idx = line.find(key)
    if idx == -1:
        return np.nan
    after = line[idx + len(key):]
    m = re.search(r',\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', after)
    return float(m.group(1)) if m else np.nan


def parse_csv(filepath, regions, season='Annual'):
    """
    Parse one statistical-indicator CSV and return
        { region: { 'Test R2': float, 'RMSE': float } }
    for the requested *season*.
    """
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    data = {}
    current_region = None
    current_season = None

    for raw in lines:
        line = raw.strip()

        if line.startswith('Area:'):
            current_region = None
            for r in regions:
                if r in line:
                    current_region = r
                    break
            current_season = None
            continue

        if '----------' in line:
            if f' {season} ' in line or f' {season} -' in line:
                current_season = season
            else:
                current_season = None
            continue

        if current_region is None or current_season != season:
            continue

        if current_region not in data:
            data[current_region] = {}

        if 'Test R2' in line and 'Training' not in line:
            if season == 'Annual':
                val = _extract_value(line, 'AllPoints Test R2 - AllPoints:')
                if np.isnan(val):
                    val = _extract_value(line, 'AllPoints Test R2 - Annual Average:')
            else:
                val = _extract_value(line, 'AllPoints Test R2 - AllPoints:')
            data[current_region]['Test R2'] = val

        elif ' RMSE ' in line or line.lstrip().startswith('RMSE'):
            if 'NRMSE' not in line and 'PWM' not in line:
                if season == 'Annual':
                    val = _extract_value(line, 'AllPoints RMSE - AllPoints:')
                    if np.isnan(val):
                        val = _extract_value(line, 'AllPoints RMSE - Annual Average:')
                else:
                    val = _extract_value(line, 'AllPoints RMSE - AllPoints:')
                data[current_region]['RMSE'] = val

    return data


def discover_excl_dirs(stat_dir):
    """Find exclusion dirs, skipping variables in SKIP_VARS."""
    results = []
    for entry in sorted(os.listdir(stat_dir)):
        full = os.path.join(stat_dir, entry)
        if os.path.isdir(full) and '-abs_excl_' in entry:
            var_name = entry.split('-abs_excl_')[-1]
            if var_name in SKIP_VARS:
                continue
            results.append((var_name, entry))
    return results


def find_csv_in_dir(dir_path, year):
    pattern = os.path.join(dir_path, f'*{year}-{year}*.csv')
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    csvs = glob.glob(os.path.join(dir_path, '*.csv'))
    return csvs[0] if csvs else None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    regions = [REGION]

    # ── 1. Parse baseline ──────────────────────────────────────────────────
    baseline_csv = find_csv_in_dir(
        os.path.join(STAT_DIR, BASELINE_DIR_NAME), YEAR
    )
    if baseline_csv is None:
        raise FileNotFoundError('No baseline CSV found')
    print(f'Baseline: {baseline_csv}')
    baseline = parse_csv(baseline_csv, regions)

    base_r2   = baseline[REGION]['Test R2']
    base_rmse = baseline[REGION]['RMSE']
    print(f'  Baseline  R²  = {base_r2:.4f}')
    print(f'  Baseline RMSE = {base_rmse:.2f}')

    # ── 2. Parse every exclusion experiment ────────────────────────────────
    excl_list = discover_excl_dirs(STAT_DIR)
    if not excl_list:
        raise RuntimeError(f'No excl_* directories found in {STAT_DIR}')

    variable_names = []
    r2_pct  = []
    rmse_pct = []

    for var_name, dir_name in excl_list:
        csv_path = find_csv_in_dir(os.path.join(STAT_DIR, dir_name), YEAR)
        if csv_path is None:
            print(f'  [SKIP] {dir_name}')
            continue

        d = parse_csv(csv_path, regions)
        excl_r2   = d.get(REGION, {}).get('Test R2', np.nan)
        excl_rmse = d.get(REGION, {}).get('RMSE', np.nan)

        dr2   = (excl_r2   - base_r2)   / abs(base_r2)   * 100 if base_r2   else np.nan
        drmse = (excl_rmse - base_rmse)  / abs(base_rmse) * 100 if base_rmse else np.nan

        variable_names.append(DISPLAY_NAMES.get(var_name, var_name))
        r2_pct.append(dr2)
        rmse_pct.append(drmse)

        print(f'  Excl {var_name:<20s}  dR²={dr2:+.2f}%   dRMSE={drmse:+.2f}%')

    r2_pct   = np.array(r2_pct)
    rmse_pct = np.array(rmse_pct)

    # ── 3. Sort variables by absolute R² impact (largest first) ────────────
    sort_idx = np.argsort(-np.abs(r2_pct))
    variable_names = [variable_names[i] for i in sort_idx]
    r2_pct   = r2_pct[sort_idx]
    rmse_pct = rmse_pct[sort_idx]

    # ── 4. Keep top 10, aggregate the rest ─────────────────────────────────
    TOP_N = 10
    if len(variable_names) > TOP_N:
        n_rest = len(variable_names) - TOP_N
        top_names = variable_names[:TOP_N]
        top_r2    = r2_pct[:TOP_N]
        top_rmse  = rmse_pct[:TOP_N]

        rest_r2   = np.nansum(r2_pct[TOP_N:])
        rest_rmse = np.nansum(rmse_pct[TOP_N:])

        top_names.append(f'Sum of {n_rest} other features')
        top_r2    = np.append(top_r2,   rest_r2)
        top_rmse  = np.append(top_rmse, rest_rmse)

        variable_names = top_names
        r2_pct   = top_r2
        rmse_pct = top_rmse

    # ── 5. Plot ────────────────────────────────────────────────────────────
    plot_global_sensitivity(variable_names, r2_pct, rmse_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_global_sensitivity(variables, r2_pct, rmse_pct):
    """
    Vertical grouped bar chart — x-axis = variables, y-axis = % change.
        blue  = R² % change   (negative → variable helps R²)
        red   = RMSE % change (positive → variable reduces error)
    """
    n = len(variables)
    bar_w   = 0.35                       # width of each bar
    x_pos   = np.arange(n)               # centre positions

    fig_width = max(8, 0.65 * n + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # ── bars ───────────────────────────────────────────────────────────
    bars_r2 = ax.bar(
        x_pos - bar_w / 2, r2_pct, width=bar_w,
        color='#4393C3', edgecolor='white', linewidth=0.5,
        label=r'$\Delta R^2$ (%)',
    )
    bars_rmse = ax.bar(
        x_pos + bar_w / 2, rmse_pct, width=bar_w,
        color='#D6604D', edgecolor='white', linewidth=0.5,
        label=r'$\Delta$RMSE (%)',
    )

    # ── value labels on each bar ───────────────────────────────────────
    for bar_set in [bars_r2, bars_rmse]:
        for bar in bar_set:
            h = bar.get_height()
            if np.isnan(h):
                continue
            va = 'bottom' if h >= 0 else 'top'
            offset = 0.03 * max(np.nanmax(np.abs(r2_pct)),
                                np.nanmax(np.abs(rmse_pct)))
            y_txt = h + offset if h >= 0 else h - offset
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_txt,
                f'{h:+.2f}%', ha='center', va=va, fontsize=11
            )

    # ── zero line ──────────────────────────────────────────────────────
    ax.axhline(0, color='black', linewidth=0.8)

    # ── axes styling ───────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=14)
    ax.set_ylim(-1.5, 3.0)
    ax.set_yticks([-3.0, -1.5, 0, 1.5, 3.0, 4.5])
    ax.set_yticklabels(['-3.0%', '-1.5%', '0.0%', '1.5%', '3.0%', '4.5%'], fontsize=12)
    ax.set_ylabel('% change with variable excluded', fontsize=15)
    ax.legend(
        loc='upper center', fontsize=12, frameon=True,
        fancybox=True, shadow=False,
    )
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    # ── save ───────────────────────────────────────────────────────────
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(
        FIG_DIR, f'Sensitivity_Global_R2_RMSE_{YEAR}.png'
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure saved -> {out_path}')


if __name__ == '__main__':
    main()
