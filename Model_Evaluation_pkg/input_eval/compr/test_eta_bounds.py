"""
test_eta_bounds.py — Sweep eta_trop clipping bounds and report R² for
geophysical_no2_trop recomputed from station-level monthly data.

Uses the monthly CSV from compr_geo_gchp_global_v53.py which has per-station:
  gchp_eta_trop, gap_SatColNO2_trop_gcshape, obs_no2

For each (eta_min, eta_max) pair:
  1. Clip gchp_eta_trop to [eta_min, eta_max]  (NaN outside)
  2. Recompute: geo_trop = gap_SatColNO2_trop_gcshape × clipped_eta_trop
  3. Average per station (annual)
  4. Report R² vs obs_no2

NOTE: This approximates the daily clipping (v5.3 clips daily eta then averages),
but monthly-level clipping gives a fast directional signal.

Usage: python3 test_eta_bounds.py
"""
import numpy as np
import pandas as pd
from scipy import stats

# ── Config ──
YEAR = 2023
obs_version = 'v6'
geono2_version = 'v5.3'
DATA_DIR = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
MONTHLY_FILE = DATA_DIR + f'NO2_monthly_{YEAR}_obs{obs_version}_geono2-{geono2_version}.csv'

# ── Sweep grid (fine on upper bound, coarse on lower since it doesn't matter) ──
ETA_MINS = [0, 1e-17, 5e-17, 1e-16]
ETA_MAXS = [5e-15, 3e-15, 2.5e-15, 2e-15, 1.8e-15, 1.5e-15, 1.2e-15, 1e-15, 8e-16, 6e-16]


def compute_r2(obs, pred):
    """R² from linear regression, dropping NaN."""
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 10:
        return np.nan, np.nan, 0
    slope, intercept, r, p, se = stats.linregress(obs[mask], pred[mask])
    return r**2, slope, int(mask.sum())


def main():
    print(f"Loading {MONTHLY_FILE}...")
    df = pd.read_csv(MONTHLY_FILE)
    print(f"  {len(df)} monthly rows")

    # Current v5.3 baseline (already clipped at [1e-17, 2e-15] daily)
    annual = df.groupby(['px_lat', 'px_lon']).agg(
        obs_no2=('obs_no2', 'mean'),
        geo_trop=('geophysical_no2_trop', 'mean'),
        geo_trop_TM5=('geophysical_no2_trop_TM5', 'mean'),
        geo_tot=('geophysical_no2_tot', 'mean'),
        sat_trop=('gap_SatColNO2_trop', 'mean'),
        sat_trop_gcshape=('gap_SatColNO2_trop_gcshape', 'mean'),
        sat_tot=('gap_SatColNO2_tot', 'mean'),
        sat_tot_gcshape=('gap_SatColNO2_tot_gcshape', 'mean'),
        eta_trop=('gchp_eta_trop', 'mean'),
        eta_tot=('gchp_eta_tot', 'mean'),
    ).reset_index()

    r2_base, slope_base, n = compute_r2(annual['obs_no2'].values, annual['geo_trop'].values)
    r2_tm5, slope_tm5, _ = compute_r2(annual['obs_no2'].values, annual['geo_trop_TM5'].values)
    r2_tot, slope_tot, _ = compute_r2(annual['obs_no2'].values, annual['geo_tot'].values)
    r2_sat, slope_sat, _ = compute_r2(annual['obs_no2'].values, annual['sat_trop'].values)
    r2_sat_gcshape, slope_sat_gcshape, _ = compute_r2(annual['obs_no2'].values, annual['sat_trop_gcshape'].values)
    r2_sat_tot, slope_sat_tot, _ = compute_r2(annual['obs_no2'].values, annual['sat_tot'].values)
    r2_sat_tot_gcshape, slope_sat_tot_gcshape, _ = compute_r2(annual['obs_no2'].values, annual['sat_tot_gcshape'].values)
    r2_eta_trop, slope_eta_trop, _ = compute_r2(annual['obs_no2'].values, annual['eta_trop'].values)
    r2_eta_tot, slope_eta_tot, _ = compute_r2(annual['obs_no2'].values, annual['eta_tot'].values)

    print(f"\n{'='*70}")
    print(f"BASELINES (from v5.3 output, N={n})")
    print(f"{'='*70}")
    print(f"  geophysical_no2_trop      R²={r2_base:.4f}  slope={slope_base:.3f}")
    print(f"  geophysical_no2_trop_TM5  R²={r2_tm5:.4f}  slope={slope_tm5:.3f}")
    print(f"  geophysical_no2_tot       R²={r2_tot:.4f}  slope={slope_tot:.3f}")
    print(f"  gap_SatColNO2_trop        R²={r2_sat:.4f}  slope={slope_sat:.3f}")
    print(f"  gap_SatColNO2_trop_gcshape R²={r2_sat_gcshape:.4f}  slope={slope_sat_gcshape:.3f}")
    print(f"  gap_SatColNO2_tot        R²={r2_sat_tot:.4f}  slope={slope_sat_tot:.3f}")
    print(f"  gap_SatColNO2_tot_gcshape R²={r2_sat_tot_gcshape:.4f}  slope={slope_sat_tot_gcshape:.3f}")
    print(f"  gchp_eta_trop            R²={r2_eta_trop:.4f}  slope={slope_eta_trop:.3f}")
    print(f"  gchp_eta_tot            R²={r2_eta_tot:.4f}  slope={slope_eta_tot:.3f}")

    # Sweep: recompute geo_trop from monthly eta × SatCol with different bounds
    print(f"\n{'='*70}")
    print(f"ETA_TROP BOUNDS SWEEP (monthly-level clipping approximation)")
    print(f"{'='*70}")
    print(f"{'eta_min':>12s}  {'eta_max':>12s}  "
          f"{'geo_R²':>8s}  {'geo_slp':>8s}  "
          f"{'eta_R²':>8s}  {'eta_slp':>8s}  "
          f"{'N_ann':>6s}  {'%_clip':>7s}")
    print(f"{'-'*12}  {'-'*12}  "
          f"{'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*7}")

    best_r2 = 0
    best_params = None

    for eta_min in ETA_MINS:
        for eta_max in ETA_MAXS:
            if eta_min >= eta_max:
                continue

            # Clip eta_trop at monthly level
            eta = df['gchp_eta_trop'].values.copy()
            clipped = np.where((eta >= eta_min) & (eta <= eta_max), eta, np.nan)
            n_total = int(np.isfinite(eta).sum())
            n_clip = n_total - int(np.isfinite(clipped).sum())
            pct_clip = 100 * n_clip / n_total if n_total > 0 else 0

            # Recompute geophysical_no2_trop = SatCol × clipped_eta
            sat_col = df['gap_SatColNO2_trop_gcshape'].values
            geo_recomp = sat_col * clipped

            # Build temp df for annual averaging
            tmp = df[['px_lat', 'px_lon', 'obs_no2']].copy()
            tmp['geo_recomp'] = geo_recomp
            tmp['eta_clipped'] = clipped
            ann = tmp.groupby(['px_lat', 'px_lon']).agg(
                obs_no2=('obs_no2', 'mean'),
                geo_recomp=('geo_recomp', 'mean'),
                eta_clipped=('eta_clipped', 'mean'),
            ).reset_index()

            r2_geo, slope_geo, n_ann = compute_r2(ann['obs_no2'].values, ann['geo_recomp'].values)
            r2_eta, slope_eta, _     = compute_r2(ann['obs_no2'].values, ann['eta_clipped'].values)

            marker = ""
            if r2_geo > best_r2:
                best_r2 = r2_geo
                best_params = (eta_min, eta_max)
                marker = " <-- best"

            print(f"{eta_min:12.2e}  {eta_max:12.2e}  "
                  f"{r2_geo:8.4f}  {slope_geo:8.3f}  "
                  f"{r2_eta:8.4f}  {slope_eta:8.3f}  "
                  f"{n_ann:6d}  {pct_clip:6.1f}%{marker}")

    print(f"\n{'='*70}")
    print(f"BEST: eta_min={best_params[0]:.2e}, eta_max={best_params[1]:.2e} → geo_R²={best_r2:.4f}")
    print(f"(vs v5.3 baseline R²={r2_base:.4f}, vs SatColNO2_trop R²={r2_sat:.4f})")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
