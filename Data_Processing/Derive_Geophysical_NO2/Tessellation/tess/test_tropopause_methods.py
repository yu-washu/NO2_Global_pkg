"""
test_tropopause_methods.py

Compare three tropopause definitions for computing NO2_trop_gcshape:

  Method A (current): Use GC tropopause pressure (Met_TROPPT) for both
                      AvK zeroing and GC profile masking.

  Method B (TM5 trop): Use TM5 tropopause layer index for both
                        AvK zeroing and GC profile masking (via pressure
                        at the TM5 tropopause layer).

  Method C (harmonized): Interpolate GC profile onto TM5 levels FIRST,
                          then use TM5 tropopause layer index to define
                          the cutoff on the interpolated profile.

Usage:
  python test_tropopause_methods.py 2023 7 15
  python test_tropopause_methods.py 2023 1 15 --max-pixels 100000
"""
import os
import sys
import argparse
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── constants ──
Na = 6.022e23
MwAir = 28.97
MIN_LAT, MAX_LAT = -60, 70
MIN_LON, MAX_LON = -180, 180
SZA_MAX = 80
QA_LIM = 0.75

# ── paths (use same mount points as tess_TROPOMI.py) ──
GCHP_DIR = '/my-projects2/1.project/gchp-v2/forTessellation/{year}/daily/'
TROPOMI_L2_DIR = '/my-projects/1.project/TROPOMI_L2_V2_NO2_2018-2023/'
OUT_DIR = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess/diagnostics/'


def read_TROPOMI(filename):
    """Read TROPOMI L2, including tm5_tropopause_layer_index."""
    with nc.Dataset(filename, 'r') as ds:
        tm5a = ds['PRODUCT']['tm5_constant_a'][:]
        tm5b = ds['PRODUCT']['tm5_constant_b'][:]
        ps = ds['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['surface_pressure'][:]
        p_bottom = (tm5a[:, 0][np.newaxis, np.newaxis, np.newaxis, :]
                    + tm5b[:, 0][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])
        p_top = (tm5a[:, 1][np.newaxis, np.newaxis, np.newaxis, :]
                 + tm5b[:, 1][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])
        p = 0.5 * (p_bottom + p_top) * 0.01  # hPa

        AvKtot = ds['PRODUCT']['averaging_kernel'][:]
        AMFtot = ds['PRODUCT']['air_mass_factor_total'][:]
        AMFtrop = ds['PRODUCT']['air_mass_factor_troposphere'][:]
        trop_layer_idx = ds['PRODUCT']['tm5_tropopause_layer_index'][:]

        mf = ds['PRODUCT']['nitrogendioxide_tropospheric_column'].multiplication_factor_to_convert_to_molecules_percm2
        no2_trop_vc = ds['PRODUCT']['nitrogendioxide_tropospheric_column'][:] * mf
        no2_trop_sc = no2_trop_vc * AMFtrop

        mf2 = ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_total_column'].multiplication_factor_to_convert_to_molecules_percm2
        no2_tot_vc = ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_total_column'][:] * mf2
        no2_tot_sc = ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_slant_column_density'][:] * mf2

        qa = ds['PRODUCT']['qa_value'][:]
        lat = ds['PRODUCT']['latitude'][:]
        lon = ds['PRODUCT']['longitude'][:]
        sza = ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['solar_zenith_angle'][:]

    return {
        'p': p, 'AvKtot': AvKtot,
        'AMFtot': AMFtot, 'AMFtrop': AMFtrop,
        'trop_layer_idx': trop_layer_idx,
        'no2_trop_vc': no2_trop_vc, 'no2_trop_sc': no2_trop_sc,
        'no2_tot_vc': no2_tot_vc, 'no2_tot_sc': no2_tot_sc,
        'qa': qa, 'lat': lat, 'lon': lon, 'sza': sza,
    }


def compute_method_A(p_gc, no2prof, p_tm5, troppt_gc, AvKtot_pix, AMFtot, AMFtrop,
                     sc_trop, trop_lay_idx):
    """Current method: GC tropopause for everything."""
    # Interpolate full GC profile to TM5 levels
    interp_f = interp1d(p_gc, no2prof, bounds_error=False, fill_value='extrapolate')
    prof_tm5_tot = interp_f(p_tm5)

    # AvKtrop = AvKtot * AMFtot/AMFtrop, zero layers above GC tropopause
    scale = AMFtot / AMFtrop
    AvKtrop_raw = AvKtot_pix * scale
    AvKtrop = np.where(p_tm5 < troppt_gc, 0.0, AvKtrop_raw)

    # Mask GC profile at GC tropopause, interpolate to TM5
    no2prof_trop = np.where(p_gc >= troppt_gc, no2prof, 0.0)
    interp_f_trop = interp1d(p_gc, no2prof_trop, bounds_error=False, fill_value=0.0)
    prof_tm5_trop = np.where(p_tm5 >= troppt_gc, interp_f_trop(p_tm5), 0.0)

    sum_prof_trop = np.sum(prof_tm5_trop)
    if sum_prof_trop <= 0:
        return np.nan
    ratio = np.sum(prof_tm5_trop * AvKtrop) / sum_prof_trop
    amf_trop_gc = AMFtrop * ratio
    if amf_trop_gc <= 0 or np.isnan(amf_trop_gc):
        return np.nan
    return sc_trop / amf_trop_gc


def compute_method_B(p_gc, no2prof, p_tm5, troppt_gc, AvKtot_pix, AMFtot, AMFtrop,
                     sc_trop, trop_lay_idx):
    """TM5 tropopause for everything.
    Use TM5 trop_layer_idx to define the cutoff on TM5 levels.
    For GC profile masking, derive TM5 tropopause pressure from p_tm5[trop_lay_idx].
    """
    if trop_lay_idx < 0 or trop_lay_idx >= len(p_tm5):
        return np.nan

    # TM5 tropopause pressure: pressure at the tropopause layer
    troppt_tm5 = p_tm5[trop_lay_idx]

    # AvKtrop = AvKtot * AMFtot/AMFtrop, zero layers above TM5 tropopause
    scale = AMFtot / AMFtrop
    AvKtrop_raw = AvKtot_pix * scale
    # TM5 layer index convention: layers 0..trop_lay_idx-1 are troposphere
    # layers trop_lay_idx..33 are stratosphere
    layer_mask = np.arange(len(p_tm5)) < trop_lay_idx
    AvKtrop = np.where(layer_mask, AvKtrop_raw, 0.0)

    # Mask GC profile at TM5 tropopause pressure, interpolate to TM5
    no2prof_trop = np.where(p_gc >= troppt_tm5, no2prof, 0.0)
    interp_f_trop = interp1d(p_gc, no2prof_trop, bounds_error=False, fill_value=0.0)
    prof_tm5_trop = np.where(layer_mask, interp_f_trop(p_tm5), 0.0)

    sum_prof_trop = np.sum(prof_tm5_trop)
    if sum_prof_trop <= 0:
        return np.nan
    ratio = np.sum(prof_tm5_trop * AvKtrop) / sum_prof_trop
    amf_trop_gc = AMFtrop * ratio
    if amf_trop_gc <= 0 or np.isnan(amf_trop_gc):
        return np.nan
    return sc_trop / amf_trop_gc


def compute_method_C(p_gc, no2prof, p_tm5, troppt_gc, AvKtot_pix, AMFtot, AMFtrop,
                     sc_trop, trop_lay_idx):
    """Harmonized: interpolate GC profile onto TM5 levels first, then use
    TM5 trop_layer_idx to define the cutoff on the ALREADY INTERPOLATED profile.
    This avoids pressure-based masking ambiguity entirely.
    """
    if trop_lay_idx < 0 or trop_lay_idx >= len(p_tm5):
        return np.nan

    # Step 1: Interpolate full GC profile onto TM5 levels
    interp_f = interp1d(p_gc, no2prof, bounds_error=False, fill_value='extrapolate')
    prof_tm5_full = interp_f(p_tm5)
    prof_tm5_full = np.maximum(prof_tm5_full, 0.0)  # no negative partial columns

    # Step 2: Use TM5 layer index to define troposphere
    # layers 0..trop_lay_idx-1 are troposphere
    layer_mask = np.arange(len(p_tm5)) < trop_lay_idx
    prof_tm5_trop = np.where(layer_mask, prof_tm5_full, 0.0)

    # Step 3: AvKtrop with TM5 layer-based zeroing
    scale = AMFtot / AMFtrop
    AvKtrop_raw = AvKtot_pix * scale
    AvKtrop = np.where(layer_mask, AvKtrop_raw, 0.0)

    sum_prof_trop = np.sum(prof_tm5_trop)
    if sum_prof_trop <= 0:
        return np.nan
    ratio = np.sum(prof_tm5_trop * AvKtrop) / sum_prof_trop
    amf_trop_gc = AMFtrop * ratio
    if amf_trop_gc <= 0 or np.isnan(amf_trop_gc):
        return np.nan
    return sc_trop / amf_trop_gc


def main():
    parser = argparse.ArgumentParser(description='Compare 3 tropopause methods for NO2_trop_gcshape')
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)
    parser.add_argument('day', type=int)
    parser.add_argument('--max-pixels', type=int, default=50000)
    args = parser.parse_args()
    year, month, day = args.year, args.month, args.day

    os.makedirs(OUT_DIR, exist_ok=True)
    yyyymmdd = f"{year}{month:02d}{day:02d}"

    # ── Load GCHP ──
    gc_file = GCHP_DIR.format(year=year) + f'01x01.Hours.13-15.{yyyymmdd}.nc4'
    print(f"Loading GCHP: {gc_file}")
    import xarray as xr
    ds = xr.open_dataset(gc_file, engine='netcdf4')
    gc_lat = ds['lat'].values.astype('float32')
    gc_lon = ds['lon'].values.astype('float32')
    P_GC = ds['Met_PMIDDRY'].values.astype('float32')
    no2_gc = ds['SpeciesConcVV_NO2'].values.astype('float32')
    a = ds['Met_AIRDEN'].values.astype('float32') * 1e3
    b = ds['Met_BXHEIGHT'].values.astype('float32')
    troppt_gc_arr = ds['Met_TROPPT'].values.astype('float32')
    ds.close()
    partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na

    # ── Find TROPOMI files ──
    patterns = [f"S5P_RPRO_L2__NO2____{yyyymmdd}T", f"S5P_OFFL_L2__NO2____{yyyymmdd}T"]
    orbit_files = sorted([os.path.join(TROPOMI_L2_DIR, f)
                          for f in os.listdir(TROPOMI_L2_DIR)
                          if any(p in f for p in patterns)])
    print(f"Found {len(orbit_files)} TROPOMI orbits for {yyyymmdd}")

    # ── Collect results ──
    results = {
        'lat': [], 'lon': [],
        'no2_trop_orig': [],       # original TROPOMI VCD trop
        'no2_trop_A': [],          # method A (current, GC tropopause)
        'no2_trop_B': [],          # method B (TM5 tropopause)
        'no2_trop_C': [],          # method C (harmonized)
        'troppt_gc_hPa': [],
        'troppt_tm5_hPa': [],
        'trop_lay_idx': [],
    }

    n_collected = 0
    n_skip_nan = 0
    for orbit_file in orbit_files:
        if n_collected >= args.max_pixels:
            break
        print(f"  Processing {os.path.basename(orbit_file)} ...", end='', flush=True)
        try:
            trop = read_TROPOMI(orbit_file)
        except Exception as e:
            print(f" SKIP ({e})")
            continue

        good = (
            (trop['qa'][0] > QA_LIM) &
            (trop['sza'][0] < SZA_MAX) &
            (trop['lat'][0] >= MIN_LAT) & (trop['lat'][0] <= MAX_LAT) &
            (trop['lon'][0] >= MIN_LON) & (trop['lon'][0] <= MAX_LON) &
            np.isfinite(trop['no2_trop_vc'][0]) &
            np.isfinite(trop['no2_tot_vc'][0])
        )
        valid_idx = np.where(good)
        if len(valid_idx[0]) == 0:
            print(" 0 pixels")
            continue

        n_avail = len(valid_idx[0])
        n_want = min(n_avail, args.max_pixels - n_collected)
        if n_want < n_avail:
            sel = np.random.choice(n_avail, n_want, replace=False)
            valid_idx = (valid_idx[0][sel], valid_idx[1][sel])

        lat_idx = np.round(
            np.interp(trop['lat'][0][valid_idx], gc_lat, np.arange(len(gc_lat)))
        ).astype(int)
        lon_idx = np.round(
            np.interp(trop['lon'][0][valid_idx], gc_lon, np.arange(len(gc_lon)))
        ).astype(int)

        n_orbit = 0
        for ii in range(len(valid_idx[0])):
            j, k = valid_idx[0][ii], valid_idx[1][ii]

            p_gc = P_GC[:, lat_idx[ii], lon_idx[ii]]
            no2prof = partial_column[:, lat_idx[ii], lon_idx[ii]]
            p_tm5 = np.ma.filled(trop['p'][0, j, k, :], np.nan)
            troppt_gc = troppt_gc_arr[lat_idx[ii], lon_idx[ii]]
            trop_lay = int(trop['trop_layer_idx'][0, j, k])

            if np.isnan(p_gc).any() or np.isnan(no2prof).any() or np.isnan(p_tm5).any():
                n_skip_nan += 1
                continue

            AvKtot_pix = trop['AvKtot'][0, j, k, :]
            AMFtot_pix = float(trop['AMFtot'][0, j, k])
            AMFtrop_pix = float(trop['AMFtrop'][0, j, k])
            sc_trop = float(trop['no2_trop_sc'][0, j, k])

            if AMFtot_pix <= 0 or AMFtrop_pix <= 0:
                continue

            vA = compute_method_A(p_gc, no2prof, p_tm5, troppt_gc,
                                  AvKtot_pix, AMFtot_pix, AMFtrop_pix,
                                  sc_trop, trop_lay)
            vB = compute_method_B(p_gc, no2prof, p_tm5, troppt_gc,
                                  AvKtot_pix, AMFtot_pix, AMFtrop_pix,
                                  sc_trop, trop_lay)
            vC = compute_method_C(p_gc, no2prof, p_tm5, troppt_gc,
                                  AvKtot_pix, AMFtot_pix, AMFtrop_pix,
                                  sc_trop, trop_lay)

            if not (np.isfinite(vA) and np.isfinite(vB) and np.isfinite(vC)):
                continue

            # TM5 tropopause pressure
            troppt_tm5 = float(p_tm5[trop_lay]) if 0 <= trop_lay < len(p_tm5) else np.nan

            results['lat'].append(float(trop['lat'][0, j, k]))
            results['lon'].append(float(trop['lon'][0, j, k]))
            results['no2_trop_orig'].append(float(trop['no2_trop_vc'][0, j, k]))
            results['no2_trop_A'].append(vA)
            results['no2_trop_B'].append(vB)
            results['no2_trop_C'].append(vC)
            results['troppt_gc_hPa'].append(float(troppt_gc))
            results['troppt_tm5_hPa'].append(troppt_tm5)
            results['trop_lay_idx'].append(trop_lay)

            n_orbit += 1
            n_collected += 1

        print(f" {n_orbit} pixels (total: {n_collected})")

    # ── Convert to arrays ──
    for key in results:
        results[key] = np.array(results[key])
    n = len(results['lat'])
    print(f"\n{'='*70}")
    print(f"TROPOPAUSE METHOD COMPARISON: {yyyymmdd}  ({n} pixels, {n_skip_nan} skipped NaN)")
    print(f"{'='*70}")

    orig = results['no2_trop_orig']
    vA = results['no2_trop_A']
    vB = results['no2_trop_B']
    vC = results['no2_trop_C']

    # ── Relative bias vs original (%) ──
    bias_A = (vA - orig) / np.abs(orig) * 100
    bias_B = (vB - orig) / np.abs(orig) * 100
    bias_C = (vC - orig) / np.abs(orig) * 100

    # ── Correlation with original ──
    corr_A = np.corrcoef(orig, vA)[0, 1]
    corr_B = np.corrcoef(orig, vB)[0, 1]
    corr_C = np.corrcoef(orig, vC)[0, 1]

    # ── RMSE (absolute, molec/cm2) ──
    rmse_A = np.sqrt(np.nanmean((vA - orig)**2))
    rmse_B = np.sqrt(np.nanmean((vB - orig)**2))
    rmse_C = np.sqrt(np.nanmean((vC - orig)**2))

    # ── Normalized Mean Bias (NMB) ──
    nmb_A = np.nansum(vA - orig) / np.nansum(orig) * 100
    nmb_B = np.nansum(vB - orig) / np.nansum(orig) * 100
    nmb_C = np.nansum(vC - orig) / np.nansum(orig) * 100

    print(f"\n--- Statistics vs original TROPOMI tropospheric VCD ---")
    print(f"{'Method':<20s} {'R':>8s} {'NMB(%)':>10s} {'RMSE(1e15)':>12s} "
          f"{'MeanBias(%)':>12s} {'StdBias(%)':>12s}")
    print(f"{'A (GC trop)':<20s} {corr_A:8.5f} {nmb_A:10.2f} {rmse_A/1e15:12.3f} "
          f"{np.nanmean(bias_A):12.2f} {np.nanstd(bias_A):12.2f}")
    print(f"{'B (TM5 trop)':<20s} {corr_B:8.5f} {nmb_B:10.2f} {rmse_B/1e15:12.3f} "
          f"{np.nanmean(bias_B):12.2f} {np.nanstd(bias_B):12.2f}")
    print(f"{'C (harmonized)':<20s} {corr_C:8.5f} {nmb_C:10.2f} {rmse_C/1e15:12.3f} "
          f"{np.nanmean(bias_C):12.2f} {np.nanstd(bias_C):12.2f}")

    # ── Tropopause mismatch statistics ──
    trop_diff = results['troppt_gc_hPa'] - results['troppt_tm5_hPa']
    valid_td = np.isfinite(trop_diff)
    print(f"\n--- Tropopause: GC - TM5 (hPa) ---")
    print(f"  mean={np.nanmean(trop_diff):.1f}  std={np.nanstd(trop_diff):.1f}  "
          f"median={np.nanmedian(trop_diff):.1f}")

    # ── Subgroup: large tropopause mismatch (|diff| > 50 hPa) ──
    big_diff = np.abs(trop_diff) > 50
    n_big = np.sum(big_diff & valid_td)
    print(f"\n--- Subset where |tropopause diff| > 50 hPa ({n_big} pixels, "
          f"{100*n_big/n:.1f}%) ---")
    if n_big > 10:
        print(f"  {'Method':<20s} {'MeanBias(%)':>12s} {'StdBias(%)':>12s}")
        print(f"  {'A (GC trop)':<20s} {np.nanmean(bias_A[big_diff]):12.2f} "
              f"{np.nanstd(bias_A[big_diff]):12.2f}")
        print(f"  {'B (TM5 trop)':<20s} {np.nanmean(bias_B[big_diff]):12.2f} "
              f"{np.nanstd(bias_B[big_diff]):12.2f}")
        print(f"  {'C (harmonized)':<20s} {np.nanmean(bias_C[big_diff]):12.2f} "
              f"{np.nanstd(bias_C[big_diff]):12.2f}")

    # ── PLOTS ──
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f'Tropopause Method Comparison — {yyyymmdd}  (N={n})', fontsize=14, y=0.98)

    vmax_sc = np.nanpercentile(orig, 99)

    # Row 1: scatter plots of each method vs original
    for col, (vals, label, bias, corr, nmb_val, rmse_val) in enumerate([
        (vA, 'A: GC tropopause', bias_A, corr_A, nmb_A, rmse_A),
        (vB, 'B: TM5 tropopause', bias_B, corr_B, nmb_B, rmse_B),
        (vC, 'C: Harmonized', bias_C, corr_C, nmb_C, rmse_C),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(orig / 1e15, vals / 1e15, s=1, alpha=0.2, rasterized=True)
        lim = max(np.nanpercentile(orig / 1e15, 99.5), np.nanpercentile(vals / 1e15, 99.5))
        ax.plot([0, lim], [0, lim], 'k--', lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel('Original VCD_trop (10$^{15}$ molec/cm$^2$)')
        ax.set_ylabel(f'{label} (10$^{{15}}$ molec/cm$^2$)')
        ax.set_title(f'{label}\nR={corr:.4f}  NMB={nmb_val:.1f}%  RMSE={rmse_val/1e15:.2f}')
        ax.set_aspect('equal')

    # Row 2: histograms of relative bias
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-100, 100, 80)
    ax.hist(bias_A, bins=bins, alpha=0.5, label=f'A (std={np.nanstd(bias_A):.1f}%)', density=True)
    ax.hist(bias_B, bins=bins, alpha=0.5, label=f'B (std={np.nanstd(bias_B):.1f}%)', density=True)
    ax.hist(bias_C, bins=bins, alpha=0.5, label=f'C (std={np.nanstd(bias_C):.1f}%)', density=True)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Relative bias vs original (%)')
    ax.set_ylabel('Density')
    ax.set_title('Bias distribution')
    ax.legend(fontsize=8)

    # Row 2: tropopause scatter
    ax = fig.add_subplot(gs[1, 1])
    valid_t = np.isfinite(results['troppt_tm5_hPa'])
    ax.scatter(results['troppt_tm5_hPa'][valid_t], results['troppt_gc_hPa'][valid_t],
               s=1, alpha=0.2, rasterized=True)
    ax.plot([100, 500], [100, 500], 'k--', lw=1)
    ax.set_xlabel('TM5 tropopause (hPa)')
    ax.set_ylabel('GC tropopause (hPa)')
    ax.set_title(f'Tropopause: GC vs TM5\nmean diff={np.nanmean(trop_diff):.1f} hPa')
    ax.set_xlim(100, 500)
    ax.set_ylim(100, 500)

    # Row 2: bias vs tropopause difference
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(trop_diff[valid_td], bias_A[valid_td], s=1, alpha=0.15, label='A', rasterized=True)
    ax.scatter(trop_diff[valid_td], bias_C[valid_td], s=1, alpha=0.15, label='C', rasterized=True)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Tropopause diff: GC − TM5 (hPa)')
    ax.set_ylabel('Relative bias vs original (%)')
    ax.set_title('Does harmonizing reduce\ntropopause-driven noise?')
    ax.set_ylim(-100, 100)
    ax.legend(fontsize=8, markerscale=5)

    # Row 3: difference maps (method - original)
    for col, (vals, label) in enumerate([
        (vA - orig, 'A − orig'),
        (vB - orig, 'B − orig'),
        (vC - orig, 'C − orig'),
    ]):
        ax = fig.add_subplot(gs[2, col])
        lat_arr = results['lat']
        lon_arr = results['lon']
        sc = ax.scatter(lon_arr, lat_arr, c=vals / 1e15, s=0.5, alpha=0.5,
                        cmap='RdBu_r', vmin=-2, vmax=2, rasterized=True)
        plt.colorbar(sc, ax=ax, label='$\\Delta$ VCD (10$^{15}$)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{label}')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 70)

    outpng = os.path.join(OUT_DIR, f'tropopause_methods_{yyyymmdd}.png')
    plt.savefig(outpng, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpng}")

    # ── Save CSV ──
    import pandas as pd
    df = pd.DataFrame(results)
    df['bias_A_pct'] = bias_A
    df['bias_B_pct'] = bias_B
    df['bias_C_pct'] = bias_C
    outcsv = os.path.join(OUT_DIR, f'tropopause_methods_{yyyymmdd}.csv')
    df.to_csv(outcsv, index=False)
    print(f"Saved: {outcsv}")

    # ── Final recommendation ──
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    methods = {'A (GC trop)': (corr_A, nmb_A, rmse_A, np.nanstd(bias_A)),
               'B (TM5 trop)': (corr_B, nmb_B, rmse_B, np.nanstd(bias_B)),
               'C (harmonized)': (corr_C, nmb_C, rmse_C, np.nanstd(bias_C))}
    # Rank by: highest R, lowest |NMB|, lowest RMSE, lowest bias std
    best_R = max(methods, key=lambda m: methods[m][0])
    best_NMB = min(methods, key=lambda m: abs(methods[m][1]))
    best_RMSE = min(methods, key=lambda m: methods[m][2])
    best_std = min(methods, key=lambda m: methods[m][3])
    print(f"  Best R:        {best_R}")
    print(f"  Best |NMB|:    {best_NMB}")
    print(f"  Best RMSE:     {best_RMSE}")
    print(f"  Best bias std: {best_std}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
