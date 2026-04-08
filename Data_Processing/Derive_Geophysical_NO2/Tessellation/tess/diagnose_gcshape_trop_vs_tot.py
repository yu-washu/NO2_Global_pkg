"""
diagnose_gcshape_trop_vs_tot.py

Diagnostic script to investigate why GC shape factor improves total column
but not tropospheric column.

For a sample day, it:
  1. Loads TROPOMI L2 and GCHP profiles
  2. Computes amf_gcshape for both total and trop
  3. Compares the AMF correction ratio (amf_gcshape / amf_original)
  4. Quantifies profile shape differences between GC and TM5 in trop vs total
  5. Checks tropopause consistency (GC vs TM5)

Usage:
  python diagnose_gcshape_trop_vs_tot.py 2023 7 15
"""
import os
import sys
import argparse
import numpy as np
import xarray as xr
import netCDF4 as nc
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── constants ──
Na = 6.022e23
MwAir = 28.97
MIN_LAT, MAX_LAT = -60, 70
MIN_LON, MAX_LON = -180, 180
SZA_MAX = 80
QA_LIM = 0.75

# ── paths ──
GCHP_DIR = '/my-projects2/1.project/gchp-v2/forTessellation/{year}/daily/'
TROPOMI_L2_DIR = '/my-projects/1.project/TROPOMI_L2_V2_NO2_2018-2023/'
OUT_DIR = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess/diagnostics/'

# ── I/O helpers (from Tess_func.py) ──
def read_TROPOMI(filename):
    with nc.Dataset(filename, 'r') as ds:
        tm5a = ds['PRODUCT']['tm5_constant_a'][:]
        tm5b = ds['PRODUCT']['tm5_constant_b'][:]
        ps = ds['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['surface_pressure'][:]
        p_bottom = tm5a[:, 0][np.newaxis, np.newaxis, np.newaxis, :] + \
                   tm5b[:, 0][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis]
        p_top = tm5a[:, 1][np.newaxis, np.newaxis, np.newaxis, :] + \
                tm5b[:, 1][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis]
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

        # AvKtrop = AvKtot * AMFtot/AMFtrop (ATBD 6.4.5)
        scale = (AMFtot / AMFtrop)[..., np.newaxis]
        AvKtrop = AvKtot * scale

        qa = ds['PRODUCT']['qa_value'][:]
        lat = ds['PRODUCT']['latitude'][:]
        lon = ds['PRODUCT']['longitude'][:]
        sza = ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['solar_zenith_angle'][:]

    return {
        'p': p, 'AvKtot': AvKtot, 'AvKtrop': AvKtrop,
        'AMFtot': AMFtot, 'AMFtrop': AMFtrop,
        'trop_layer_idx': trop_layer_idx,
        'no2_trop_vc': no2_trop_vc, 'no2_trop_sc': no2_trop_sc,
        'no2_tot_vc': no2_tot_vc, 'no2_tot_sc': no2_tot_sc,
        'qa': qa, 'lat': lat, 'lon': lon, 'sza': sza,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    parser.add_argument('month', type=int)
    parser.add_argument('day', type=int)
    parser.add_argument('--max-pixels', type=int, default=50000,
                        help='Max pixels to sample (default: 50000)')
    args = parser.parse_args()
    year, month, day = args.year, args.month, args.day

    os.makedirs(OUT_DIR, exist_ok=True)
    yyyymmdd = f"{year}{month:02d}{day:02d}"

    # ── Load GCHP ──
    gc_file = GCHP_DIR.format(year=year) + f'01x01.Hours.13-15.{yyyymmdd}.nc4'
    print(f"Loading GCHP: {gc_file}")
    ds = xr.open_dataset(gc_file, engine='netcdf4')
    gc_lat = ds['lat'].values.astype('float32')
    gc_lon = ds['lon'].values.astype('float32')
    P_GC = ds['Met_PMIDDRY'].values.astype('float32')
    no2_gc = ds['SpeciesConcVV_NO2'].values.astype('float32')
    a = ds['Met_AIRDEN'].values.astype('float32') * 1e3
    b = ds['Met_BXHEIGHT'].values.astype('float32')
    troppt_gc = ds['Met_TROPPT'].values.astype('float32')
    ds.close()
    partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na

    # ── Find TROPOMI files (flat directory, all years) ──
    l2_dir = TROPOMI_L2_DIR
    patterns = [f"S5P_RPRO_L2__NO2____{yyyymmdd}T", f"S5P_OFFL_L2__NO2____{yyyymmdd}T"]
    orbit_files = sorted([os.path.join(l2_dir, f) for f in os.listdir(l2_dir)
                          if any(p in f for p in patterns)])
    print(f"Found {len(orbit_files)} TROPOMI orbits for {yyyymmdd}")

    # ── Collect per-pixel diagnostics ──
    results = {
        'lat': [], 'lon': [],
        'amf_tot_orig': [], 'amf_tot_gc': [],
        'amf_trop_orig': [], 'amf_trop_gc': [],
        'ratio_tot': [],       # amf_tot_gc / amf_tot_orig  (= sum(prof*AvK)/sum(prof) for total)
        'ratio_trop': [],      # same for trop
        'troppt_gc_hPa': [],
        'troppt_tm5_hPa': [],  # TM5 tropopause from L2 layer index
        'n_trop_layers_tm5': [],
        'n_trop_layers_gc': [],
        'no2_trop_vc': [],     # original VCD trop
        'no2_tot_vc': [],      # original VCD total
        'no2_trop_gcshape': [],
        'no2_tot_gcshape': [],
        'frac_trop_gc': [],    # fraction of total GC column that's tropospheric
        'profile_corr_trop': [],  # correlation of GC trop shape with AvKtrop weighting
    }

    n_collected = 0
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

        # Subsample if needed
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
            si, gi = j, k  # scanline, ground_pixel

            p_gc = P_GC[:, lat_idx[ii], lon_idx[ii]]
            no2prof = partial_column[:, lat_idx[ii], lon_idx[ii]]
            p_tm5 = np.ma.filled(trop['p'][0, si, gi, :], np.nan)

            if np.isnan(p_gc).any() or np.isnan(no2prof).any() or np.isnan(p_tm5).any():
                continue

            troppt_val = troppt_gc[lat_idx[ii], lon_idx[ii]]

            # TM5 tropopause from L2 layer index
            trop_lay = int(trop['trop_layer_idx'][0, j, k])
            if 0 <= trop_lay < 34:
                troppt_tm5 = float(p_tm5[trop_lay])
            else:
                troppt_tm5 = np.nan

            # ── Total column GC shape ──
            interp_f = interp1d(p_gc, no2prof, bounds_error=False, fill_value='extrapolate')
            prof_tm5_tot = interp_f(p_tm5)

            AvKtot_pix = trop['AvKtot'][0, si, gi, :]
            AMFtot_pix = float(trop['AMFtot'][0, si, gi])
            AMFtrop_pix = float(trop['AMFtrop'][0, si, gi])

            sum_prof_tot = np.sum(prof_tm5_tot)
            if sum_prof_tot <= 0 or AMFtot_pix <= 0 or AMFtrop_pix <= 0:
                continue

            ratio_tot = np.sum(prof_tm5_tot * AvKtot_pix) / sum_prof_tot
            amf_tot_gc = AMFtot_pix * ratio_tot

            sc_tot = float(trop['no2_tot_sc'][0, si, gi])
            no2_tot_gcshape = sc_tot / amf_tot_gc if amf_tot_gc else np.nan

            # ── Tropospheric column GC shape ──
            AvKtrop_raw = trop['AvKtrop'][0, si, gi, :]
            AvKtrop_zeroed = np.where(p_tm5 < troppt_val, 0.0, AvKtrop_raw)

            no2prof_trop = np.where(p_gc >= troppt_val, no2prof, 0.0)
            sum_trop_gc = np.sum(no2prof_trop)
            if sum_trop_gc <= 0:
                continue

            interp_f_trop = interp1d(p_gc, no2prof_trop, bounds_error=False, fill_value=0.0)
            prof_tm5_trop = np.where(p_tm5 >= troppt_val, interp_f_trop(p_tm5), 0.0)
            sum_prof_trop = np.sum(prof_tm5_trop)
            if sum_prof_trop <= 0:
                continue

            ratio_trop = np.sum(prof_tm5_trop * AvKtrop_zeroed) / sum_prof_trop
            amf_trop_gc = AMFtrop_pix * ratio_trop

            sc_trop = float(trop['no2_trop_sc'][0, si, gi])
            no2_trop_gcshape = sc_trop / amf_trop_gc if amf_trop_gc else np.nan

            if not np.isfinite(no2_trop_gcshape) or not np.isfinite(no2_tot_gcshape):
                continue

            # ── Record ──
            results['lat'].append(float(trop['lat'][0, j, k]))
            results['lon'].append(float(trop['lon'][0, j, k]))
            results['amf_tot_orig'].append(AMFtot_pix)
            results['amf_tot_gc'].append(amf_tot_gc)
            results['amf_trop_orig'].append(AMFtrop_pix)
            results['amf_trop_gc'].append(amf_trop_gc)
            results['ratio_tot'].append(ratio_tot)
            results['ratio_trop'].append(ratio_trop)
            results['troppt_gc_hPa'].append(float(troppt_val))
            results['troppt_tm5_hPa'].append(troppt_tm5)
            results['n_trop_layers_tm5'].append(int(np.sum(p_tm5 >= troppt_val)))
            results['n_trop_layers_gc'].append(int(np.sum(p_gc >= troppt_val)))
            results['no2_trop_vc'].append(float(trop['no2_trop_vc'][0, j, k]))
            results['no2_tot_vc'].append(float(trop['no2_tot_vc'][0, j, k]))
            results['no2_trop_gcshape'].append(no2_trop_gcshape)
            results['no2_tot_gcshape'].append(no2_tot_gcshape)
            results['frac_trop_gc'].append(sum_trop_gc / np.sum(no2prof))

            # Normalized profile shape correlation in troposphere
            prof_norm = prof_tm5_trop / sum_prof_trop if sum_prof_trop > 0 else prof_tm5_trop
            avk_trop_valid = AvKtrop_zeroed[p_tm5 >= troppt_val]
            prof_valid = prof_norm[p_tm5 >= troppt_val]
            if len(avk_trop_valid) > 2:
                c = np.corrcoef(avk_trop_valid, prof_valid)[0, 1]
                results['profile_corr_trop'].append(c if np.isfinite(c) else np.nan)
            else:
                results['profile_corr_trop'].append(np.nan)

            n_orbit += 1
            n_collected += 1

        print(f" {n_orbit} pixels collected (total: {n_collected})")

    # ── Convert to arrays ──
    for key in results:
        results[key] = np.array(results[key])
    n = len(results['lat'])
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC SUMMARY: {yyyymmdd}  ({n} pixels)")
    print(f"{'='*70}")

    # ── 1. AMF correction ratio statistics ──
    r_tot = results['ratio_tot']
    r_trop = results['ratio_trop']
    print(f"\n--- AMF correction ratio: sum(prof_gc * AvK) / sum(prof_gc) ---")
    print(f"  Total:  mean={np.nanmean(r_tot):.4f}  std={np.nanstd(r_tot):.4f}  "
          f"median={np.nanmedian(r_tot):.4f}  [p5={np.nanpercentile(r_tot,5):.4f}, p95={np.nanpercentile(r_tot,95):.4f}]")
    print(f"  Trop:   mean={np.nanmean(r_trop):.4f}  std={np.nanstd(r_trop):.4f}  "
          f"median={np.nanmedian(r_trop):.4f}  [p5={np.nanpercentile(r_trop,5):.4f}, p95={np.nanpercentile(r_trop,95):.4f}]")
    print(f"  → Trop has {'higher' if np.nanstd(r_trop) > np.nanstd(r_tot) else 'lower'} "
          f"variability ({np.nanstd(r_trop)/np.nanstd(r_tot):.2f}x total's std)")

    # ── 2. AMF change magnitude ──
    amf_change_tot = (results['amf_tot_gc'] - results['amf_tot_orig']) / results['amf_tot_orig'] * 100
    amf_change_trop = (results['amf_trop_gc'] - results['amf_trop_orig']) / results['amf_trop_orig'] * 100
    print(f"\n--- % change in AMF (gcshape vs original) ---")
    print(f"  Total:  mean={np.nanmean(amf_change_tot):.2f}%  std={np.nanstd(amf_change_tot):.2f}%")
    print(f"  Trop:   mean={np.nanmean(amf_change_trop):.2f}%  std={np.nanstd(amf_change_trop):.2f}%")

    # ── 3. Column change ──
    col_change_tot = (results['no2_tot_gcshape'] - results['no2_tot_vc']) / results['no2_tot_vc'] * 100
    col_change_trop = (results['no2_trop_gcshape'] - results['no2_trop_vc']) / results['no2_trop_vc'] * 100
    print(f"\n--- % change in VCD (gcshape vs original) ---")
    print(f"  Total:  mean={np.nanmean(col_change_tot):.2f}%  std={np.nanstd(col_change_tot):.2f}%")
    print(f"  Trop:   mean={np.nanmean(col_change_trop):.2f}%  std={np.nanstd(col_change_trop):.2f}%")

    # ── 4. Tropopause comparison ──
    trop_diff = results['troppt_gc_hPa'] - results['troppt_tm5_hPa']
    valid_trop = np.isfinite(trop_diff)
    print(f"\n--- Tropopause: GC - TM5 (hPa) ---")
    print(f"  mean={np.nanmean(trop_diff):.1f}  std={np.nanstd(trop_diff):.1f}  "
          f"median={np.nanmedian(trop_diff):.1f}  "
          f"[p5={np.nanpercentile(trop_diff[valid_trop],5):.1f}, p95={np.nanpercentile(trop_diff[valid_trop],95):.1f}]")
    print(f"  GC trop layers: mean={np.nanmean(results['n_trop_layers_gc']):.1f}")
    print(f"  TM5 trop layers: mean={np.nanmean(results['n_trop_layers_tm5']):.1f}")

    # ── 5. Fraction tropospheric ──
    print(f"\n--- GC tropospheric fraction of total column ---")
    print(f"  mean={np.nanmean(results['frac_trop_gc']):.3f}  "
          f"std={np.nanstd(results['frac_trop_gc']):.3f}")

    # ── 6. Profile-AvK correlation ──
    pc = results['profile_corr_trop']
    print(f"\n--- Correlation(GC_trop_profile_shape, AvKtrop) ---")
    print(f"  mean={np.nanmean(pc):.3f}  std={np.nanstd(pc):.3f}")

    # ── PLOTS ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'GC Shape Factor Diagnostics — {yyyymmdd}', fontsize=14)

    # (0,0) Histogram of ratio_tot vs ratio_trop
    ax = axes[0, 0]
    bins = np.linspace(0.5, 1.5, 60)
    ax.hist(r_tot, bins=bins, alpha=0.6, label=f'Total (std={np.nanstd(r_tot):.3f})', density=True)
    ax.hist(r_trop, bins=bins, alpha=0.6, label=f'Trop (std={np.nanstd(r_trop):.3f})', density=True)
    ax.axvline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel('Σ(prof_gc × AvK) / Σ(prof_gc)')
    ax.set_ylabel('Density')
    ax.set_title('AMF correction ratio')
    ax.legend(fontsize=9)

    # (0,1) Histogram of % AMF change
    ax = axes[0, 1]
    bins = np.linspace(-50, 50, 60)
    ax.hist(amf_change_tot, bins=bins, alpha=0.6, label='Total', density=True)
    ax.hist(amf_change_trop, bins=bins, alpha=0.6, label='Trop', density=True)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('% change in AMF')
    ax.set_title('AMF change distribution')
    ax.legend(fontsize=9)

    # (0,2) Tropopause scatter: GC vs TM5
    ax = axes[0, 2]
    valid = np.isfinite(results['troppt_tm5_hPa'])
    ax.scatter(results['troppt_tm5_hPa'][valid], results['troppt_gc_hPa'][valid],
               s=1, alpha=0.3)
    lims = (100, 500)
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlabel('TM5 tropopause (hPa)')
    ax.set_ylabel('GC tropopause (hPa)')
    ax.set_title(f'Tropopause: GC vs TM5\nmean diff={np.nanmean(trop_diff):.1f} hPa')
    ax.set_xlim(lims); ax.set_ylim(lims)

    # (1,0) Scatter: ratio_tot vs ratio_trop
    ax = axes[1, 0]
    ax.scatter(r_tot, r_trop, s=1, alpha=0.3)
    lims = (0.5, 1.5)
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlabel('Ratio (total)')
    ax.set_ylabel('Ratio (trop)')
    ax.set_title('AMF correction: total vs trop')
    ax.set_xlim(lims); ax.set_ylim(lims)

    # (1,1) VCD change scatter: tot vs trop
    ax = axes[1, 1]
    ax.scatter(col_change_tot, col_change_trop, s=1, alpha=0.3)
    lims = (-80, 80)
    ax.plot(lims, lims, 'k--', lw=1)
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('% VCD change (total)')
    ax.set_ylabel('% VCD change (trop)')
    ax.set_title('VCD correction: total vs trop')
    ax.set_xlim(lims); ax.set_ylim(lims)

    # (1,2) Trop ratio vs tropopause difference
    ax = axes[1, 2]
    ax.scatter(trop_diff[valid], r_trop[valid], s=1, alpha=0.3)
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.axvline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Tropopause diff: GC − TM5 (hPa)')
    ax.set_ylabel('Trop AMF correction ratio')
    ax.set_title('Does tropopause mismatch\ndrive trop correction noise?')

    plt.tight_layout()
    outpng = os.path.join(OUT_DIR, f'diagnose_gcshape_{yyyymmdd}.png')
    plt.savefig(outpng, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpng}")

    # ── Save CSV for further analysis ──
    import pandas as pd
    df = pd.DataFrame(results)
    outcsv = os.path.join(OUT_DIR, f'diagnose_gcshape_{yyyymmdd}.csv')
    df.to_csv(outcsv, index=False)
    print(f"Saved: {outcsv}")


if __name__ == '__main__':
    main()
