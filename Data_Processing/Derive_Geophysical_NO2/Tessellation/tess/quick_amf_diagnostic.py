#!/usr/bin/env python3
"""
Quick L2-pixel-level diagnostic: compare AMF_trop (TM5 profile) vs AMF_trop (GC profile)
without tessellation.  Runs in ~5-10 min for one day.

Usage:
    python quick_amf_diagnostic.py --year 2023 --mon 7 --day 1
"""
import os
import sys
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

# ---------- Add Tess_func path ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tess'))
from Tess_func import read_TROPOMI

# ---------- Constants ----------
Na    = 6.023e23
MwAir = 28.97
sza_max, QAlim = 80, 0.75
min_lat, max_lat = -60, 70
min_lon, max_lon = -180, 180

gchp_dir          = '/my-projects2/1.project/gchp-v2/forTessellation/{year}/'
tropomi_l2_in_dir = '/my-projects/1.project/TROPOMI_L2_V2_NO2_2018-2023/'

def process_day(year, month, day):
    label = f"{year:04d}{month:02d}{day:02d}"

    # ---- Load GCHP ----
    gc_file = gchp_dir.format(year=year) + f'daily/01x01.Hours.13-15.{label}.nc4'
    print(f"Loading GCHP: {gc_file}", flush=True)
    ds = xr.open_dataset(gc_file, engine='netcdf4')
    gc_lat    = ds['lat'].astype('float32').values
    gc_lon    = ds['lon'].astype('float32').values
    P_GC      = ds['Met_PMIDDRY'].astype('float32').values
    no2_gc    = ds['SpeciesConcVV_NO2'].astype('float32').values
    a         = ds['Met_AIRDEN'].astype('float32').values * 1e3
    b         = ds['Met_BXHEIGHT'].astype('float32').values
    troppt_gc = ds['Met_TROPPT'].astype('float32').values
    ds.close()
    partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na

    # ---- Collect pixel-level results ----
    lats_all, lons_all = [], []
    amf_trop_tm5_all, amf_trop_gc_all = [], []
    amf_tot_tm5_all, amf_tot_gc_all = [], []
    no2_trop_vc_all = []

    patterns = [f"S5P_RPRO_L2__NO2____{label}T", f"S5P_OFFL_L2__NO2____{label}T"]
    files = [f for f in os.listdir(tropomi_l2_in_dir)
             if any(p in f for p in patterns) and not f.startswith('.')]
    print(f"Found {len(files)} orbits", flush=True)

    for fi, fname in enumerate(files):
        print(f"  Orbit {fi+1}/{len(files)}: {fname[:50]}...", flush=True)
        try:
            trop = read_TROPOMI(os.path.join(tropomi_l2_in_dir, fname))
        except Exception as e:
            print(f"    SKIP: {e}", flush=True)
            continue

        nscan, npix = trop['AMFtot'].shape[1:]
        scanIndex   = np.broadcast_to(np.arange(nscan)[None,:,None], (1,nscan,npix))
        groundIndex = np.broadcast_to(np.arange(npix)[None,None,:],  (1,nscan,npix))

        good = (
            (trop['sza']         <  sza_max) &
            (trop['QualityFlag'] >  QAlim)   &
            (trop['Latitude']    >= min_lat)  &
            (trop['Latitude']    <= max_lat)  &
            (trop['Longitude']   >= min_lon)  &
            (trop['Longitude']   <= max_lon)
        )
        if not good.any():
            continue

        valid = np.where(good[0])
        lat_idx = np.round(
            np.interp(trop['Latitude'][good], gc_lat, np.arange(len(gc_lat)))
        ).astype(int)
        lon_idx = np.round(
            np.interp(trop['Longitude'][good], gc_lon, np.arange(len(gc_lon)))
        ).astype(int)

        for idx, (j, k) in enumerate(zip(*valid)):
            si, gi = int(scanIndex[0,j,k]), int(groundIndex[0,j,k])
            p_gc    = P_GC[:, lat_idx[idx], lon_idx[idx]]
            prof_gc = partial_column[:, lat_idx[idx], lon_idx[idx]]
            p_tm5   = np.ma.filled(np.array(trop['p'][0, si, gi, :]), np.nan)

            if np.isnan(p_gc).any() or np.isnan(prof_gc).any() or np.isnan(p_tm5).any():
                continue

            AMFtot_px  = trop['AMFtot'][0, si, gi]
            AMFtrop_px = trop['AMFtrop'][0, si, gi]
            AvKtot     = trop['AvKtot'][0, si, gi, :]

            # ---- Total AMF with GC shape ----
            prof_tm5 = interp1d(p_gc, prof_gc, bounds_error=False,
                                fill_value='extrapolate')(p_tm5)
            sum_prof = np.sum(prof_tm5)
            if sum_prof <= 0:
                continue
            amf_tot_gc = AMFtot_px * np.sum(prof_tm5 * AvKtot) / sum_prof

            # ---- Tropospheric AMF with GC shape ----
            # Reuse prof_tm5 (full profile already interpolated to TM5 levels
            # with extrapolation, line 108); mask to troposphere afterward.
            # Use AMFtot * AvKtot = scattering weight m_l, which is independent
            # of the tropopause definition used by TROPOMI.
            troppt_val    = troppt_gc[lat_idx[idx], lon_idx[idx]]
            prof_tm5_trop = np.where(p_tm5 >= troppt_val,
                                     np.maximum(prof_tm5, 0.0), 0.0)
            sum_prof_trop = np.sum(prof_tm5_trop)
            if sum_prof_trop <= 0:
                continue
            amf_trop_gc = AMFtot_px * np.sum(prof_tm5_trop * AvKtot) / sum_prof_trop

            if (np.isnan(amf_tot_gc) or np.isnan(amf_trop_gc)
                or amf_tot_gc <= 0 or amf_trop_gc <= 0):
                continue

            lats_all.append(float(trop['Latitude'][0, j, k]))
            lons_all.append(float(trop['Longitude'][0, j, k]))
            amf_trop_tm5_all.append(float(AMFtrop_px))
            amf_trop_gc_all.append(float(amf_trop_gc))
            amf_tot_tm5_all.append(float(AMFtot_px))
            amf_tot_gc_all.append(float(amf_tot_gc))
            no2_trop_vc_all.append(float(trop['no2_trop_vc'][0, j, k]))

    # Convert to arrays
    lats   = np.array(lats_all)
    lons   = np.array(lons_all)
    amf_trop_tm5 = np.array(amf_trop_tm5_all)
    amf_trop_gc  = np.array(amf_trop_gc_all)
    amf_tot_tm5  = np.array(amf_tot_tm5_all)
    amf_tot_gc   = np.array(amf_tot_gc_all)
    no2_trop     = np.array(no2_trop_vc_all)

    print(f"\nTotal valid pixels: {len(lats)}", flush=True)

    # ---- Save raw data for later analysis ----
    out_dir = os.path.dirname(__file__)
    npz_out = os.path.join(out_dir, f"amf_diagnostic_{label}.npz")
    np.savez_compressed(npz_out,
                        lat=lats, lon=lons,
                        amf_trop_tm5=amf_trop_tm5, amf_trop_gc=amf_trop_gc,
                        amf_tot_tm5=amf_tot_tm5,   amf_tot_gc=amf_tot_gc,
                        no2_trop=no2_trop)
    print(f"Saved: {npz_out}", flush=True)

    # ---- Plots ----
    make_plots(lats, lons, amf_trop_tm5, amf_trop_gc,
               amf_tot_tm5, amf_tot_gc, no2_trop, label, out_dir)


def make_plots(lats, lons, amf_trop_tm5, amf_trop_gc,
               amf_tot_tm5, amf_tot_gc, no2_trop, label, out_dir):

    # Define NO2 bins: "clean" vs "polluted"
    no2_p50 = np.percentile(no2_trop[no2_trop > 0], 75)
    polluted = no2_trop > no2_p50
    clean    = ~polluted

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'AMF Diagnostic — {label}  (N={len(lats):,})', fontsize=14)

    # ---- Row 1: Tropospheric AMF ----
    # 1a) Scatter: AMF_trop TM5 vs GC, colored by NO2
    ax = axes[0, 0]
    sc = ax.scatter(amf_trop_tm5, amf_trop_gc, c=no2_trop * 1e-15,
                    s=0.5, alpha=0.3, cmap='YlOrRd', rasterized=True,
                    norm=mcolors.LogNorm(vmin=0.5, vmax=30))
    lim = [0, max(np.percentile(amf_trop_tm5, 99), np.percentile(amf_trop_gc, 99))]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlabel('AMF_trop (TM5)')
    ax.set_ylabel('AMF_trop (GC shape)')
    ax.set_title('Trop AMF: TM5 vs GC')
    ax.set_xlim(lim); ax.set_ylim(lim)
    plt.colorbar(sc, ax=ax, label='NO₂ trop VCD [1e15 molec/cm²]')

    # 1b) Histogram of ratio AMF_trop_GC / AMF_trop_TM5
    ax = axes[0, 1]
    ratio_trop = amf_trop_gc / amf_trop_tm5
    ratio_trop_clip = np.clip(ratio_trop, 0.2, 3.0)
    ax.hist(ratio_trop_clip[clean],    bins=80, alpha=0.6, density=True, label=f'Clean (N={clean.sum():,})')
    ax.hist(ratio_trop_clip[polluted], bins=80, alpha=0.6, density=True, label=f'Polluted top-25% (N={polluted.sum():,})')
    ax.axvline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('AMF_trop_GC / AMF_trop_TM5')
    ax.set_ylabel('Density')
    ax.set_title('Trop AMF ratio distribution')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95,
            f'Polluted median: {np.median(ratio_trop[polluted]):.3f}\n'
            f'Clean median: {np.median(ratio_trop[clean]):.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    # 1c) Map of ratio
    ax = axes[0, 2]
    sc2 = ax.scatter(lons, lats, c=ratio_trop, s=0.2, alpha=0.3,
                     cmap='RdBu_r', vmin=0.5, vmax=1.5, rasterized=True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('AMF_trop ratio (GC/TM5) — spatial')
    plt.colorbar(sc2, ax=ax, label='Ratio')

    # ---- Row 2: Total AMF (same layout) ----
    ax = axes[1, 0]
    sc = ax.scatter(amf_tot_tm5, amf_tot_gc, c=no2_trop * 1e-15,
                    s=0.5, alpha=0.3, cmap='YlOrRd', rasterized=True,
                    norm=mcolors.LogNorm(vmin=0.5, vmax=30))
    lim = [0, max(np.percentile(amf_tot_tm5, 99), np.percentile(amf_tot_gc, 99))]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlabel('AMF_tot (TM5)')
    ax.set_ylabel('AMF_tot (GC shape)')
    ax.set_title('Total AMF: TM5 vs GC')
    ax.set_xlim(lim); ax.set_ylim(lim)
    plt.colorbar(sc, ax=ax, label='NO₂ trop VCD [1e15 molec/cm²]')

    ax = axes[1, 1]
    ratio_tot = amf_tot_gc / amf_tot_tm5
    ratio_tot_clip = np.clip(ratio_tot, 0.2, 3.0)
    ax.hist(ratio_tot_clip[clean],    bins=80, alpha=0.6, density=True, label=f'Clean (N={clean.sum():,})')
    ax.hist(ratio_tot_clip[polluted], bins=80, alpha=0.6, density=True, label=f'Polluted top-25% (N={polluted.sum():,})')
    ax.axvline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('AMF_tot_GC / AMF_tot_TM5')
    ax.set_ylabel('Density')
    ax.set_title('Total AMF ratio distribution')
    ax.legend(fontsize=8)
    ax.text(0.02, 0.95,
            f'Polluted median: {np.median(ratio_tot[polluted]):.3f}\n'
            f'Clean median: {np.median(ratio_tot[clean]):.3f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    ax = axes[1, 2]
    sc2 = ax.scatter(lons, lats, c=ratio_tot, s=0.2, alpha=0.3,
                     cmap='RdBu_r', vmin=0.5, vmax=1.5, rasterized=True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('AMF_tot ratio (GC/TM5) — spatial')
    plt.colorbar(sc2, ax=ax, label='Ratio')

    plt.tight_layout()
    fig_out = os.path.join(out_dir, f"amf_diagnostic_{label}.png")
    plt.savefig(fig_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {fig_out}", flush=True)

    # ---- Additional: scatter of NO2_trop_vc ratio ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f'Impact on tropospheric NO₂ VCD — {label}', fontsize=13)

    # VCD ratio = AMF_trop_TM5 / AMF_trop_GC  (since VCD = SCD / AMF)
    vcd_ratio = amf_trop_tm5 / amf_trop_gc  # >1 means GC gives higher VCD

    ax = axes2[0]
    sc = ax.scatter(no2_trop * 1e-15, vcd_ratio, s=0.5, alpha=0.3,
                    c=np.abs(lats), cmap='viridis', rasterized=True)
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('NO₂ trop VCD (TM5) [1e15 molec/cm²]')
    ax.set_ylabel('VCD_GC / VCD_TM5  (= AMF_TM5 / AMF_GC)')
    ax.set_title('VCD change vs pollution level')
    ax.set_xlim(0, 30)
    ax.set_ylim(0.5, 2.0)
    plt.colorbar(sc, ax=ax, label='|Latitude|')

    # Binned median
    ax = axes2[1]
    bins = np.linspace(0, 20, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    no2_e15 = no2_trop * 1e-15
    medians, q25, q75 = [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (no2_e15 >= b0) & (no2_e15 < b1)
        if mask.sum() > 10:
            medians.append(np.median(vcd_ratio[mask]))
            q25.append(np.percentile(vcd_ratio[mask], 25))
            q75.append(np.percentile(vcd_ratio[mask], 75))
        else:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
    medians, q25, q75 = np.array(medians), np.array(q25), np.array(q75)
    ax.fill_between(bin_centers, q25, q75, alpha=0.3, label='IQR')
    ax.plot(bin_centers, medians, 'b-o', ms=4, label='Median')
    ax.axhline(1.0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('NO₂ trop VCD (TM5) [1e15 molec/cm²]')
    ax.set_ylabel('VCD_GC / VCD_TM5')
    ax.set_title('Binned median: VCD change vs pollution')
    ax.set_ylim(0.5, 2.0)
    ax.legend()

    plt.tight_layout()
    fig2_out = os.path.join(out_dir, f"amf_vcd_impact_{label}.png")
    plt.savefig(fig2_out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {fig2_out}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--mon',  type=int, default=7)
    parser.add_argument('--day',  type=int, default=1)
    args = parser.parse_args()
    process_day(args.year, args.mon, args.day)
