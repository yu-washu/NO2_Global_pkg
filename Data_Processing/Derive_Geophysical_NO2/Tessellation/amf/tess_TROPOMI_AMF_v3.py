#!/usr/bin/env python3
import sys
print("DEBUG: Script starting...", flush=True)
import os
import shutil
import tempfile
import subprocess
import argparse
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import psutil
from Tess_func import (
    read_TROPOMI,
    write_tessellation_input_grid_file,
    load_and_save_TROPOMI_AMFtot_to_nc,
)

def print_status(stage):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[{stage}] - Memory: {mem:.2f} GB")

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# spatial grid
max_lat, min_lat, max_lon, min_lon = 70, -60, 180, -180
latres, lonres = 0.01, 0.01
out_lat_edges = np.arange(min_lat, max_lat + latres, latres)
out_lon_edges = np.arange(min_lon, max_lon + lonres, lonres)
tlat = out_lat_edges[:-1] + latres / 2
tlon = out_lon_edges[:-1] + lonres / 2

# QC thresholds
sza_max, QAlim = 80, 0.75
qcstr = 'SZA{}-QA{}'.format(sza_max, int(QAlim * 100))

Na    = 6.023e23  # Avogadro's number
MwAir = 28.97     # g/mol

AMF_TROP_MIN = 0.5  # exclude pixels where amf_trop_gcshape < this value

# directories
gchp_dir          = '/my-projects2/1.project/gchp-v2/forTessellation/{year}/'
tess_code_dir     = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/gfortran_0p025_global/'
tropomi_l2_in_dir = '/my-projects/1.project/TROPOMI_L2_V2_NO2_2018-2023/'
tess_temp_dir     = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/temp/'

os.makedirs(tess_temp_dir, exist_ok=True)

# Two variables: native AMFtot and GCHP-profile-shaped AMF
tess_var_str = ['amf_trop', 'amf_trop_gcshape', 'amf_tot', 'amf_tot_gcshape']

# ─── PER-DAY WORKER ──────────────────────────────────────────────────────────────
def process_single_day(year, month, day):
    out_dir = f'/my-projects2/1.project/NO2col-v3/TROPOMI/{year}/daily_AMFtot/'
    os.makedirs(out_dir, exist_ok=True)
    label   = f"{year:04d}{month:02d}{day:02d}"
    scratch = tempfile.mkdtemp(prefix=f"tess_amf_{label}_", dir=tess_temp_dir)

    try:
        # 1) Copy & rename the executable into scratch
        exe_src = os.path.join(tess_code_dir, 'tessellate_ifx')
        exe_dst = os.path.join(scratch, f"tess_ifx_{label}")
        shutil.copy(exe_src, exe_dst)
        os.chmod(exe_dst, 0o755)

        # 2) Prepare file paths
        flabel   = f"{label}_{qcstr}"
        tess_in  = os.path.join(scratch, f"tessellate_input_{flabel}.dat")
        tess_out = os.path.join(scratch, f"tessellate_output_{flabel}.dat")
        setup_f  = os.path.join(scratch, f"tessellate_setup_{flabel}.dat")
        nc_out   = os.path.join(out_dir,  f"Tropomi_AMFtot_Regrid_{flabel}.nc")

        # 3) Load that day's GCHP file for profile-shaped AMF computation
        gc_file = (
            gchp_dir.format(year=year)
            + f'daily/01x01.Hours.13-15.{year}{month:02d}{day:02d}.nc4'
        )
        print(f"Loading GCHP file: {gc_file}", flush=True)
        ds = xr.open_dataset(gc_file, engine='netcdf4')
        gc_lat = ds['lat'].astype('float32').values
        gc_lon = ds['lon'].astype('float32').values
        P_GC      = ds['Met_PMIDDRY'].astype('float32').values
        no2_gc    = ds['SpeciesConcVV_NO2'].astype('float32').values
        a         = ds['Met_AIRDEN'].astype('float32').values * 1e3  # g/m3
        b         = ds['Met_BXHEIGHT'].astype('float32').values
        troppt_gc = ds['Met_TROPPT'].astype('float32').values
        ds.close()

        partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na  # molec cm-2

        # 4) Find matching TROPOMI orbit files
        yyyymmdd = label
        patterns = [
            f"S5P_RPRO_L2__NO2____{yyyymmdd}T",
            f"S5P_OFFL_L2__NO2____{yyyymmdd}T",
        ]

        if not os.path.exists(tropomi_l2_in_dir):
            print(f"ERROR: TROPOMI input directory not found: {tropomi_l2_in_dir}", flush=True)
            return

        print(f"Searching for TROPOMI files in {tropomi_l2_in_dir}", flush=True)
        files = os.listdir(tropomi_l2_in_dir)
        print(f"Found {len(files)} files in {tropomi_l2_in_dir}", flush=True)

        # Initialise aggregate statistics counters
        processed_count   = 0
        total_pixels_all  = 0
        passed_pixels_all = 0
        sza_pass_all      = 0
        qa_pass_all       = 0
        lat_pass_all      = 0
        lon_pass_all      = 0

        tess_files = []

        for i, fname in enumerate(files, start=1):
            if not any(pattern in fname for pattern in patterns):
                continue
            if fname.startswith("._") or fname.startswith("."):
                continue

            processed_count += 1

            try:
                trop = read_TROPOMI(os.path.join(tropomi_l2_in_dir, fname))
            except Exception as e:
                print(f"WARNING: Skipping corrupted file {fname}: {e}", flush=True)
                continue

            nscan, npix = trop['AMFtot'].shape[1:]
            scanIndex   = np.broadcast_to(
                np.arange(nscan)[None, :, None], (1, nscan, npix)
            )
            groundIndex = np.broadcast_to(
                np.arange(npix)[None, None, :], (1, nscan, npix)
            )

            good = (
                (trop['sza']          <  sza_max) &
                (trop['QualityFlag']  >  QAlim)   &
                (trop['Latitude']     >= min_lat)  &
                (trop['Latitude']     <= max_lat)  &
                (trop['Longitude']    >= min_lon)  &
                (trop['Longitude']    <= max_lon)
            )

            total_pixels_all  += good.size
            passed_pixels_all += int(np.sum(good))
            sza_pass_all      += int(np.sum(trop['sza']         <  sza_max))
            qa_pass_all       += int(np.sum(trop['QualityFlag'] >  QAlim))
            lat_pass_all      += int(np.sum((trop['Latitude']   >= min_lat) & (trop['Latitude']  <= max_lat)))
            lon_pass_all      += int(np.sum((trop['Longitude']  >= min_lon) & (trop['Longitude'] <= max_lon)))

            if not good.any():
                continue

            # Native AMFtot (replace NaN with 0 as tessellation sentinel)
            amf_vals = trop['AMFtot'][good].copy()
            tropomi_amf = np.where(np.isnan(amf_vals), 0.0, amf_vals)

            # GCHP-profile-shaped AMF (total and tropospheric)
            amf_tot_gcshape  = np.full(trop['AMFtot'].shape, np.nan)
            amf_trop_gcshape = np.full(trop['AMFtrop'].shape, np.nan)
            amf_trop_gc_pixel = np.full(trop['AMFtrop'].shape, np.nan)
            valid = np.where(good[0])
            lat_idx = np.round(
                np.interp(trop['Latitude'][good], gc_lat, np.arange(len(gc_lat)))
            ).astype(int)
            lon_idx = np.round(
                np.interp(trop['Longitude'][good], gc_lon, np.arange(len(gc_lon)))
            ).astype(int)

            for idx, (j, k) in enumerate(zip(*valid)):
                si, gi   = int(scanIndex[0, j, k]), int(groundIndex[0, j, k])
                p_gc     = P_GC[:, lat_idx[idx], lon_idx[idx]]
                prof_gc  = partial_column[:, lat_idx[idx], lon_idx[idx]]
                p_tm5    = np.ma.filled(np.array(trop['p'][0, si, gi, :]), np.nan)

                if np.isnan(p_gc).any() or np.isnan(prof_gc).any() or np.isnan(p_tm5).any():
                    continue

                prof_tm5  = interp1d(p_gc, prof_gc,
                                     bounds_error=False,
                                     fill_value='extrapolate')(p_tm5)
                AMFtot_px = trop['AMFtot'][0, si, gi]
                AvKtot    = trop['AvKtot'][0, si, gi, :]
                sum_prof  = np.sum(prof_tm5)

                denom = AMFtot_px * np.sum(prof_tm5 * AvKtot)
                if denom and not np.isnan(denom):
                    amf_tot_gcshape[0, j, k] = denom / sum_prof

                # tropospheric AMF: GC profile truncated at tropopause
                troppt_val   = troppt_gc[lat_idx[idx], lon_idx[idx]]
                prof_gc_trop = np.where(p_gc >= troppt_val, prof_gc, 0.0)
                sum_trop_gc = np.sum(prof_gc_trop)
                if sum_trop_gc > 0:
                    interp_trop  = interp1d(p_gc, prof_gc_trop,
                                            bounds_error=False, fill_value=0.0)
                    prof_tm5_trop = np.where(p_tm5 >= troppt_val, interp_trop(p_tm5), 0.0)
                    sum_prof_trop = np.sum(prof_tm5_trop)
                    if sum_prof_trop > 0:
                        AMFtrop_px = trop['AMFtrop'][0, si, gi]
                        AvKtrop_raw    = trop['AvKtrop'][0, si, gi, :]
                        AvKtrop = np.where(p_tm5 < troppt_val, 0.0, AvKtrop_raw)  # zero stratospheric layers
                        denom_trop = AMFtrop_px * np.sum(prof_tm5_trop * AvKtrop)
                        if denom_trop and not np.isnan(denom_trop):
                            amf_trop_gcshape[0, j, k] = denom_trop / sum_prof_trop
                            amf_trop_gc_pixel[0, j, k] = denom_trop / sum_prof_trop

            # AMF quality filter: exclude all vars where amf_trop_gcshape < AMF_TROP_MIN
            amf_bad = ~np.isfinite(amf_trop_gc_pixel) | (amf_trop_gc_pixel < AMF_TROP_MIN)
            amf_tot_gcshape[amf_bad] = np.nan
            amf_trop_gcshape[amf_bad] = np.nan
            trop['AMFtrop'] = np.where(amf_bad, np.nan, trop['AMFtrop'])
            trop['AMFtot'] = np.where(amf_bad, np.nan, trop['AMFtot'])
            n_qc_pass = int(good.sum())
            n_amf_bad = int(amf_bad[good].sum())
            n_remain = n_qc_pass - n_amf_bad
            pct = 100 * n_amf_bad / n_qc_pass if n_qc_pass > 0 else 0
            print(f"    AMF filter (amf_trop_gcshape < {AMF_TROP_MIN}): "
                  f"removed {n_amf_bad:,} / {n_qc_pass:,} pixels ({pct:.1f}%), "
                  f"{n_remain:,} remaining", flush=True)

            amf_trop_vals     = np.where(np.isnan(trop['AMFtrop'][good]), 0.0, trop['AMFtrop'][good])
            amf_trop_gc_vals  = amf_trop_gcshape[good].copy();  amf_trop_gc_vals[np.isnan(amf_trop_gc_vals)]  = 0.0
            amf_tot_vals      = np.where(np.isnan(trop['AMFtot'][good]), 0.0, trop['AMFtot'][good])
            amf_tot_gc_vals   = amf_tot_gcshape[good].copy();   amf_tot_gc_vals[np.isnan(amf_tot_gc_vals)]   = 0.0

            # Stack into Nx12 array: 8 corner coords + 4 data variables
            arr = np.vstack([
                trop['CornerLongitude'][good, 0], trop['CornerLatitude'][good, 0],
                trop['CornerLongitude'][good, 1], trop['CornerLatitude'][good, 1],
                trop['CornerLongitude'][good, 2], trop['CornerLatitude'][good, 2],
                trop['CornerLongitude'][good, 3], trop['CornerLatitude'][good, 3],
                amf_trop_vals,
                amf_trop_gc_vals,
                amf_tot_vals,
                amf_tot_gc_vals,
            ]).T

            orbit_dat = os.path.join(scratch, f"orbit_{i:02d}.dat")
            np.savetxt(
                orbit_dat,
                arr,
                fmt='%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %15.5E %15.5E %15.5E %15.5E'
            )
            tess_files.append(orbit_dat)

        if not tess_files:
            print("No matching TROPOMI files found for this day.", flush=True)
            return

        # Print comprehensive filtering statistics
        print(f"\n{'='*70}", flush=True)
        print(f"FILTERING STATISTICS SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Files in directory:             {len(files):>10,}", flush=True)
        print(f"Files processed (matched):      {processed_count:>10,}", flush=True)
        print(f"Orbits with data after filter:  {len(tess_files):>10,}", flush=True)
        print(f"\nPIXEL STATISTICS:", flush=True)
        print(f"Total pixels:                   {total_pixels_all:>10,} (100.0%)", flush=True)
        print(f"Pixels passing ALL filters:     {passed_pixels_all:>10,} ({100*passed_pixels_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Pixels filtered out:            {total_pixels_all-passed_pixels_all:>10,} ({100*(total_pixels_all-passed_pixels_all)/total_pixels_all:5.1f}%)", flush=True)
        print(f"\nINDIVIDUAL FILTER PERFORMANCE:", flush=True)
        print(f"SZA < {sza_max}°:                    {sza_pass_all:>10,} ({100*sza_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"QualityFlag > {QAlim}:               {qa_pass_all:>10,} ({100*qa_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Latitude in bounds:             {lat_pass_all:>10,} ({100*lat_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Longitude in bounds:            {lon_pass_all:>10,} ({100*lon_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"{'='*70}", flush=True)

        print_status(f"Finished reading {len(tess_files)} orbits")

        # 4) Build master tessellation input file
        with open(tess_in, 'w') as fid:
            fid.write(f"{len(tess_var_str)}  4\n")
            for fdat in tess_files:
                data = np.loadtxt(fdat)
                if data.ndim < 2:
                    continue
                n_corners = 8
                fmt = ' '.join(['%10.4f'] * n_corners + ['%15.5E'] * (data.shape[1] - n_corners))
                np.savetxt(fid, data, fmt=fmt)

        # 5) Write the grid setup file
        write_tessellation_input_grid_file(
            setup_f, tess_in, tess_out, out_lat_edges, out_lon_edges
        )

        print_status("Finished building input for Tessellation")

        # 6) Run the Fortran tessellation binary
        subprocess.run([exe_dst, setup_f], cwd=scratch, check=True)
        print_status("Finished Tessellation")

        # 7) Convert output to NetCDF
        load_and_save_TROPOMI_AMFtot_to_nc(tess_out, nc_out, len(tess_var_str), tlat, tlon)
        print_status("Daily AMFtot Tessellation saved")

        os.system(f"rm -rf {scratch}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ─── DISPATCH ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, required=True)
    parser.add_argument('--day',  type=int, required=True)
    args = parser.parse_args()

    process_single_day(args.year, args.mon, args.day)
