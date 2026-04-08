#!/usr/bin/env python3
import sys
print("DEBUG: Script starting...", flush=True)
import os
import shutil
import tempfile
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import psutil
from Tess_func import (
    read_OMI_MINDS,
    write_tessellation_input_grid_file,
    load_and_save_OMI_MINDS_to_nc
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
ECF_max, sza_max, QAFlag, RowAnomalyFlag = 0.1, 75, 0, 0
qcstr = 'ECF{:03d}-SZA{}-QA{}-RA{}'.format(
    int(ECF_max * 100),
    sza_max,
    int(QAFlag),
    int(RowAnomalyFlag)
)
Na = 6.023e23  # Avogadro’s number
MwAir = 28.97 # unit is g/mol

# directories
gchp_dir           = '/my-projects2/1.project/gchp/forTessellation/{year}/'
tess_code_dir      = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/gfortran_0p025_global/'
omi_l2_in_dir      = '/my-projects/1.project/OMI_L2_v1.1/'
tess_temp_dir      = '/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/temp/'

os.makedirs(tess_temp_dir, exist_ok=True)

tess_var_str = [
    'omi_NO2_total',
    'omi_NO2_gcshape_total'
]

# ─── PER-DAY WORKER ──────────────────────────────────────────────────────────────
def process_single_day(year, month, day):
    out_dir            = f'/my-projects2/1.project/NO2_col/OMI-MINDS/{year}/daily/'
    os.makedirs(out_dir, exist_ok=True)
    label = f"{year:04d}{month:02d}{day:02d}"
    scratch = tempfile.mkdtemp(prefix=f"tess_{label}_", dir=tess_temp_dir)

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
        nc_out   = os.path.join(out_dir,  f"OMI-MINDS_Regrid_{flabel}.nc")

        # 3) Load that day's GCHP file
        gc_file = (
            gchp_dir.format(year=year)
            + f'daily/01x01.Hours.13-15.{year}{month:02d}{day:02d}.nc4'
        )
        ds = xr.open_dataset(gc_file, engine='netcdf4')
        gc_lat = ds['lat'].astype('float32').values
        gc_lon = ds['lon'].astype('float32').values
        P_GC   = ds['Met_PMIDDRY'].astype('float32').values
        no2_gc = ds['SpeciesConcVV_NO2'].astype('float32').values
        a      = ds['Met_AIRDEN'].astype('float32').values* 1e3 # unit is g/m3
        b      = ds['Met_BXHEIGHT'].astype('float32').values
        ds.close()

        partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na # unit is molec cm-2
        

        # 4) Loop over all TROPOMI orbits that day
        tess_files = []
        yyyymmdd = label
        patterns = [f"OMI-Aura_L2-OMI_MINDS_NO2_{year:04d}m{month:02d}{day:02d}t"]
        
        if not os.path.exists(omi_l2_in_dir):
            print(f"ERROR: OMI input directory not found: {omi_l2_in_dir}", flush=True)
            return

        omi_indir_year = os.path.join(omi_l2_in_dir, f"{year}")
        print(f"Searching for OMI files in {omi_indir_year}", flush=True)
        files = os.listdir(omi_indir_year)
        print(f"Found {len(files)} files in {omi_indir_year}", flush=True)
        
        # Initialize aggregate statistics counters
        processed_count = 0
        total_pixels_all = 0
        passed_pixels_all = 0
        cf_pass_all = 0
        sza_pass_all = 0
        qa_pass_all = 0
        row_pass_all = 0
        lat_pass_all = 0
        lon_pass_all = 0
        
        for i, fname in enumerate(files, start=1):
            # Check if filename matches any of the patterns
            if not any(pattern in fname for pattern in patterns):
                continue
            
            # Skip hidden files
            if fname.startswith("._") or fname.startswith("."):
                continue

            processed_count += 1
            omi = read_OMI_MINDS(os.path.join(omi_indir_year, fname))
            nscan, npix = omi['no2_tot_sc'].shape
            scanIndex   = np.broadcast_to(
                np.arange(nscan)[None,:,None],
                (1, nscan, npix)
            )
            groundIndex = np.broadcast_to(
                np.arange(npix)[None,None,:],
                (1, nscan, npix)
            )

            good = (
                (omi['ECF'] <= ECF_max) &
                (omi['sza'] < sza_max) &
                (omi['QualityFlag'] == 0) & 
                (omi['RowAnomalyFlag'] == 0) &
                (omi['Latitude'] >= min_lat) &
                (omi['Latitude'] <= max_lat) &
                (omi['Longitude']>= min_lon) &
                (omi['Longitude']<= max_lon)
            )

            # Accumulate filtering statistics
            total_pixels = good.size
            passed_pixels = np.sum(good)
            
            total_pixels_all += total_pixels
            passed_pixels_all += passed_pixels
            cf_pass_all += np.sum(omi['ECF'] <= ECF_max)
            sza_pass_all += np.sum(omi['sza'] < sza_max)
            qa_pass_all += np.sum(omi['QualityFlag'] == 0)
            row_pass_all += np.sum(omi['RowAnomalyFlag'] == 0)
            lat_pass_all += np.sum((omi['Latitude'] >= min_lat) & (omi['Latitude'] <= max_lat))
            lon_pass_all += np.sum((omi['Longitude'] >= min_lon) & (omi['Longitude'] <= max_lon))

            # compute GC-shaped total column
            no2_gcshape = np.full(omi['no2_tot_vc'].shape, np.nan)
            valid = np.where(good)

            lat_idx = np.round(
                np.interp(omi['Latitude'][good], gc_lat, np.arange(len(gc_lat)))
            ).astype(int)
            lon_idx = np.round(
                np.interp(omi['Longitude'][good], gc_lon, np.arange(len(gc_lon)))
            ).astype(int)
            
            for idx, (j, k) in enumerate(zip(*valid)):
                si, gi = int(scanIndex[0, j, k]), int(groundIndex[0, j, k])
                p_gc       = P_GC[:, lat_idx[idx], lon_idx[idx]]
                # normalized a priori NO2 shape factor
                no2prof_gc = partial_column[:, lat_idx[idx], lon_idx[idx]] / np.sum(partial_column[:, lat_idx[idx], lon_idx[idx]])
                p_tm5      = omi['swp']
                if p_tm5.ndim > 1:
                    p_tm5 = p_tm5[si, gi, :]
                p_tm5 = np.ma.filled(p_tm5, np.nan)
                
                
                if np.isnan(p_gc).any() or np.isnan(no2prof_gc).any() or np.isnan(p_tm5).any():
                    continue

                interp_f = interp1d(p_gc, no2prof_gc, bounds_error=False, fill_value='extrapolate')
                prof_tm5 = interp_f(p_tm5)
                
                #sc_tot = omi['no2_tot_sc'][si, gi]
                sc_tot = np.ma.filled(omi['no2_tot_sc'][si, gi], np.nan)
                sw     = np.ma.filled(omi['sw'][si, gi, :], np.nan)

                denom  = np.sum(prof_tm5 * sw)

                if denom and not np.isnan(denom):
                    no2_gcshape[si, gi] = sc_tot / denom

            # stack into Nx11 array
            arr = np.vstack([
                omi['CornerLongitude'][good,0], omi['CornerLatitude'][good,0],
                omi['CornerLongitude'][good,1], omi['CornerLatitude'][good,1],
                omi['CornerLongitude'][good,2], omi['CornerLatitude'][good,2],
                omi['CornerLongitude'][good,3], omi['CornerLatitude'][good,3],
                omi['no2_tot_vc'][good],
                no2_gcshape[good]
            ]).T

            orbit_dat = os.path.join(scratch, f"orbit_{i:02d}.dat")
            np.savetxt(
                orbit_dat,
                arr,
                fmt='%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %15.5E %15.5E'
            )
            tess_files.append(orbit_dat)
        
        if not tess_files:
            print("No matching OMI files found for this day.", flush=True)
            return  # nothing to do this day
        
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
        print(f"Cloud Fraction <= {ECF_max}:        {cf_pass_all:>10,} ({100*cf_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"SZA < {sza_max}°:                    {sza_pass_all:>10,} ({100*sza_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"QualityFlag == 0:               {qa_pass_all:>10,} ({100*qa_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"RowAnomalyFlag == 0:            {row_pass_all:>10,} ({100*row_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Latitude in bounds:             {lat_pass_all:>10,} ({100*lat_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Longitude in bounds:            {lon_pass_all:>10,} ({100*lon_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"{'='*70}", flush=True)
        
        print_status(f"Finished AMF correction for {len(tess_files)} orbits")

        # 5) Build the master input for the Fortran code
        with open(tess_in, 'w') as fid:
            fid.write(f"{len(tess_var_str)}  4\n")
            for fdat in tess_files:
                data = np.loadtxt(fdat)
                if data.ndim < 2:
                    continue
                fmt = ' '.join(['%10.4f'] * (data.shape[1] - 1) + ['%15.5E'])
                np.savetxt(fid, data, fmt=fmt)
                # os.remove(fdat)

        # write the grid file
        write_tessellation_input_grid_file(
            setup_f, tess_in, tess_out, out_lat_edges, out_lon_edges
        )

        print_status("Finished building impute for Tessellation")
        
        # 6) run the Fortran binary
        subprocess.run([exe_dst, setup_f], cwd=scratch, check=True)
        print_status("Finished Tessellation")
        # os.remove(setup_f)

        # 7) convert to netCDF
        load_and_save_OMI_MINDS_to_nc(tess_out, nc_out, len(tess_var_str), tlat, tlon)
        # os.remove(tess_out)
        print_status("Daily Tessellation saved")
    
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