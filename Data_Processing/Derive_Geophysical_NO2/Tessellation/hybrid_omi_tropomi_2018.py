#!/usr/bin/env python3
"""
Hybrid OMI-TROPOMI Yearly Average for 2018
- Uses OMI data for months 1-5 (Jan-May)
- Uses TROPOMI data for months 6-12 (Jun-Dec)
"""
import os
import xarray as xr
import argparse
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def create_hybrid_yearly_average(year=2018):
    """
    Create hybrid yearly average for 2018:
    - OMI for months 1-5
    - TROPOMI for months 6-12
    """
    print(f"Creating hybrid OMI-TROPOMI yearly average for {year}")
    print(f"Process ID: {os.getpid()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    
    # Quality control strings for each dataset
    omi_qcstr = 'ECF030-SZA75-QA0-RA0'
    tropomi_qcstr = 'CF010-SZA75-QA75'
    
    # Base directories
    omi_monthly_dir = f'/my-projects2/1.project/NO2_col/OMI-MINDS/{year}/monthly'
    tropomi_monthly_dir = '/my-projects2/1.project/NO2_col/TROPOMI/monthly'
    output_dir = f'/my-projects2/1.project/NO2_col/TROPOMI/{year}'
    os.makedirs(output_dir, exist_ok=True)
    
    ds_list = []
    valid_months = 0
    months_info = []
    
    # Load OMI data for months 1-5
    print("\n=== Loading OMI data (months 1-5) ===")
    for month in range(1, 6):
        fname = f"OMI-MINDS_Regrid_{year}{month:02d}_Monthly_{omi_qcstr}.nc"
        fpath = os.path.join(omi_monthly_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing OMI monthly file: {fname}")
            continue
            
        try:
            # Try with dask chunks first, fall back to no chunking
            try:
                ds = xr.open_dataset(fpath, chunks="auto")
            except (ValueError, ImportError):
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'month' dimension
            ds = ds.expand_dims(month=[month])
            # Pick only the NO2 variables
            ds_list.append(ds[["NO2_tot", "NO2_tot_gcshape"]])
            valid_months += 1
            months_info.append(f"Month {month:02d} (OMI)")
            print(f"  ✓ Month {month:02d} (OMI)")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}")
            continue
    
    # Load TROPOMI data for months 6-12
    print("\n=== Loading TROPOMI data (months 6-12) ===")
    for month in range(6, 13):
        fname = f"Tropomi_Regrid_{year}{month:02d}_Monthly_{tropomi_qcstr}.nc"
        fpath = os.path.join(tropomi_monthly_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing TROPOMI monthly file: {fname}")
            continue
            
        try:
            # Try with dask chunks first, fall back to no chunking
            try:
                ds = xr.open_dataset(fpath, chunks="auto")
            except (ValueError, ImportError):
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'month' dimension
            ds = ds.expand_dims(month=[month])
            # Pick only the NO2 variables
            ds_list.append(ds[["NO2_tot", "NO2_tot_gcshape"]])
            valid_months += 1
            months_info.append(f"Month {month:02d} (TROPOMI)")
            print(f"  ✓ Month {month:02d} (TROPOMI)")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}")
            continue
    
    if not ds_list:
        print(f"[ERROR] No monthly data found for hybrid average")
        return False
    
    print(f"\n=== Averaging {valid_months}/12 months ===")
    print("Months used:")
    for info in months_info:
        print(f"  - {info}")
    
    try:
        # Concatenate along the 'month' axis and compute mean
        ds_all = xr.concat(ds_list, dim="month")
        yearly_mean = ds_all.mean(dim="month", skipna=True)
        
        # Add comprehensive metadata
        yearly_mean.attrs.update({
            'title': f'Hybrid OMI-TROPOMI yearly mean NO2 for {year}',
            'description': 'Combined OMI (Jan-May) and TROPOMI (Jun-Dec) monthly averages',
            'source': 'OMI-MINDS monthly (Jan-May) + TROPOMI monthly (Jun-Dec)',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'months_averaged': valid_months,
            'total_months': 12,
            'omi_months': '1-5',
            'tropomi_months': '6-12',
            'omi_quality_control': omi_qcstr,
            'tropomi_quality_control': tropomi_qcstr,
            'processed_by_pid': os.getpid(),
            'months_included': ', '.join(months_info)
        })
        
        # Save hybrid yearly mean
        out_fname = f"TROPOMI_OMI_Regrid_{year}_{tropomi_qcstr}.nc"
        out_path = os.path.join(output_dir, out_fname)
        
        print(f"\n=== Writing hybrid yearly mean ===")
        print(f"Output file: {out_path}")
        
        # Enhanced compression settings
        encoding = {}
        for var in yearly_mean.data_vars:
            encoding[var] = {
                "zlib": True, 
                "complevel": 4,
                "shuffle": True
            }
        
        yearly_mean.to_netcdf(out_path, encoding=encoding)
        
        print(f"✓ Successfully created hybrid yearly average for {year}")
        print(f"  Output: {out_path}")
        print(f"  File size: {os.path.getsize(out_path) / 1024**2:.2f} MB")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True, out_path
        
    except Exception as e:
        print(f"[ERROR] Failed to create hybrid yearly average for {year}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def plot_hybrid(ds, title, out_png):
    """Plot hybrid OMI-TROPOMI NO2 data"""
    # Load grid arrays
    try:
        x = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
        y = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 10),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    vars_no2 = [v for v in ds.data_vars if 'NO2' in v]
    
    for ax, var in zip(axes, vars_no2):
        v = ds[var].values
        if v.size == 0 or np.all(np.isnan(v)):
            print(f"[WARN] variable {var} has no valid data for plot '{title}'; skipping")
            continue
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
        
        try:
            maxval = np.nanmax(v)
        except ValueError:
            maxval = 0.0
        vmax = maxval * 0.8 if maxval > 0 else 1.0
        
        mesh = ax.pcolormesh(x, y, v,
                             transform=ccrs.PlateCarree(),
                             cmap='RdYlBu_r',
                             vmin=0, vmax=4e+16)
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.05, fraction=0.05)
        cbar.set_label(f'{var} [mol/cm²]')
        ax.set_title(f"{title}: {var}\n(OMI: Jan-May, TROPOMI: Jun-Dec)", pad=10)
        ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
        
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved plot: {out_png}")

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description='Create hybrid OMI-TROPOMI yearly average for 2018',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Create hybrid average for 2018
  %(prog)s --no-plot          # Skip plotting
  %(prog)s --year 2018        # Explicitly specify year
        """
    )
    parser.add_argument('--year', type=int, default=2018,
                       help='Year to process (default: 2018)')
    parser.add_argument('--no-plot', action='store_true',
                       help="Don't create plots")
    args = parser.parse_args()
    
    if args.year != 2018:
        print(f"WARNING: This script is designed for 2018 (OMI transition year)")
        print(f"You specified {args.year}. Proceeding anyway...")
    
    try:
        start_time = datetime.now()
        
        # Create hybrid yearly average
        success, out_path = create_hybrid_yearly_average(args.year)
        
        if success and not args.no_plot and out_path:
            # Create plot
            print("\n=== Creating visualization ===")
            ds = xr.open_dataset(out_path)
            plot_path = out_path.replace('.nc', '_plot.png')
            plot_hybrid(ds, f"{args.year} Hybrid (OMI+TROPOMI)", plot_path)
            ds.close()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            print(f"\n{'='*60}")
            print(f"✓ Successfully completed hybrid yearly average for {args.year}")
            print(f"  - OMI data: months 1-5 (Jan-May)")
            print(f"  - TROPOMI data: months 6-12 (Jun-Dec)")
            print(f"  - Total processing time: {duration}")
            print(f"{'='*60}")
            sys.exit(0)
        else:
            print(f"\n✗ Failed to process hybrid average for {args.year}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠ Processing interrupted")
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
