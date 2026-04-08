import os
import xarray as xr
import argparse
import sys
from datetime import datetime
import calendar
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def average_daily_to_monthly(year, month):
    """
    Aggregate daily TROPOMI NO₂ data into monthly means.
    """
    CloudFraction_max, sza_max, QAlim = 0.2, 75, 0.75
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = f'/my-projects2/1.project/NO2_col/TROPOMI-v2/{year}'
    daily_dir = os.path.join(base_dir, "daily")
    monthly_dir = os.path.join(base_dir, "monthly")
    os.makedirs(monthly_dir, exist_ok=True)
    
    # Get number of days in the month
    days_in_month = calendar.monthrange(year, month)[1]
    
    ds_list = []
    valid_days = 0
    
    print(f"Processing {year}-{month:02d} ({days_in_month} days)")
    print(f"Process ID: {os.getpid()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for day in range(1, days_in_month + 1):
        fname = f"Tropomi_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc"
        fpath = os.path.join(daily_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing daily file: {fname}")
            continue
            
        try:
            # Try with dask chunks first, fall back to no chunking
            try:
                ds = xr.open_dataset(fpath, chunks={"lat": 100, "lon": 100})
            except (ValueError, ImportError):
                # If dask not available, open without chunking
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'day' dimension
            ds = ds.expand_dims(day=[day])
            # Pick only the NO2 variables
            ds_list.append(ds[["NO2_tot", "NO2_tot_gcshape"]])
            valid_days += 1
            print(f"  ✓ Day {day:02d}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}")
            continue
   
    if not ds_list:
        print(f"[ERROR] No daily data found for {year}-{month:02d}")
        return False
    
    print(f"  Averaging {valid_days}/{days_in_month} days...")
    
    try:
        # Concatenate along the 'day' axis and compute mean
        ds_all = xr.concat(ds_list, dim="day")
        monthly_mean = ds_all.mean(dim="day", skipna=True)
        
        # Add metadata
        monthly_mean.attrs.update({
            'title': f'Monthly mean TROPOMI NO2 for {year}-{month:02d}',
            'source': 'TROPOMI daily data',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_averaged': valid_days,
            'total_days_in_month': days_in_month,
            'quality_control': qcstr,
            'processed_by_pid': os.getpid()
        })
        
        # Save monthly mean
        out_fname = f"Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}_noHP.nc"
        out_path = os.path.join(monthly_dir, out_fname)
        
        print(f"  Writing monthly mean to {out_path}")
        monthly_mean.to_netcdf(
            out_path,
            encoding={var: {"zlib": True, "complevel": 4} 
                      for var in monthly_mean.data_vars}
        )
        
        print(f"  ✓ Successfully created monthly average for {year}-{month:02d}")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to create monthly average for {year}-{month:02d}: {str(e)}")
        return False

def average_monthly_to_yearly(year):
    """
    Aggregate pre-computed monthly TROPOMI NO₂ means into a single yearly mean.
    """
    CloudFraction_max, sza_max, QAlim = 0.1, 75, 0.75
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = f'/my-projects2/1.project/NO2_col/TROPOMI-v2/{year}'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    os.makedirs(yearly_dir, exist_ok=True)
    
    ds_list = []
    valid_months = 0
    
    print(f"Creating yearly average for {year}")
    print(f"Process ID: {os.getpid()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for month in range(1, 13):
        fname = f"Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        fpath = os.path.join(monthly_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing monthly file: {fname}")
            continue
            
        try:
            # Try with dask chunks first, fall back to no chunking
            try:
                ds = xr.open_dataset(fpath, chunks={"lat": 100, "lon": 100})
            except (ValueError, ImportError):
                # If dask not available, open without chunking
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'month' dimension
            ds = ds.expand_dims(month=[month])
            # Pick only the NO2 variables
            ds_list.append(ds[["NO2_tot", "NO2_tot_gcshape"]])
            valid_months += 1
            print(f"  ✓ Month {month:02d}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}")
            continue
    
    if not ds_list:
        print(f"[ERROR] No monthly data found for year {year}")
        return False
    
    print(f"  Averaging {valid_months}/12 months...")
    
    try:
        # Concatenate along the 'month' axis and compute mean
        ds_all = xr.concat(ds_list, dim="month")
        yearly_mean = ds_all.mean(dim="month", skipna=True)
        
        # Add metadata
        yearly_mean.attrs.update({
            'title': f'Yearly mean TROPOMI NO2 for {year}',
            'source': 'TROPOMI monthly averages',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'months_averaged': valid_months,
            'total_months': 12,
            'quality_control': qcstr,
            'processed_by_pid': os.getpid()
        })
        
        # Save yearly mean
        out_fname = f"Tropomi_Regrid_{year}_{qcstr}_noHP.nc"
        out_path = os.path.join(yearly_dir, out_fname)
        
        print(f"  Writing yearly mean to {out_path}")
        yearly_mean.to_netcdf(
            out_path,
            encoding={var: {"zlib": True, "complevel": 4} 
                      for var in yearly_mean.data_vars}
        )
        
        print(f"  ✓ Successfully created yearly average for {year}")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to create yearly average for {year}: {str(e)}")
        return False

def plot_tropomi(ds, title, out_png):
    """Plot TROPOMI NO2 data"""
    # Load grid arrays
    try:
        x = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
        y = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    vars_no2 = [v for v in ds.data_vars if 'NO2' in v]
    
    for ax, var in zip(axes, vars_no2):
        v = ds[var].values
        if v.size == 0 or np.all(np.isnan(v)):
            print(f"[WARN] variable {var} has no valid data for plot '{title}'; skipping")
            continue
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
        try:
            maxval = np.nanmax(v)
        except ValueError:
            maxval = 0.0
        vmax = maxval * 0.8 if maxval > 0 else 1.0
        mesh = ax.pcolormesh(x, y, v,
                             transform=ccrs.PlateCarree(),
                             cmap='RdYlBu_r',
                             vmin=0, vmax=1e-16)
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.05, fraction=0.05)
        cbar.set_label(var)
        ax.set_title(f"{title}: {var}", pad=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    """Main processing function with command line arguments"""
    parser = argparse.ArgumentParser(description='Process TROPOMI NO2 data averaging')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=int, metavar='MONTH', choices=range(1, 13),
                       help='Process specific month only (1-12)')
    parser.add_argument('--yearly-only', action='store_true', 
                       help='Only create yearly average from existing monthly files')
    parser.add_argument('--no-plot', action='store_true',
                        help="Don't make PNG plots")
    args = parser.parse_args()
    year = args.year
    
    print(f"Processing year {year}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    
    # Set up quality control string
    CloudFraction_max, sza_max, QAlim = 0.2, 75, 0.75
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = f'/my-projects2/1.project/NO2_col/TROPOMI-v2/{year}'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    
    try:
        start_time = datetime.now()
        success = False
        
        if args.yearly_only:
            # Only create yearly average
            success = average_monthly_to_yearly(year)
            
            # Create plot if requested
            if success and not args.no_plot:
                yearly_path = os.path.join(yearly_dir, f"Tropomi_Regrid_{year}_{qcstr}_noHP.nc")
                if os.path.exists(yearly_path): 
                    ds_y = xr.open_dataset(yearly_path)
                    out_png = os.path.join(yearly_dir, f"Tropomi_Regrid_{year}_plot_noHP.png")
                    plot_tropomi(ds_y, f"{year} Yearly", out_png)
                    print(f"Saved plot {out_png}")
                    
        elif args.month:
            # Process specific month only
            success = average_daily_to_monthly(year, args.month)
            
            # Create plot if requested
            if success and not args.no_plot:
                monthly_path = os.path.join(monthly_dir, f"Tropomi_Regrid_{year}{args.month:02d}_Monthly_{qcstr}_noHP.nc")
                if os.path.exists(monthly_path):
                    ds_m = xr.open_dataset(monthly_path)
                    out_png = os.path.join(monthly_dir, f"Tropomi_Regrid_{year}{args.month:02d}_plot_noHP.png")
                    plot_tropomi(ds_m, f"{year}-{args.month:02d}", out_png)
                    print(f"[{args.month:02d}] Saved plot {out_png}")
        else:
            print("Error: Please specify either --month N or --yearly-only")
            print("For parallel processing, use separate jobs for each month")
            sys.exit(1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            if args.month:
                print(f"\n✓ Successfully completed monthly processing for {year}-{args.month:02d}")
            else:
                print(f"\n✓ Successfully completed yearly processing for {year}")
            print(f"Total processing time: {duration}")
            sys.exit(0)
        else:
            print(f"\n✗ Failed to process {year}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠ Processing interrupted for year {year}")
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error processing year {year}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()