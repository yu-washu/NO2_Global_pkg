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

def slice_latitude(ds, lat_min=-60, lat_max=70):
    """
    Slice dataset to specified latitude range
    """
    # Find the latitude coordinate name (could be 'lat', 'latitude', etc.)
    lat_coord = None
    for coord in ds.coords:
        if coord.lower() in ['lat', 'latitude', 'y']:
            lat_coord = coord
            break
    
    if lat_coord is None:
        print("[WARN] No latitude coordinate found, returning original dataset")
        return ds
    
    print(f"  Slicing latitude from {lat_min} to {lat_max} degrees using coordinate '{lat_coord}'")
    
    # Slice the dataset
    lat_slice = ds.sel({lat_coord: slice(lat_min, lat_max)})
    
    # Add slicing info to attributes
    lat_slice.attrs.update({
        'latitude_slice': f'{lat_min} to {lat_max} degrees',
        'original_lat_range': f'{float(ds[lat_coord].min().values):.2f} to {float(ds[lat_coord].max().values):.2f}'
    })
    
    print(f"  Latitude range after slicing: {float(lat_slice[lat_coord].min().values):.2f} to {float(lat_slice[lat_coord].max().values):.2f}")
    
    return lat_slice

def average_daily_to_monthly(year, month, lat_min=-60, lat_max=70):
    """
    Aggregate daily TROPOMI NO₂ data into monthly means with latitude slicing.
    """
    base_dir = f'/my-projects2/1.project/gchp/forObservation-Geophysical/{year}/'
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
        fname = f"1x1km.Hours.13-15.{year}{month:02d}{day:02d}.nc4"
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
            
            # Apply latitude slicing
            ds = slice_latitude(ds, lat_min, lat_max)
            
            # Ensure there's a 'day' dimension
            ds = ds.expand_dims(day=[day])
            ds_list.append(ds)
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
            'title': f'GCHP monthly mean 3 Hours NO2 col for {year}-{month:02d}',
            'source': 'GCHP c180 daily output',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_averaged': valid_days,
            'total_days_in_month': days_in_month,
            'processed_by_pid': os.getpid(),
            'latitude_slice': f'{lat_min} to {lat_max} degrees'
        })
        
        # Save monthly mean with compression
        out_fname = f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.nc"
        out_path = os.path.join(monthly_dir, out_fname)
        
        print(f"  Writing monthly mean to {out_path}")
        
        # Enhanced compression settings
        encoding = {}
        for var in monthly_mean.data_vars:
            encoding[var] = {
                "zlib": True, 
                "complevel": 4,  # Higher compression
                "shuffle": True,  # Improve compression for floating point data
                "fletcher32": True  # Add checksum for data integrity
            }
        
        monthly_mean.to_netcdf(out_path, encoding=encoding)
        
        # Print file size info
        file_size = os.path.getsize(out_path) / (1024 * 1024)  # MB
        print(f"  ✓ Successfully created monthly average for {year}-{month:02d}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to create monthly average for {year}-{month:02d}: {str(e)}")
        return False

def average_monthly_to_yearly(year, lat_min=-60, lat_max=70):
    """
    Aggregate pre-computed monthly TROPOMI NO₂ means into a single yearly mean with latitude slicing.
    """
    base_dir = f'/my-projects2/1.project/gchp/forObservation-Geophysical/{year}/'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    os.makedirs(yearly_dir, exist_ok=True)
    
    ds_list = []
    valid_months = 0
    
    print(f"Creating yearly average for {year}")
    print(f"Process ID: {os.getpid()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for month in range(1, 13):
        # Look for the latitude-sliced monthly files first, then fall back to original
        fname = f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.nc"
        
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
            ds_list.append(ds)
            valid_months += 1
            print(f"  ✓ Month {month:02d}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {os.path.basename(fpath)}: {str(e)}")
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
            'title': f'Yearly mean GCHP for {year}',
            'source': 'GCHP monthly averages',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'months_averaged': valid_months,
            'total_months': 12,
            'processed_by_pid': os.getpid(),
            'latitude_slice': f'{lat_min} to {lat_max} degrees'
        })
        
        # Save yearly mean with enhanced compression
        out_fname = f"1x1km.Hours.13-15.{year}.AnnualMean.nc"
        out_path = os.path.join(yearly_dir, out_fname)
        
        print(f"  Writing yearly mean to {out_path}")
        
        # Enhanced compression settings
        encoding = {}
        for var in yearly_mean.data_vars:
            encoding[var] = {
                "zlib": True, 
                "complevel": 4,  # Higher compression
                "shuffle": True,  # Improve compression for floating point data
                "fletcher32": True  # Add checksum for data integrity
            }
        
        yearly_mean.to_netcdf(out_path, encoding=encoding)
        
        # Print file size info
        file_size = os.path.getsize(out_path) / (1024 * 1024)  # MB
        print(f"  ✓ Successfully created yearly average for {year}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to create yearly average for {year}: {str(e)}")
        return False

def plot_map(ds, title, out_png):
    """Plot GCHP NO2 data"""
    # Load grid arrays
    try:
        x = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
        y = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
        
        # Also slice the coordinate arrays if needed
        if 'latitude_slice' in ds.attrs:
            lat_coord = None
            for coord in ds.coords:
                if coord.lower() in ['lat', 'latitude', 'y']:
                    lat_coord = coord
                    break
            
            if lat_coord is not None:
                lat_values = ds[lat_coord].values
                lat_mask = (y >= lat_values.min()) & (y <= lat_values.max())
                y = y[lat_mask]
                x = x[lat_mask]
                
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 12),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Make it a list if only one subplot needed
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    vars_no2 = [v for v in ds.data_vars if 'NO2' in v.upper()]
    
    # If we have more NO2 variables than subplots, adjust
    if len(vars_no2) > len(axes):
        vars_no2 = vars_no2[:len(axes)]
    elif len(vars_no2) < len(axes):
        # Create additional subplots if needed or hide unused ones
        for i in range(len(vars_no2), len(axes)):
            axes[i].set_visible(False)
    
    for i, var in enumerate(vars_no2):
        ax = axes[i]
        v = ds[var].values
        
        if v.size == 0 or np.all(np.isnan(v)):
            print(f"[WARN] variable {var} has no valid data for plot '{title}'; skipping")
            ax.set_visible(False)
            continue
            
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        
        # Set extent based on actual data range
        lat_coord = None
        lon_coord = None
        for coord in ds.coords:
            if coord.lower() in ['lat', 'latitude', 'y']:
                lat_coord = coord
            elif coord.lower() in ['lon', 'longitude', 'x']:
                lon_coord = coord
        
        if lat_coord and lon_coord:
            lat_range = [float(ds[lat_coord].min()), float(ds[lat_coord].max())]
            lon_range = [float(ds[lon_coord].min()), float(ds[lon_coord].max())]
            ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], 
                         crs=ccrs.PlateCarree())
        else:
            ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
        
        try:
            maxval = np.nanmax(v)
        except ValueError:
            maxval = 0.0
        vmax = maxval * 0.8 if maxval > 0 else 1.0
        
        # Use the coordinate arrays for plotting
        if lat_coord and lon_coord:
            mesh = ax.pcolormesh(ds[lon_coord], ds[lat_coord], v,
                               transform=ccrs.PlateCarree(),
                               cmap='RdYlBu_r',
                               vmin=0, vmax=vmax)
        else:
            # Fallback to loaded coordinate arrays
            mesh = ax.pcolormesh(x, y, v,
                               transform=ccrs.PlateCarree(),
                               cmap='RdYlBu_r',
                               vmin=0, vmax=vmax)
        
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                           pad=0.05, fraction=0.05)
        cbar.set_label(var)
        ax.set_title(f"{title}: {var}", pad=10)
        ax.gridlines(draw_labels=True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot to {out_png}")

def process_full_workflow(year, lat_min=-60, lat_max=70, make_plots=True):
    """
    Complete workflow: process all months, then create yearly average
    """
    print(f"\n{'='*60}")
    print(f"FULL WORKFLOW FOR YEAR {year}")
    print(f"Latitude range: {lat_min} to {lat_max} degrees")
    print(f"{'='*60}")
    
    monthly_successes = 0
    
    # Process each month
    for month in range(1, 13):
        print(f"\n--- Processing Month {month:02d} ---")
        if average_daily_to_monthly(year, month, lat_min, lat_max):
            monthly_successes += 1
            
            # Create monthly plot if requested
            if make_plots:
                base_dir = '/my-projects2/1.project/gchp/forObservation-Geophysical/'
                monthly_dir = os.path.join(base_dir, "monthly")
                monthly_path = os.path.join(monthly_dir, f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.nc")
                
                if os.path.exists(monthly_path):
                    try:
                        ds_m = xr.open_dataset(monthly_path)
                        out_png = os.path.join(monthly_dir, f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.png")
                        plot_map(ds_m, f"{year}-{month:02d} Monthly", out_png)
                    except Exception as e:
                        print(f"  [WARN] Failed to create plot for month {month:02d}: {str(e)}")
    
    print(f"\n--- Monthly Processing Summary ---")
    print(f"Successfully processed: {monthly_successes}/12 months")
    
    if monthly_successes == 0:
        print("[ERROR] No monthly data was successfully processed")
        return False
    
    # Create yearly average
    print(f"\n--- Creating Yearly Average ---")
    yearly_success = average_monthly_to_yearly(year, lat_min, lat_max)
    
    # Create yearly plot if requested
    if yearly_success and make_plots:
        base_dir = '/my-projects2/1.project/gchp/forObservation-Geophysical/'
        yearly_dir = os.path.join(base_dir, "yearly")
        yearly_path = os.path.join(yearly_dir, f"1x1km.Hours.13-15.{year}.AnnualMean.nc")
        
        if os.path.exists(yearly_path):
            try:
                ds_y = xr.open_dataset(yearly_path)
                out_png = os.path.join(yearly_dir, f"1x1km.Hours.13-15.{year}.AnnualMean.png")
                print(f"  Attempting to create plot: {out_png}")
                print(f"  Dataset variables: {list(ds_y.data_vars)}")
                plot_map(ds_y, f"{year} Yearly", out_png)
            except Exception as e:
                print(f"  [WARN] Failed to create yearly plot: {str(e)}")
    
    return yearly_success

def main():
    """Main processing function with command line arguments"""
    parser = argparse.ArgumentParser(description='Process TROPOMI NO2 data averaging with latitude slicing')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=int, metavar='MONTH', choices=range(1, 13),
                       help='Process specific month only (1-12)')
    parser.add_argument('--yearly-only', action='store_true', 
                       help='Only create yearly average from existing monthly files')
    parser.add_argument('--full-workflow', action='store_true',
                       help='Process all months then create yearly average')
    parser.add_argument('--lat-min', type=float, default=-60,
                       help='Minimum latitude for slicing (default: -60)')
    parser.add_argument('--lat-max', type=float, default=70,
                       help='Maximum latitude for slicing (default: 70)')
    parser.add_argument('--no-plot', action='store_true',
                        help="Don't make PNG plots")
    args = parser.parse_args()
    
    year = args.year
    lat_min = args.lat_min
    lat_max = args.lat_max
    make_plots = not args.no_plot
    
    print(f"Processing year {year}")
    print(f"Latitude slice: {lat_min} to {lat_max}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    
    try:
        start_time = datetime.now()
        success = False
        
        if args.full_workflow:
            # Complete workflow: all months then yearly
            success = process_full_workflow(year, lat_min, lat_max, make_plots)
            
        elif args.yearly_only:
            # Only create yearly average
            success = average_monthly_to_yearly(year, lat_min, lat_max)
            
            # Create plot if requested
            if success and make_plots:
                base_dir = '/my-projects2/1.project/gchp/forObservation-Geophysical/'
                yearly_dir = os.path.join(base_dir, "yearly")
                yearly_path = os.path.join(yearly_dir, f"1x1km.Hours.13-15.{year}.AnnualMean.nc")
                if os.path.exists(yearly_path):
                    ds_y = xr.open_dataset(yearly_path)
                    out_png = os.path.join(yearly_dir, f"1x1km.Hours.13-15.{year}.AnnualMean.png")
                    print(f"  Attempting to create plot: {out_png}")
                    print(f"  Dataset variables: {list(ds_y.data_vars)}")
                    plot_map(ds_y, f"{year} Yearly", out_png)
                    
        elif args.month:
            # Process specific month only
            success = average_daily_to_monthly(year, args.month, lat_min, lat_max)
            
            # Create plot if requested
            if success and make_plots:
                base_dir = '/my-projects2/1.project/gchp/forObservation-Geophysical/'
                monthly_dir = os.path.join(base_dir, "monthly")
                monthly_path = os.path.join(monthly_dir, f"1x1km.Hours.13-15.{year}{args.month:02d}.MonMean.nc")
                if os.path.exists(monthly_path):
                    ds_m = xr.open_dataset(monthly_path)
                    out_png = os.path.join(monthly_dir, f"1x1km.Hours.13-15.{year}{args.month:02d}.MonMean.png")
                    print(f"  Attempting to create plot: {out_png}")
                    print(f"  Dataset variables: {list(ds_m.data_vars)}")
                    plot_map(ds_m, f"{year}-{args.month:02d}", out_png)
        else:
            print("Error: Please specify one of:")
            print("  --month N          (process specific month)")
            print("  --yearly-only      (create yearly from existing monthly files)")
            print("  --full-workflow    (process all months then yearly)")
            sys.exit(1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            print(f"\n{'='*60}")
            print(f"✓ PROCESSING COMPLETED SUCCESSFULLY")
            print(f"Total processing time: {duration}")
            print(f"{'='*60}")
            sys.exit(0)
        else:
            print(f"\n✗ Processing failed for year {year}")
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