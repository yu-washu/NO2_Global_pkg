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
import psutil
import gc
import time
import warnings

def check_memory():
    """Monitor memory usage"""
    mem = psutil.virtual_memory()
    return {
        'used_gb': mem.used / 1e9,
        'available_gb': mem.available / 1e9,
        'percent': mem.percent
    }

def log_memory(step, flush_output=True):
    """Log memory usage with timestamp"""
    mem = check_memory()
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"  [{timestamp}] {step}: {mem['used_gb']:.1f}GB used, {mem['available_gb']:.1f}GB free ({mem['percent']:.1f}%)")
    if flush_output:
        sys.stdout.flush()

def force_cleanup():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    time.sleep(0.1)  # Brief pause for system cleanup

def average_daily_to_monthly_optimized(year, month):
    """
    Optimized daily to monthly aggregation with memory monitoring
    """
    CloudFraction_max, sza_max, QAlim = 0.1, 75, 0
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = f'/my-projects2/1.project/NO2_col/OMI/{year}/'
    daily_dir = os.path.join(base_dir, "daily")
    monthly_dir = os.path.join(base_dir, "monthly")
    os.makedirs(monthly_dir, exist_ok=True)
    
    # Get number of days in the month
    days_in_month = calendar.monthrange(year, month)[1]
    
    print(f"Processing {year}-{month:02d} ({days_in_month} days)", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    log_memory("Initial")
    
    # Adaptive chunking based on available memory
    initial_mem = check_memory()
    if initial_mem['available_gb'] > 150:
        chunk_size = 200
        print("  Using large chunks (200×200) for optimal speed", flush=True)
    elif initial_mem['available_gb'] > 100:
        chunk_size = 150
        print("  Using medium chunks (150×150) for balanced performance", flush=True)
    else:
        chunk_size = 100
        print("  Using conservative chunks (100×100) due to memory constraints", flush=True)
    
    ds_list = []
    valid_days = 0
    
    for day in range(1, days_in_month + 1):
        fname = f"OMI-MINDS_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc"
        fpath = os.path.join(daily_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing daily file: {fname}", flush=True)
            continue
        
        # Memory check before loading each file
        mem_before = check_memory()
        if mem_before['available_gb'] < 30:  # Critical memory threshold
            print(f"    [WARNING] Low memory before day {day:02d}, cleaning up...", flush=True)
            force_cleanup()
            log_memory(f"After cleanup before day {day:02d}")
            
        try:
            # Use native chunking from the file (like GCHP does)
            try:
                ds = xr.open_dataset(fpath, chunks="auto")
            except (ValueError, ImportError):
                # If dask not available, open without chunking
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'day' dimension
            ds = ds.expand_dims(day=[day])
            # Pick only the NO2 variables to save memory
            ds_subset = ds[["NO2_tot", "NO2_tot_gcshape"]]
            ds_list.append(ds_subset)
            valid_days += 1
            
            print(f"  ✓ Day {day:02d} loaded", flush=True)
            
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}", flush=True)
            continue
   
    if not ds_list:
        print(f"[ERROR] No daily data found for {year}-{month:02d}", flush=True)
        return False
    
    log_memory(f"Loaded {valid_days} days")
    print(f"  Averaging {valid_days}/{days_in_month} days...", flush=True)
    
    try:
        # Concatenate with memory monitoring
        concat_start = time.time()
        ds_all = xr.concat(ds_list, dim="day")
        concat_time = time.time() - concat_start
        
        log_memory("After concatenation")
        print(f"  Concatenation completed in {concat_time:.1f}s", flush=True)
        
        # Compute mean
        mean_start = time.time()
        monthly_mean = ds_all.mean(dim="day", skipna=True)
        mean_time = time.time() - mean_start
        
        log_memory("After mean computation")
        print(f"  Mean computation completed in {mean_time:.1f}s", flush=True)
        
        # Cleanup before writing
        del ds_all, ds_list
        force_cleanup()
        log_memory("After cleanup before writing")
        
        # Add metadata
        monthly_mean.attrs.update({
            'title': f'Monthly mean OMI-MINDS NO2 for {year}-{month:02d}',
            'source': 'OMI-MINDS daily data',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_averaged': valid_days,
            'total_days_in_month': days_in_month,
            'quality_control': qcstr,
            'processed_by_pid': os.getpid(),
            'chunk_size_used': chunk_size,
            'concat_time_seconds': concat_time,
            'mean_time_seconds': mean_time
        })
        
        # Optimized encoding for faster writes
        out_fname = f"OMI-MINDS_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        out_path = os.path.join(monthly_dir, out_fname)
        
        print(f"  Writing monthly mean to {out_path}", flush=True)
        write_start = time.time()
        
        # Use faster compression settings
        encoding = {var: {
            "zlib": True, 
            "complevel": 4,  # Fast compression
            "shuffle": True,
            "fletcher32": True  # Add checksum for data integrity
        } for var in monthly_mean.data_vars}
        
        monthly_mean.to_netcdf(out_path, encoding=encoding)
        
        write_time = time.time() - write_start
        output_size = os.path.getsize(out_path) / 1e6
        
        print(f"  ✓ File written in {write_time:.1f}s ({output_size:.1f} MB)", flush=True)
        print(f"  ✓ Successfully created monthly average for {year}-{month:02d}", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        
        # Force immediate exit to prevent memory issues during cleanup
        print(f"  File successfully written. Exiting cleanly.", flush=True)
        sys.exit(0)
        
    except Exception as e:
        print(f"  [ERROR] Failed to create monthly average for {year}-{month:02d}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def average_monthly_to_yearly_optimized(year):
    """
    Optimized monthly to yearly aggregation with comprehensive safety features
    """
    CloudFraction_max, sza_max, QAlim = 0.1, 75, 0
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = f'/my-projects2/1.project/NO2_col/OMI/{year}/'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    os.makedirs(yearly_dir, exist_ok=True)
    
    print(f"Creating yearly average for {year}", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Available CPU cores: {psutil.cpu_count()}", flush=True)
    print(f"Total system memory: {psutil.virtual_memory().total / 1e9:.1f} GB", flush=True)
    log_memory("Initial")
    
    # Check and report monthly files first
    monthly_files = []
    total_input_size = 0
    
    for month in range(1, 13):
        fname = f"OMI-MINDS_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        fpath = os.path.join(monthly_dir, fname)
        
        if os.path.exists(fpath):
            file_size = os.path.getsize(fpath) / 1e6  # MB
            monthly_files.append((month, fpath, file_size))
            total_input_size += file_size
            print(f"  Found month {month:02d}: {file_size:.1f} MB", flush=True)
        else:
            print(f"  [WARN] Missing monthly file: {fname}", flush=True)
    
    if not monthly_files:
        print(f"[ERROR] No monthly data found for year {year}", flush=True)
        return False
    
    print(f"  Total input data: {total_input_size:.1f} MB ({total_input_size/1000:.1f} GB)", flush=True)
    print(f"  Will process {len(monthly_files)}/12 months", flush=True)
    
    # Adaptive chunking based on available memory and data size
    initial_mem = check_memory()
    estimated_peak_memory = (total_input_size * 1.8) / 1000  # GB, with overhead
    
    print(f"  Estimated peak memory usage: {estimated_peak_memory:.1f} GB", flush=True)
    
    if estimated_peak_memory > initial_mem['available_gb'] * 0.8:
        print(f"  [WARNING] Estimated memory usage is high, using conservative settings", flush=True)
        chunk_size = 100
        use_batch_processing = True
    elif initial_mem['available_gb'] > 150:
        chunk_size = 200
        use_batch_processing = False
        print("  Using large chunks (200×200) for optimal speed", flush=True)
    elif initial_mem['available_gb'] > 100:
        chunk_size = 150
        use_batch_processing = False
        print("  Using medium chunks (150×150) for balanced performance", flush=True)
    else:
        chunk_size = 100
        use_batch_processing = True
        print("  Using conservative chunks (100×100) and batch processing", flush=True)
    
    # Load datasets with comprehensive error handling
    ds_list = []
    
    print("  Loading monthly files with progress monitoring...", flush=True)
    
    for i, (month, fpath, file_size) in enumerate(monthly_files):
        # Memory check before each file
        mem_before = check_memory()
        
        if mem_before['available_gb'] < 40:  # Conservative threshold
            print(f"    [WARNING] Low memory before month {month:02d}, cleaning up...", flush=True)
            force_cleanup()
            
            # If still low memory after cleanup, consider batch processing
            mem_after_gc = check_memory()
            if mem_after_gc['available_gb'] < 30:
                print(f"    [WARNING] Memory critically low, will use batch processing", flush=True)
                use_batch_processing = True
                chunk_size = min(chunk_size, 100)
        
        start_time = time.time()
        
        try:
            # Use native chunking from the file (like GCHP does)
            try:
                ds = xr.open_dataset(fpath, chunks="auto")
            except (ValueError, ImportError):
                # If dask not available, open without chunking
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'month' dimension
            ds = ds.expand_dims(month=[month])
            # Pick only the NO2 variables to minimize memory usage
            ds_subset = ds[["NO2_tot", "NO2_tot_gcshape"]]
            ds_list.append(ds_subset)
            
            load_time = time.time() - start_time
            mem_after = check_memory()
            
            print(f"    ✓ Month {month:02d} loaded in {load_time:.1f}s (mem: {mem_after['percent']:.1f}%)", flush=True)
            
        except Exception as e:
            print(f"    [ERROR] Failed to load month {month:02d}: {str(e)}", flush=True)
            continue
    
    if not ds_list:
        print(f"[ERROR] No monthly data could be loaded for year {year}", flush=True)
        return False
    
    log_memory(f"Loaded {len(ds_list)} monthly files")
    
    # Process data with batch processing if needed
    try:
        if use_batch_processing and len(ds_list) > 6:
            print(f"  Using batch processing for memory safety", flush=True)
            yearly_mean = process_in_batches(ds_list, year, chunk_size)
        else:
            print(f"  Processing all {len(ds_list)} months together...", flush=True)
            yearly_mean = process_all_together(ds_list, year, chunk_size)
        
        if yearly_mean is None:
            return False
        
        # Write output with optimized settings
        out_fname = f"OMI-MINDS_Regrid_{year}_{qcstr}.nc"
        out_path = os.path.join(yearly_dir, out_fname)
        
        return write_yearly_output(yearly_mean, out_path, total_input_size, chunk_size)
        
    except Exception as e:
        print(f"  [ERROR] Failed to create yearly average for {year}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def process_all_together(ds_list, year, chunk_size):
    """Process all months together (faster but more memory)"""
    try:
        # Concatenate with timing
        print("    Concatenating all monthly datasets...", flush=True)
        concat_start = time.time()
        
        ds_all = xr.concat(ds_list, dim="month")
        concat_time = time.time() - concat_start
        
        log_memory("After concatenation")
        print(f"    Concatenation completed in {concat_time:.1f}s", flush=True)
        
        # Compute mean
        print("    Computing yearly mean...", flush=True)
        mean_start = time.time()
        
        yearly_mean = ds_all.mean(dim="month", skipna=True)
        mean_time = time.time() - mean_start
        
        log_memory("After mean computation")
        print(f"    Mean computation completed in {mean_time:.1f}s", flush=True)
        
        # Add timing metadata
        yearly_mean.attrs.update({
            'concat_time_seconds': concat_time,
            'mean_time_seconds': mean_time,
            'processing_method': 'all_together'
        })
        
        # Cleanup
        del ds_all, ds_list
        force_cleanup()
        log_memory("After cleanup")
        
        return yearly_mean
        
    except Exception as e:
        print(f"    [ERROR] Failed in process_all_together: {str(e)}", flush=True)
        return None

def process_in_batches(ds_list, year, chunk_size):
    """Process in smaller batches (slower but safer)"""
    print("    Using batch processing for memory safety...", flush=True)
    
    batch_size = 6  # Process 6 months at a time
    batch_means = []
    
    for i in range(0, len(ds_list), batch_size):
        batch = ds_list[i:i+batch_size]
        batch_months = [i+j+1 for j in range(len(batch))]
        
        print(f"      Processing batch: months {batch_months}", flush=True)
        
        try:
            batch_concat = xr.concat(batch, dim="month")
            batch_mean = batch_concat.mean(dim="month", skipna=True)
            batch_means.append(batch_mean)
            
            # Cleanup batch
            del batch_concat
            force_cleanup()
            
            log_memory(f"After batch {i//batch_size + 1}")
            
        except Exception as e:
            print(f"      [ERROR] Failed processing batch {i//batch_size + 1}: {str(e)}", flush=True)
            return None
    
    # Combine batch results
    print("    Combining batch results...", flush=True)
    try:
        final_concat = xr.concat(batch_means, dim="batch")
        yearly_mean = final_concat.mean(dim="batch", skipna=True)
        
        yearly_mean.attrs.update({
            'processing_method': 'batch_processing',
            'batch_size': batch_size,
            'num_batches': len(batch_means)
        })
        
        # Cleanup
        del final_concat, batch_means
        force_cleanup()
        log_memory("After batch combination")
        
        return yearly_mean
        
    except Exception as e:
        print(f"    [ERROR] Failed combining batches: {str(e)}", flush=True)
        return None

def write_yearly_output(yearly_mean, out_path, total_input_size, chunk_size):
    """Write yearly output with optimized settings"""
    print(f"  Writing yearly mean to {out_path}", flush=True)
    write_start = time.time()
    
    # Add comprehensive metadata
    yearly_mean.attrs.update({
        'title': f'Yearly mean OMI-MINDS NO2 for {yearly_mean.attrs.get("year", "unknown")}',
        'source': 'OMI-MINDS monthly averages',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processed_by_pid': os.getpid(),
        'processing_host': os.uname().nodename,
        'chunk_size_used': chunk_size,
        'total_input_size_mb': total_input_size
    })
    
    # Use consistent compression level (complevel=4 like monthly files)
    complevel = 4
    print("    Using complevel=4 compression for consistency with monthly files", flush=True)
    
    # Optimized encoding
    encoding = {}
    for var in yearly_mean.data_vars:
        encoding[var] = {
            "zlib": True,
            "complevel": complevel,
            "shuffle": True,
            "fletcher32": True  # Add checksum for data integrity
        }
    
    try:
        yearly_mean.to_netcdf(out_path, encoding=encoding)
        
        write_time = time.time() - write_start
        output_size = os.path.getsize(out_path) / 1e6
        compression_ratio = total_input_size / output_size if output_size > 0 else 0
        
        print(f"  ✓ File written in {write_time:.1f}s", flush=True)
        print(f"  ✓ Output file size: {output_size:.1f} MB", flush=True)
        print(f"  ✓ Compression ratio: {compression_ratio:.1f}x", flush=True)
        print(f"  ✓ Successfully created yearly average", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        
        # Force immediate exit to prevent memory issues during cleanup
        print(f"  File successfully written. Exiting cleanly.", flush=True)
        sys.exit(0)
        
    except Exception as e:
        print(f"  [ERROR] Failed to write output file: {str(e)}", flush=True)
        return False

def plot_tropomi(ds, title, out_png):
    """Plot TROPOMI NO2 data with error handling"""
    try:
        # Load grid arrays
        x = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
        y = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot", flush=True)
        return

    try:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        vars_no2 = [v for v in ds.data_vars if 'NO2' in v]
        
        for ax, var in zip(axes, vars_no2):
            v = ds[var].values
            if v.size == 0 or np.all(np.isnan(v)):
                print(f"[WARN] variable {var} has no valid data for plot '{title}'; skipping", flush=True)
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
                                 vmin=0, vmax=4e-6)
            cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                                pad=0.05, fraction=0.05)
            cbar.set_label(var)
            ax.set_title(f"{title}: {var}", pad=10)
        plt.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Plot saved: {out_png}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create plot: {str(e)}", flush=True)

def main():
    """Main processing function with comprehensive error handling"""
    parser = argparse.ArgumentParser(description='Process TROPOMI NO2 data averaging (Optimized & Safe)')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=int, metavar='MONTH', choices=range(1, 13),
                       help='Process specific month only (1-12)')
    parser.add_argument('--yearly-only', action='store_true', 
                       help='Only create yearly average from existing monthly files')
    parser.add_argument('--no-plot', action='store_true',
                        help="Don't make PNG plots")
    args = parser.parse_args()
    year = args.year
    
    # Enable unbuffered output for real-time monitoring
    sys.stdout.reconfigure(line_buffering=True)
    
    print(f"=== TROPOMI NO2 PROCESSING (OPTIMIZED & SAFE) ===", flush=True)
    print(f"Processing year {year}", flush=True)
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}", flush=True)
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}", flush=True)
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    print(f"Script start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Suppress xarray chunking warnings for cleaner output
    warnings.filterwarnings("ignore", message=".*chunks separate.*")
    
    # Set up quality control string
    CloudFraction_max, sza_max, QAlim = 0.1, 75, 0
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = f'/my-projects2/1.project/NO2_col/OMI/{year}/'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    
    try:
        start_time = datetime.now()
        success = False
        
        if args.yearly_only:
            # Only create yearly average
            success = average_monthly_to_yearly_optimized(year)
            
            # Create plot if requested
            if success and not args.no_plot:
                yearly_path = os.path.join(yearly_dir, f"OMI-MINDS_Regrid_{year}_{qcstr}.nc")
                if os.path.exists(yearly_path):
                    print("Creating visualization plot...", flush=True)
                    ds_y = xr.open_dataset(yearly_path)
                    out_png = os.path.join(yearly_dir, f"OMI-MINDS_Regrid_{year}_plot.png")
                    plot_tropomi(ds_y, f"{year} Yearly", out_png)
                    
        elif args.month:
            # Process specific month only
            success = average_daily_to_monthly_optimized(year, args.month)
            
            # Create plot if requested
            if success and not args.no_plot:
                monthly_path = os.path.join(monthly_dir, f"OMI-MINDS_Regrid_{year}{args.month:02d}_Monthly_{qcstr}.nc")
                if os.path.exists(monthly_path):
                    print(f"Creating visualization plot for {year}-{args.month:02d}...", flush=True)
                    ds_m = xr.open_dataset(monthly_path)
                    out_png = os.path.join(monthly_dir, f"OMI-MINDS_Regrid_{year}{args.month:02d}_plot.png")
                    plot_tropomi(ds_m, f"{year}-{args.month:02d}", out_png)
        else:
            print("Error: Please specify either --month N or --yearly-only", flush=True)
            print("For parallel processing, use separate jobs for each month", flush=True)
            sys.exit(1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            if args.month:
                print(f"\n✓ Successfully completed monthly processing for {year}-{args.month:02d}", flush=True)
            else:
                print(f"\n✓ Successfully completed yearly processing for {year}", flush=True)
            print(f"Total processing time: {duration}", flush=True)
            log_memory("Final")
            sys.exit(0)
        else:
            print(f"\n✗ Failed to process {year}", flush=True)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠ Processing interrupted for year {year}", flush=True)
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error processing year {year}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()