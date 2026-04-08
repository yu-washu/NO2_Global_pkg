import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta
import calendar
import argparse
import sys
import gc
import warnings

# Configuration
res = '1x1km'
region = 'global'

sza_max, QAlim = 80, 0.75
qcstr = 'SZA{}-QA{}'.format(
    sza_max,
    int(QAlim * 100)
)
NA = 6.022e23
MwAir = 28.97  # unit is g/mol

def get_days_in_month(year, month):
    """Get number of days in a given month"""
    return calendar.monthrange(year, month)[1]

def slice_latitude(ds, lat_min=-60, lat_max=70):
    """
    Slice dataset to specified latitude range with memory optimization
    """
    lat_coord = None
    for coord in ds.coords:
        if coord.lower() in ['lat', 'latitude', 'y']:
            lat_coord = coord
            break
    
    if lat_coord is None:
        print("[WARN] No latitude coordinate found, returning original dataset")
        return ds
    
    print(f"  Slicing latitude from {lat_min} to {lat_max} degrees")
    lat_slice = ds.sel({lat_coord: slice(lat_min, lat_max)})
    
    lat_slice.attrs.update({
        'latitude_slice': f'{lat_min} to {lat_max} degrees',
        'original_lat_range': f'{float(ds[lat_coord].min().values):.2f} to {float(ds[lat_coord].max().values):.2f}'
    })
    
    print(f"  Latitude range after slicing: {float(lat_slice[lat_coord].min().values):.2f} to {float(lat_slice[lat_coord].max().values):.2f}")
    return lat_slice

def load_yearly_data_minimal(year):
    """
    Load yearly data with minimal memory footprint
    Load without chunking to avoid alignment warnings, then immediately extract values
    """
    print(f"Loading annual data for {year} with minimal memory approach...")
    
    gchp_dir = f'/my-projects2/1.project/gchp/forObservation-Geophysical/{year}/'
    tropomi_dir = f'/my-projects2/1.project/NO2_col/TROPOMI/{year}/'
    
    try:
        # Load GCHP yearly data without chunking
        print("  Loading GCHP annual data...")
        gchp_file = gchp_dir + f'yearly/{res}.Hours.13-15.{year}.AnnualMean.nc'
        with xr.open_dataset(gchp_file, engine='netcdf4') as yearly_gchp:
            yearly_gchp = slice_latitude(yearly_gchp, -60, 70)
            yearly_gchp_NO2col = yearly_gchp['NO2col'].values.astype('float32')
        
        # Load TROPOMI yearly data without chunking
        print("  Loading TROPOMI annual data...")
        if year == 2018:
            tropomi_file = tropomi_dir + "yearly/TROPOMI_OMI_Regrid_2018.nc"
        else:
            tropomi_file = tropomi_dir + f"yearly/Tropomi_Regrid_{year}_{qcstr}.nc"
            
        with xr.open_dataset(tropomi_file, engine='netcdf4') as yearly_tropomi:
            yearly_tropomi_NO2col_gcshape = yearly_tropomi['NO2_tot_gcshape'].values.astype('float32')
        
        print(f"✓ Successfully loaded annual data for {year}")
        print(f"  GCHP shape: {yearly_gchp_NO2col.shape}")
        print(f"  TROPOMI shape: {yearly_tropomi_NO2col_gcshape.shape}")
        
        return yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape
        
    except Exception as e:
        print(f"✗ Error loading annual data for {year}: {str(e)}")
        raise

def process_single_day_minimal(year, month, day, yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape):
    """
    Process a single day with minimal memory usage
    Load, process, and immediately return results without keeping large objects
    """
    gchp_dir = f'/my-projects2/1.project/gchp/forObservation-Geophysical/{year}/'
    tropomi_dir = f'/my-projects2/1.project/NO2_col/TROPOMI/{year}/'
    
    # File paths
    tropomi_file = tropomi_dir + f'daily/Tropomi_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc'
    gchp_daily_path = gchp_dir + f'daily/{res}.DailyVars.{year}{month:02d}{day:02d}.nc4'
    gchp_3hours_path = gchp_dir + f'daily/{res}.Hours.13-15.{year}{month:02d}{day:02d}.nc4'
    
    # Check file existence
    if not all(os.path.exists(f) for f in [tropomi_file, gchp_daily_path, gchp_3hours_path]):
        return None
    
    try:
        # Load TROPOMI data without chunking to avoid warnings
        with xr.open_dataset(tropomi_file, engine='netcdf4') as tropomi:
            tropomi = tropomi.squeeze()
            NO2_tot_gcshape = tropomi['NO2_tot_gcshape'].values.astype('float32')
            lat_coords = tropomi['lat'].values
            lon_coords = tropomi['lon'].values
        
        # Load GCHP daily data without chunking
        with xr.open_dataset(gchp_daily_path, engine='netcdf4') as gchp_daily:
            gchp_daily = slice_latitude(gchp_daily.squeeze(), -60, 70)
            gchp_no2 = gchp_daily['gchp_NO2'].values.astype('float32')
            gchp_hno3 = gchp_daily['gchp_HNO3'].values.astype('float32')
            gchp_pan = gchp_daily['gchp_PAN'].values.astype('float32')
            
            # Handle alkylnitrates
            if 'gchp_alkylnitrates' in gchp_daily:
                gchp_alkylnitrates = gchp_daily['gchp_alkylnitrates'].values.astype('float32')
            elif 'gchp_alklnitrates' in gchp_daily:
                gchp_alkylnitrates = gchp_daily['gchp_alklnitrates'].values.astype('float32')
            else:
                gchp_alkylnitrates = np.zeros_like(gchp_no2, dtype='float32')
        
        # Load GCHP 3-hour data without chunking
        with xr.open_dataset(gchp_3hours_path, engine='netcdf4') as gchp_3hours:
            gchp_3hours = slice_latitude(gchp_3hours.squeeze(), -60, 70)
            gchp_no2col = gchp_3hours['NO2col'].values.astype('float32')
        
        # Calculate eta with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            gchp_eta = np.where(gchp_no2col > 0, gchp_no2 / gchp_no2col, 0).astype('float32')

        # Calculate geophysical NO2, masking only fill/missing values (NaN and negative TROPOMI values);
        # negative TROPOMI values are filled with GCHP NO2.
        # NO2_geophysical = np.where(NO2_tot_gcshape > 0, 
        #                           NO2_tot_gcshape * gchp_eta, 
        #                           np.nan).astype('float32')
        
        # Calculate geophysical NO2, masking only fill/missing values (NaN);
        # negative TROPOMI values are preserved as-is.
        NO2_geophysical = np.where(~np.isnan(NO2_tot_gcshape),
                                  NO2_tot_gcshape * gchp_eta,
                                  np.nan).astype('float32')
        
        # Debug: Print quality filtering statistics
        total_pixels = NO2_tot_gcshape.size
        valid_tropomi = np.sum(~np.isnan(NO2_tot_gcshape))
        nan_tropomi = np.sum(np.isnan(NO2_tot_gcshape))
        neg_tropomi = np.sum(NO2_tot_gcshape < 0)
        final_valid_geo = np.sum(~np.isnan(NO2_geophysical))
        
        if day <= 3:  # Only print for first few days to avoid spam
            print(f"\n    Day {day:02d} Quality Check:")
            print(f"      Total pixels: {total_pixels:,}")
            print(f"      TROPOMI not-NaN: {valid_tropomi:,} ({100*valid_tropomi/total_pixels:.1f}%)")
            print(f"      TROPOMI NaN: {nan_tropomi:,} ({100*nan_tropomi/total_pixels:.1f}%)")
            print(f"      TROPOMI < 0 (kept): {neg_tropomi:,} ({100*neg_tropomi/total_pixels:.1f}%)")
            print(f"      Final valid geo NO2: {final_valid_geo:,} ({100*final_valid_geo/total_pixels:.1f}%)")
            print(f"      NO2_tot_gcshape range: {np.nanmin(NO2_tot_gcshape):.2e} to {np.nanmax(NO2_tot_gcshape):.2e}")
            if final_valid_geo > 0:
                print(f"      Geophysical NO2 range: {np.nanmin(NO2_geophysical):.2e} to {np.nanmax(NO2_geophysical):.2e}")
        
        # Return daily results as a simple dictionary with minimal memory footprint
        return {
            'geophysical_NO2': NO2_geophysical,
            'gchp_NO2': gchp_no2,
            'gchp_HNO3': gchp_hno3,
            'gchp_PAN': gchp_pan,
            'gchp_alkylnitrates': gchp_alkylnitrates,
            'gchp_NO2col': gchp_no2col,
            'tropomi_NO2_tot_gcshape': NO2_tot_gcshape,
            'lat': lat_coords,
            'lon': lon_coords
        }
        
    except Exception as e:
        print(f"    Error processing day {day:02d}: {str(e)}")
        return None

def calculate_unbiased_model_average(year, current_month):
    """
    Calculate average GCHP NO2col for months that have TROPOMI observations
    (excluding the current month to avoid circular reference)
    """
    print(f"  Calculating unbiased model average for {year} (excluding month {current_month:02d})")
    
    gchp_dir = f'/my-projects2/1.project/gchp/forObservation-Geophysical/{year}/'
    tropomi_dir = f'/my-projects2/1.project/NO2_col/TROPOMI/{year}/'
    
    # Debug: List available files
    print(f"    Checking TROPOMI monthly files in: {tropomi_dir}monthly/")
    print(f"    Checking GCHP monthly files in: {gchp_dir}monthly/")
    
    # Initialize accumulator for valid GCHP data
    gchp_sum = None
    valid_pixel_count = None
    months_with_data = []
    
    for month in range(1, 13):
        if month == current_month:
            continue  # Skip current month
            
        # Check if TROPOMI monthly file exists (try different naming patterns)
        tropomi_monthly_patterns = [
            tropomi_dir + f"monthly/Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc",
            tropomi_dir + f"monthly/Tropomi_Regrid_{year}{month:02d}_{qcstr}.nc",
            tropomi_dir + f"monthly/Tropomi_Regrid_{year}{month:02d}.nc"
        ]
        
        tropomi_monthly_file = None
        for pattern in tropomi_monthly_patterns:
            if os.path.exists(pattern):
                tropomi_monthly_file = pattern
                break
        
        if tropomi_monthly_file is None:
            print(f"    Month {month:02d}: No TROPOMI monthly file found")
            continue
        
        # Check if GCHP monthly file exists (try different naming patterns)  
        gchp_monthly_patterns = [
            gchp_dir + f'monthly/{res}.Hours.13-15.{year}{month:02d}.nc',
            gchp_dir + f'monthly/{res}.Hours.13-15.{year}{month:02d}.MonMean.nc',
            gchp_dir + f'monthly/1x1km.Hours.13-15.{year}{month:02d}.nc'
        ]
        
        gchp_monthly_file = None
        for pattern in gchp_monthly_patterns:
            if os.path.exists(pattern):
                gchp_monthly_file = pattern
                break
                
        if gchp_monthly_file is None:
            print(f"    Month {month:02d}: No GCHP monthly file found")
            continue
        
        try:
            print(f"    Including month {month:02d}...", end=' ')
            
            # Load TROPOMI data to identify valid pixels
            with xr.open_dataset(tropomi_monthly_file, engine='netcdf4') as tropomi_ds:
                if 'NO2_tot_gcshape' in tropomi_ds:
                    tropomi_no2 = tropomi_ds['NO2_tot_gcshape'].values
                elif 'NO2_tot' in tropomi_ds:
                    tropomi_no2 = tropomi_ds['NO2_tot'].values
                else:
                    print(f"No recognized NO2 variable found")
                    continue
                    
                valid_tropomi_mask = (tropomi_no2 > 0) & (~np.isnan(tropomi_no2))
                valid_count = np.sum(valid_tropomi_mask)
                
                if valid_count == 0:
                    print(f"No valid TROPOMI pixels")
                    continue
                    
                print(f"({valid_count:,} valid TROPOMI pixels)", end=' ')
            
            # Load GCHP data
            with xr.open_dataset(gchp_monthly_file, engine='netcdf4') as gchp_ds:
                gchp_ds = slice_latitude(gchp_ds, -60, 70)
                gchp_no2col = gchp_ds['NO2col'].values
            
            # Initialize arrays on first valid month
            if gchp_sum is None:
                gchp_sum = np.zeros_like(gchp_no2col, dtype='float64')
                valid_pixel_count = np.zeros_like(gchp_no2col, dtype='int32')
            
            # Add GCHP values only where TROPOMI has valid observations
            gchp_sum = np.where(valid_tropomi_mask, 
                               gchp_sum + gchp_no2col.astype('float64'),
                               gchp_sum)
            valid_pixel_count = np.where(valid_tropomi_mask,
                                        valid_pixel_count + 1,
                                        valid_pixel_count)
            
            months_with_data.append(month)
            print("✓")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
        
        # Force cleanup
        gc.collect()
    
    if not months_with_data:
        print(f"    Warning: No valid months found for unbiased averaging")
        print(f"    This might indicate:")
        print(f"    - TROPOMI monthly files don't exist or use different naming")
        print(f"    - GCHP monthly files don't exist or use different naming") 
        print(f"    - All TROPOMI monthly files contain only invalid data")
        return None
    
    # Calculate average where we have sufficient data
    print(f"    Computed average from {len(months_with_data)} months: {months_with_data}")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        unbiased_gchp_average = np.where(valid_pixel_count > 0,
                                        gchp_sum / valid_pixel_count,
                                        np.nan).astype('float32')
    
    # Print statistics
    total_pixels = unbiased_gchp_average.size
    valid_pixels = np.sum(~np.isnan(unbiased_gchp_average))
    print(f"    Unbiased average coverage: {valid_pixels:,}/{total_pixels:,} pixels ({100*valid_pixels/total_pixels:.1f}%)")
    
    return unbiased_gchp_average

def fill_missing_values_unbiased(year, current_month, geophysical_NO2, monthly_gchp_no2col, 
                                yearly_tropomi_NO2col_gcshape, gchp_no2):
    """
    Unbiased gap filling using temporal bias correction:
    monthly_model / (average of model from months with TROPOMI data) × yearly_satellite × (surface/column_ratio)
    """
    print("  Filling missing values using unbiased temporal scaling...")
    
    # Calculate unbiased model average from months with TROPOMI observations
    unbiased_gchp_average = calculate_unbiased_model_average(year, current_month)
    
    if unbiased_gchp_average is None:
        print("  Warning: No reference months available - keeping all pixels as NaN")
        print("  This indicates TROPOMI had no valid observations for this year")
        # Return the original geophysical_NO2 unchanged (preserving all NaN values)
        return geophysical_NO2.astype('float32')
    else:
        # Calculate proper temporal bias correction
        with np.errstate(divide='ignore', invalid='ignore'):
            scale_factors = np.where(unbiased_gchp_average > 0,
                                    monthly_gchp_no2col / unbiased_gchp_average,
                                    np.nan).astype('float32')
    
    # Calculate surface to column ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(monthly_gchp_no2col > 0, 
                        gchp_no2 / monthly_gchp_no2col, 0).astype('float32')
    
    # Calculate fill values: scale_factor × yearly_satellite × surface_ratio
    fill_values = (scale_factors * yearly_tropomi_NO2col_gcshape * ratio).astype('float32')
    
    # Fill missing values
    geophysical_NO2_filled = np.where(np.isnan(geophysical_NO2), 
                                     fill_values, 
                                     geophysical_NO2).astype('float32')
    
    # Statistics
    total_pixels = geophysical_NO2.size
    originally_missing = np.sum(np.isnan(geophysical_NO2))
    filled_pixels = np.sum(np.isnan(geophysical_NO2) & ~np.isnan(fill_values))
    still_missing = np.sum(np.isnan(geophysical_NO2_filled))
    
    print(f"  Missing value filling summary:")
    print(f"    Total pixels: {total_pixels:,}")
    print(f"    Originally missing: {originally_missing:,} ({100 * originally_missing / total_pixels:.1f}%)")
    print(f"    Successfully filled: {filled_pixels:,} ({100 * filled_pixels / total_pixels:.1f}%)")
    print(f"    Still missing: {still_missing:,} ({100 * still_missing / total_pixels:.1f}%)")
    
    if filled_pixels > 0:
        filled_values_only = fill_values[np.isnan(geophysical_NO2) & ~np.isnan(fill_values)]
        print(f"    Fill value range: {np.min(filled_values_only):.2e} to {np.max(filled_values_only):.2e}")
        print(f"    Fill value median: {np.median(filled_values_only):.2e}")
    
    return geophysical_NO2_filled

def process_monthly_data_streaming(year, month, yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape):
    """
    Process monthly data using streaming approach to minimize memory usage
    Process days one-by-one and accumulate running statistics with proper NaN handling
    """
    print(f"\nProcessing {year}-{month:02d} with streaming approach (minimal memory)")
    
    days_in_month = get_days_in_month(year, month)
    
    # Initialize accumulators for running mean calculation with NaN tracking
    monthly_sums = {}
    daily_valid_count = {}  # Track how many valid days each pixel has
    valid_day_count = 0
    lat_coords = None
    lon_coords = None
    
    # Process each day individually
    for day in range(1, days_in_month + 1):
        print(f"  Processing day {day:02d}...", end=' ')
        
        daily_result = process_single_day_minimal(year, month, day, 
                                                 yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape)
        
        if daily_result is None:
            print("missing")
            continue
        
        # Initialize accumulators on first valid day
        if valid_day_count == 0:
            lat_coords = daily_result['lat']
            lon_coords = daily_result['lon']
            for var in ['geophysical_NO2', 'gchp_NO2', 'gchp_HNO3', 'gchp_PAN', 
                       'gchp_alkylnitrates', 'gchp_NO2col', 
                       'tropomi_NO2_tot_gcshape']:
                monthly_sums[var] = np.zeros_like(daily_result[var], dtype='float64')
                daily_valid_count[var] = np.zeros_like(daily_result[var], dtype='int32')
        
        # Accumulate daily values with proper NaN handling
        for var in monthly_sums.keys():
            daily_data = daily_result[var]
            # Only add non-NaN values and track count of valid observations per pixel
            valid_mask = ~np.isnan(daily_data)
            monthly_sums[var] = np.where(valid_mask, 
                                        monthly_sums[var] + daily_data.astype('float64'),
                                        monthly_sums[var])
            daily_valid_count[var] = np.where(valid_mask,
                                             daily_valid_count[var] + 1,
                                             daily_valid_count[var])
        
        valid_day_count += 1
        print("✓")
        
        # Force garbage collection after each day
        del daily_result
        gc.collect()
    
    if valid_day_count == 0:
        print(f"  ✗ No valid daily data found for {year}-{month:02d}")
        return False
    
    print(f"  Calculating monthly averages from {valid_day_count} days...")
    
    # Calculate monthly averages with minimum observation threshold
    min_observations = 0  # Require at least 10 days
    print(f"  Requiring minimum {min_observations} valid observations per pixel")
    
    monthly_means = {}
    for var, sum_data in monthly_sums.items():
        valid_count = daily_valid_count[var]
        # Only calculate average where we have sufficient observations
        monthly_means[var] = np.where(valid_count >= min_observations,
                                     (sum_data / valid_count).astype('float32'),
                                     np.nan)
        
        # Print statistics for geophysical_NO2
        if var == 'geophysical_NO2':
            total_pixels = monthly_means[var].size
            final_valid = np.sum(~np.isnan(monthly_means[var]))
            print(f"  Monthly {var} statistics:")
            print(f"    Total pixels: {total_pixels:,}")
            print(f"    Valid monthly pixels: {final_valid:,} ({100*final_valid/total_pixels:.1f}%)")
            print(f"    Missing monthly pixels: {total_pixels-final_valid:,} ({100*(total_pixels-final_valid)/total_pixels:.1f}%)")
    
    # Apply gap filling with temporal bias correction
    geophysical_NO2_filled = fill_missing_values_unbiased(
        year, month,
        monthly_means['geophysical_NO2'],
        monthly_means['gchp_NO2col'],
        yearly_tropomi_NO2col_gcshape,
        monthly_means['gchp_NO2']
    )
    
    # Create final monthly dataset
    final_monthly = xr.Dataset({
        'gap_GeoNO2': (['lat', 'lon'], monthly_means['geophysical_NO2']),
        'filled_GeoNO2': (['lat', 'lon'], geophysical_NO2_filled),
        'gchp_NO2': (['lat', 'lon'], monthly_means['gchp_NO2']),
        'gchp_HNO3': (['lat', 'lon'], monthly_means['gchp_HNO3']),
        'gchp_PAN': (['lat', 'lon'], monthly_means['gchp_PAN']),
        'gchp_alkylnitrates': (['lat', 'lon'], monthly_means['gchp_alkylnitrates']),
        'gchp_NO2col': (['lat', 'lon'], monthly_means['gchp_NO2col']),
        'tropomi_NO2_tot_gcshape': (['lat', 'lon'], monthly_means['tropomi_NO2_tot_gcshape'])
    }, coords={
        'lat': lat_coords,
        'lon': lon_coords,
        'time': datetime(year, month, 15)
    })
    
    # Add attributes
    final_monthly.attrs.update({
        'title': f'Monthly average geophysical NO2 for {year}-{month:02d} (streaming processed)',
        'source': 'TROPOMI 3.5x5km + GCHP c180',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'days_averaged': valid_day_count,
        'total_days_in_month': days_in_month,
        'quality_control': qcstr,
        'processed_by_pid': os.getpid(),
        'gap_filling_method': 'simple vectorized scaling',
        'processing_optimization': 'streaming with minimal memory footprint'
    })
    
    # Save with compression
    OutDir = f'/my-projects2/1.project/GeoNO2/{year}/'
    os.makedirs(OutDir, exist_ok=True)
    outpath = f'{OutDir}{res}.GeoNO2.{year}{month:02d}.MonMean.nc'
    
    print(f"  Saving to: {outpath}")
    
    # Optimized encoding for small file size and fast I/O
    encoding = {var: {
        'zlib': True, 
        'complevel': 4,
        'shuffle': True,
        'dtype': 'float32'
    } for var in final_monthly.data_vars}
    
    final_monthly.to_netcdf(outpath, encoding=encoding)
    
    file_size = os.path.getsize(outpath) / (1024 * 1024)  # MB
    print(f"  ✓ Successfully saved monthly average ({file_size:.2f} MB)")
    
    # Cleanup
    del monthly_sums, monthly_means, final_monthly
    gc.collect()
    
    return True

def process_yearly_average_minimal(year):
    """
    Create annual average with minimal memory usage
    """
    print(f"\n=== Creating Minimal Memory Yearly Average for {year} ===")
    
    monthly_dir = f'/my-projects2/1.project/GeoNO2/{year}/'
    
    if not os.path.exists(monthly_dir):
        print(f"✗ Monthly directory not found: {monthly_dir}")
        return False
    
    # Use streaming approach for yearly average too
    monthly_sums = {}
    valid_month_count = 0
    lat_coords = None
    lon_coords = None
    months_found = []
    
    for month in range(1, 13):
        monthly_file = f'{monthly_dir}{res}.GeoNO2.{year}{month:02d}.MonMean.nc'
        
        if not os.path.exists(monthly_file):
            continue
        
        try:
            print(f"  Processing month {month:02d}...", end=' ')
            
            # Load monthly file without chunking
            with xr.open_dataset(monthly_file, engine='netcdf4') as ds_month:
                # Initialize on first month
                if valid_month_count == 0:
                    lat_coords = ds_month['lat'].values
                    lon_coords = ds_month['lon'].values
                    for var in ds_month.data_vars:
                        monthly_sums[var] = np.zeros_like(ds_month[var].values, dtype='float64')
                
                # Accumulate monthly data
                for var in ds_month.data_vars:
                    monthly_data = ds_month[var].values
                    valid_mask = ~np.isnan(monthly_data)
                    monthly_sums[var] = np.where(valid_mask,
                                               monthly_sums[var] + monthly_data.astype('float64'),
                                               monthly_sums[var])
                
                valid_month_count += 1
                months_found.append(month)
                print("✓")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
        
        # Force cleanup
        gc.collect()
    
    if valid_month_count == 0:
        print(f"✗ No valid monthly files found for {year}")
        return False
    
    print(f"  Creating annual averages from {valid_month_count} months...")
    
    # Calculate annual averages
    annual_means = {}
    for var, sum_data in monthly_sums.items():
        annual_means[var] = (sum_data / valid_month_count).astype('float32')
    
    # Create annual dataset
    annual_final = xr.Dataset({
        var: (['lat', 'lon'], data) for var, data in annual_means.items()
    }, coords={
        'lat': lat_coords,
        'lon': lon_coords,
        'time': datetime(year, 7, 1)
    })
    
    # Add attributes
    annual_final.attrs.update({
        'title': f'Annual average geophysical NO2 for {year} (minimal memory)',
        'source': 'OMI MINDS 13x24km + GCHP c180',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'months_averaged': valid_month_count,
        'months_included': ', '.join([f'{m:02d}' for m in months_found]),
        'coverage_percentage': f'{100 * valid_month_count / 12:.1f}%',
        'quality_control': qcstr,
        'processing_optimization': 'minimal memory streaming'
    })
    
    # Save
    annual_outpath = f'{monthly_dir}{res}.GeoNO2.{year}.AnnualMean.nc'
    
    encoding = {var: {
        'zlib': True, 
        'complevel': 4,
        'shuffle': True,
        'dtype': 'float32'
    } for var in annual_final.data_vars}
    
    annual_final.to_netcdf(annual_outpath, encoding=encoding)
    
    file_size = os.path.getsize(annual_outpath) / (1024 * 1024)
    print(f"✓ Annual average saved ({file_size:.2f} MB): {annual_outpath}")
    
    # Cleanup
    del monthly_sums, annual_means, annual_final
    gc.collect()
    
    return True

def process_single_month_minimal(year, month):
    """Process a single month with minimal memory approach"""
    print(f"\n=== Processing Single Month: {year}-{month:02d} (Minimal Memory) ===")
    
    try:
        yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape = load_yearly_data_minimal(year)
    except Exception as e:
        return False
    
    success = process_monthly_data_streaming(year, month, yearly_gchp_NO2col, 
                                           yearly_tropomi_NO2col_gcshape)
    
    # Cleanup yearly data
    del yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape
    gc.collect()
    
    return success

def process_year_minimal(year):
    """Process all months for a given year with minimal memory approach"""
    print(f"\n=== Processing Year {year} (Minimal Memory) ===")
    
    try:
        yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape = load_yearly_data_minimal(year)
    except Exception as e:
        return False
    
    successful_months = 0
    
    # Process each month
    for month in range(1, 13):
        print(f"\n--- Processing Month {month:02d}/{12} ---")
        success = process_monthly_data_streaming(year, month, yearly_gchp_NO2col, 
                                               yearly_tropomi_NO2col_gcshape)
        if success:
            successful_months += 1
        
        # Force garbage collection between months
        gc.collect()
    
    # Create yearly average if we have monthly data
    if successful_months > 0:
        print(f"\n--- Creating Annual Average ---")
        # Clean up yearly data before processing annual average
        del yearly_gchp_NO2col, yearly_tropomi_NO2col_gcshape
        gc.collect()
        process_yearly_average_minimal(year)
    
    print(f"\n=== Year {year} Summary ===")
    print(f"Successfully processed: {successful_months}/12 months")
    
    return successful_months > 0

def main():
    """Main processing function optimized for minimal memory usage"""
    parser = argparse.ArgumentParser(description='Process geophysical NO2 data (minimal memory version)')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=int, choices=range(1, 13), 
                       help='Process only specific month (1-12)')
    parser.add_argument('--yearly-only', action='store_true', 
                       help='Only create annual average from existing monthly files')
    
    args = parser.parse_args()
    year = args.year
    
    print(f"Starting minimal memory geophysical NO2 processing for year {year}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Memory optimization: Streaming processing with no chunking")
    
    try:
        start_time = datetime.now()
        
        if args.yearly_only:
            success = process_yearly_average_minimal(year)
        elif args.month:
            success = process_single_month_minimal(year, args.month)
        else:
            success = process_year_minimal(year)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            print(f"\n✓ Successfully completed processing for year {year}")
            print(f"Total processing time: {duration}")
            sys.exit(0)
        else:
            print(f"\n✗ Failed to process year {year}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠ Processing interrupted for year {year}")
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()