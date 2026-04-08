#!/usr/bin/env python3
import xarray as xr
import numpy as np
import os
import argparse
import sys
from datetime import datetime
import glob

def extract_monthly_filled_geno2(year, month, input_dir, output_dir, gchp_outdir):
    """
    Extract filled_GeoNO2 from a single monthly NetCDF file and save as numpy array
    """
    # Input NetCDF file path
    nc_file = f'{input_dir}1x1km.GeoNO2.{year}{month:02d}.MonMean.nc'
    
    if not os.path.exists(nc_file):
        print(f"  ✗ NetCDF file not found: {nc_file}")
        return False
    
    try:
        print(f"  Processing {year}-{month:02d}...", end=' ')
        
        # Load the NetCDF file
        with xr.open_dataset(nc_file, engine='netcdf4') as ds:
            # Check if filled_GeoNO2 variable exists
            if 'filled_GeoNO2' not in ds:
                print(f"✗ 'filled_GeoNO2' variable not found in {nc_file}")
                print(f"    Available variables: {list(ds.data_vars.keys())}")
                return False
            
            # Extract the filled_GeoNO2 data
            filled_geno2 = ds['filled_GeoNO2'].values
            # gchp_no2 = ds['gchp_NO2'].values
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(gchp_outdir, exist_ok=True)
        # Output numpy file path
        npy_file = f'{output_dir}GeoNO2_001x001_Global_map_{year}{month:02d}.npy'
        # gchp_file = f'{gchp_outdir}gchp_NO2_001x001_Global_map_{year}{month:02d}.npy'
        
        # Save as numpy array
        np.save(npy_file, filled_geno2)
        # np.save(gchp_file, gchp_no2)
        
        # Get file sizes for reporting
        nc_size = os.path.getsize(nc_file) / (1024 * 1024)  # MB
        npy_size = os.path.getsize(npy_file) / (1024 * 1024)  # MB
        
        print(f"✓ ({nc_size:.1f}MB → {npy_size:.1f}MB)")
        
        # Print data statistics
        valid_pixels = np.sum(~np.isnan(filled_geno2))
        total_pixels = filled_geno2.size
        data_range = (np.nanmin(filled_geno2), np.nanmax(filled_geno2))
        
        print(f"    Data shape: {filled_geno2.shape}")
        print(f"    Valid pixels: {valid_pixels:,}/{total_pixels:,} ({100*valid_pixels/total_pixels:.1f}%)")
        print(f"    Data range: {data_range[0]:.2e} to {data_range[1]:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def extract_year_filled_geno2(year, input_dir, output_dir, gchp_outdir):
    """
    Extract filled_GeoNO2 from all monthly NetCDF files for a given year
    """
    print(f"\n=== Extracting filled_GeoNO2 for year {year} ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GCHP output directory: {gchp_outdir}")
    
    successful_months = 0
    failed_months = []
    
    for month in range(1, 13):
        success = extract_monthly_filled_geno2(year, month, input_dir, output_dir, gchp_outdir)
        if success:
            successful_months += 1
        else:
            failed_months.append(month)
    
    print(f"\n=== Year {year} Summary ===")
    print(f"Successfully extracted: {successful_months}/12 months")
    if failed_months:
        print(f"Failed months: {[f'{m:02d}' for m in failed_months]}")
    
    return successful_months

def extract_single_month_filled_geno2(year, month, input_dir, output_dir, gchp_outdir):
    """
    Extract filled_GeoNO2 from a single monthly NetCDF file
    """
    print(f"\n=== Extracting filled_GeoNO2 for {year}-{month:02d} ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GCHP output directory: {gchp_outdir}")
    
    success = extract_monthly_filled_geno2(year, month, input_dir, output_dir, gchp_outdir)
    
    if success:
        print(f"\n✓ Successfully extracted {year}-{month:02d}")
    else:
        print(f"\n✗ Failed to extract {year}-{month:02d}")
    
    return success

def find_available_files(input_dir, year):
    """
    Find all available NetCDF files for a given year and report what's found
    """
    print(f"\n=== Scanning for available files in {year} ===")
    
    pattern = f'{input_dir}1x1km.GeoNO2.{year}*.MonMean.nc'
    files = glob.glob(pattern)
    files.sort()
    
    if not files:
        print(f"✗ No NetCDF files found matching pattern: {pattern}")
        return []
    
    print(f"Found {len(files)} NetCDF files:")
    available_months = []
    
    for file in files:
        filename = os.path.basename(file)
        try:
            # Extract month from filename (assuming format: 1x1km.GeoNO2.YYYYMM.MonMean.nc)
            year_month = filename.split('.')[2]
            month = int(year_month[-2:])
            available_months.append(month)
            
            file_size = os.path.getsize(file) / (1024 * 1024)  # MB
            print(f"  {month:02d}: {filename} ({file_size:.1f} MB)")
            
        except (IndexError, ValueError):
            print(f"  ?: {filename} (couldn't parse month)")
    
    return available_months

def main():
    """
    Main function to extract filled_GeoNO2 from NetCDF files
    """
    parser = argparse.ArgumentParser(description='Extract filled_GeoNO2 from NetCDF files to numpy arrays')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=int, choices=range(1, 13), 
                       help='Process only specific month (1-12)')
    
    args = parser.parse_args()
    year = args.year
    
    # Set default directories
    input_dir = f'/my-projects2/1.project/GeoNO2-v5/{year}/'
    
    output_dir = f'/my-projects2/1.project/NO2_DL_global/input_variables/GeoNO2-v5_input/{year}/'
    gchp_outdir = f'/my-projects2/1.project/NO2_DL_global/input_variables/GCHP-v2_input/{year}/'
    
    print(f"filled_GeoNO2 extraction script")
    print(f"Year: {year}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Scan for available files
    available_months = find_available_files(input_dir, year)
    
    if not available_months:
        print(f"✗ No NetCDF files found for year {year}")
        sys.exit(1)
    
    try:
        start_time = datetime.now()
        
        if args.month:
            if args.month not in available_months:
                print(f"✗ Month {args.month:02d} not available. Available months: {[f'{m:02d}' for m in available_months]}")
                sys.exit(1)
            success = extract_single_month_filled_geno2(year, args.month, input_dir, output_dir, gchp_outdir)
            exit_code = 0 if success else 1
        else:
            successful_months = extract_year_filled_geno2(year, input_dir, output_dir, gchp_outdir)
            exit_code = 0 if successful_months > 0 else 1
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nTotal processing time: {duration}")
        print(f"Extraction completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print(f"\n⚠ Extraction interrupted")
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()