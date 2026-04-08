#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import argparse
import glob
from pathlib import Path

def print_status(message):
    """Print status message with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def find_january_files(base_dir, data_type='GeoNO2'):
    """
    Find all January numpy files for the specified data type
    
    Args:
        base_dir (str): Base directory containing year subdirectories
        data_type (str): Either 'GeoNO2' or 'gchp_NO2'
    
    Returns:
        dict: Dictionary mapping year to file path and directory
    """
    january_files = {}
    
    # Define subdirectories based on data type
    if data_type == 'GeoNO2':
        subdir = "GeoNO2_input"
        file_prefix = "GeoNO2_001x001_Global_map_"
    elif data_type == 'gchp_NO2':
        subdir = "GCHP_input"
        file_prefix = "gchp_NO2_001x001_Global_map_"
    else:
        raise ValueError("data_type must be either 'GeoNO2' or 'gchp_NO2'")
    
    # Search for year directories
    for year_dir in glob.glob(os.path.join(base_dir, subdir, "*")):
        if os.path.isdir(year_dir):
            year_name = os.path.basename(year_dir)
            try:
                year = int(year_name)
                # Look for January file (month 01)
                january_file = os.path.join(year_dir, f"{file_prefix}{year}01.npy")
                if os.path.exists(january_file):
                    january_files[year] = {
                        'file_path': january_file,
                        'year_dir': year_dir
                    }
            except ValueError:
                continue
    
    return january_files

def plot_january_comparison(years_to_plot, data_type='GeoNO2', base_dir=None, output_dir=None):
    """
    Plot January data comparison in the style of plot_multiple_geophy
    
    Args:
        years_to_plot (list): List of years to plot
        data_type (str): Type of data ('GeoNO2' or 'gchp_NO2')
        base_dir (str): Base directory for data
        output_dir (str): Directory to save plots
    """
    # Load coordinate arrays (same as in plot_multiple_geophy)
    lon = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global.npy')
    lat = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global.npy')
    
    print_status(f"Loaded coordinate arrays: lon {lon.shape}, lat {lat.shape}")
    
    # Find available January files
    january_files = find_january_files(base_dir, data_type)
    
    # Filter for requested years
    available_years = []
    file_paths = []
    for year in sorted(years_to_plot):
        if year in january_files:
            available_years.append(year)
            file_paths.append(january_files[year]['file_path'])
        else:
            print(f"Warning: January {data_type} data not found for year {year}")
    
    if not available_years:
        print(f"No January {data_type} data found for requested years")
        return
    
    print_status(f"Found January {data_type} data for years: {available_years}")
    
    # Load all data to determine global color scale
    all_data = []
    data_arrays = {}
    
    for year, file_path in zip(available_years, file_paths):
        try:
            arr = np.load(file_path)
            data_arrays[year] = arr
            valid_data = arr[~np.isnan(arr)]
            if len(valid_data) > 0:
                all_data.extend(valid_data)
            print_status(f"Loaded {year}: shape {arr.shape}, valid pixels: {len(valid_data):,}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not data_arrays:
        print("No valid data loaded!")
        return
    
    # Calculate global vmin and vmax (similar to plot_multiple_geophy approach)
    if all_data:
        global_max = np.nanmax(all_data)
        global_min = 0  # Following the vmin=0 pattern from plot_multiple_geophy
        global_vmax = global_max * 0.5  # Following the vmax=np.nanmax(arr)*0.5 pattern
        print_status(f"Global color scale: vmin={global_min}, vmax={global_vmax:.2e}")
    
    # Grid layout (same logic as plot_multiple_geophy)
    n = len(available_years)
    if n <= 3:
        nrows, ncols = 1, n
    elif n <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3
    
    # Create figure (same style as plot_multiple_geophy)
    fig = plt.figure(figsize=(ncols*4, nrows*3))
    cmap = plt.get_cmap('RdYlBu_r')  # Same colormap as plot_multiple_geophy
    
    for i, year in enumerate(available_years):
        if i >= 9:  # Limit to 9 plots maximum
            break
            
        ax = fig.add_subplot(nrows, ncols, i+1, projection=ccrs.PlateCarree())
        arr = data_arrays[year]
        
        # Plot using pcolormesh (same as plot_multiple_geophy)
        mesh = ax.pcolormesh(lon, lat, arr,
                           transform=ccrs.PlateCarree(), cmap=cmap,
                           vmin=0, vmax=20)
        
        # Add coastlines (same as plot_multiple_geophy)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # Set extent (same as plot_multiple_geophy)
        ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
        
        # Add colorbar (same style as plot_multiple_geophy)
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
        cbar.set_label(f'{data_type} {year}')
        
        # Set title (same style as plot_multiple_geophy)
        ax.set_title(f'{data_type} January {year}', fontsize=10)
    
    # Main title (same style as plot_multiple_geophy)
    plt.suptitle(f'{data_type} January Comparison', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure (same approach as plot_multiple_geophy)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_png = os.path.join(output_dir, f'{data_type}_January_comparison.png')
    else:
        out_png = f'{data_type}_January_comparison.png'
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print_status(f"Figure saved: {out_png}")
    
    plt.show()
    plt.close()

def plot_single_january(year, month=1, data_type='GeoNO2', base_dir=None, output_dir=None):
    """
    Plot single January data (similar to individual call of plot_multiple_geophy)
    
    Args:
        year (int): Year to plot
        month (int): Month (default 1 for January)
        data_type (str): Type of data ('GeoNO2' or 'gchp_NO2')
        base_dir (str): Base directory for data
        output_dir (str): Directory to save plot
    """
    # Load coordinate arrays
    lon = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global.npy')
    lat = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global.npy')
    
    # Determine file path
    if data_type == 'GeoNO2':
        subdir = "GeoNO2_input"
        file_prefix = "GeoNO2_001x001_Global_map_"
    elif data_type == 'gchp_NO2':
        subdir = "GCHP_input"
        file_prefix = "gchp_NO2_001x001_Global_map_"
    else:
        raise ValueError("data_type must be either 'GeoNO2' or 'gchp_NO2'")
    
    file_path = os.path.join(base_dir, subdir, str(year), f"{file_prefix}{year}{month:02d}.npy")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Load data
    try:
        arr = np.load(file_path)
        print_status(f"Loaded {year}-{month:02d}: shape {arr.shape}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return
    
    # Create figure (same style as plot_multiple_geophy)
    fig = plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('RdYlBu_r')
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Plot using pcolormesh (same as plot_multiple_geophy)
    mesh = ax.pcolormesh(lon, lat, arr,
                       transform=ccrs.PlateCarree(), cmap=cmap,
                       vmin=0, vmax=20)
    
    # Add coastlines
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Set extent
    ax.set_extent([-180, 180, -90, 90], ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
    cbar.set_label(data_type)
    
    # Set title
    ax.set_title(f'{data_type} {year}-{month:02d}', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_png = os.path.join(output_dir, f'{data_type}_{year}{month:02d}.png')
    else:
        out_png = f'{data_type}_{year}{month:02d}.png'
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print_status(f"Figure saved: {out_png}")
    
    plt.show()
    plt.close()

def main():
    """
    Main function to plot January NO2 data in plot_multiple_geophy style
    """
    parser = argparse.ArgumentParser(description='Plot January NO2 data in plot_multiple_geophy style')
    parser.add_argument('--base_dir', type=str, 
                       default='/my-projects2/1.project/NO2_DL_global/input_variables/',
                       help='Base directory containing input data')
    parser.add_argument('--data_type', choices=['GeoNO2', 'gchp_NO2', 'both'], 
                       default='both',
                       help='Type of data to plot')
    parser.add_argument('--output_dir', type=str,
                       default='/my-projects2/1.project/NO2_DL_global/input_variables/GeoNO2_input/',
                       help='Directory to save plots (optional)')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Specific years to plot (required)')
    parser.add_argument('--single', action='store_true',
                       help='Create individual plots for each year instead of comparison')
    
    args = parser.parse_args()
    
    if not args.years:
        print("Error: Please specify years to plot using --years")
        print("Example: python script.py --years 2019 2020 2021")
        return
    
    print_status("January NO2 Data Plotter (plot_multiple_geophy style)")
    print(f"Base directory: {args.base_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Years: {args.years}")
    print(f"Output directory: {args.output_dir}")
    
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        return
    
    data_types = ['GeoNO2', 'gchp_NO2'] if args.data_type == 'both' else [args.data_type]
    
    for data_type in data_types:
        print_status(f"Processing {data_type} data")
        
        if args.single:
            # Create individual plots for each year
            for year in args.years:
                plot_single_january(year, month=1, data_type=data_type, 
                                   base_dir=args.base_dir, output_dir=args.output_dir)
        else:
            # Create comparison plot
            plot_january_comparison(args.years, data_type=data_type, 
                                  base_dir=args.base_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()