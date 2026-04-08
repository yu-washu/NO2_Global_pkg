#!/usr/bin/env python3
"""
Create publishable seasonal NO2 plots
- One figure per year per data type (GeoNO2 or gchp_NO2)
- 4 rows x 3 columns = 12 months
- Seasonal order: DJF, MAM, JJA, SON
- Years: 2005-2023
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as tick
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import argparse
import gc
from datetime import datetime
from pathlib import Path

# Memory optimization: Limit matplotlib cache
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0

def print_status(message):
    """Print status message with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

def load_monthly_data(year, data_type='GeoNO2', base_dir=None):
    """
    Load all 12 months of data for a given year - MEMORY OPTIMIZED
    Returns file paths instead of loading all data at once
    
    Args:
        year (int): Year to load
        data_type (str): Either 'GeoNO2' or 'gchp_NO2'
        base_dir (str): Base directory containing year subdirectories
    
    Returns:
        dict: Dictionary mapping month (1-12) to file path (not loaded data)
    """
    if data_type == 'GeoNO2':
        subdir = "GeoNO2_input"
        file_prefix = "GeoNO2_001x001_Global_map_"
    elif data_type == 'gchp_NO2':
        subdir = "GCHP_input"
        file_prefix = "gchp_NO2_001x001_Global_map_"
    else:
        raise ValueError("data_type must be either 'GeoNO2' or 'gchp_NO2'")
    
    year_dir = os.path.join(base_dir, subdir, str(year))
    
    if not os.path.exists(year_dir):
        print_status(f"WARNING: Year directory not found: {year_dir}")
        return {}
    
    monthly_files = {}
    
    for month in range(1, 13):
        file_path = os.path.join(year_dir, f"{file_prefix}{year}{month:02d}.npy")
        
        if os.path.exists(file_path):
            monthly_files[month] = file_path
            print_status(f"  Found {year}-{month:02d}")
        else:
            print_status(f"  WARNING: File not found: {file_path}")
    
    return monthly_files

def create_ocean_mask(lon_1d, lat_1d, base_dir):
    """
    Create or load cached ocean mask using Natural Earth land polygons
    Returns: 2D boolean array (True = ocean, False = land)
    """
    # Cache file path
    mask_file = os.path.join(base_dir, 'ocean_mask_13000x36000.npy')
    
    # Try to load cached mask first
    if os.path.exists(mask_file):
        print_status(f"  Loading cached ocean mask from {mask_file}...")
        ocean_mask = np.load(mask_file)
        print_status(f"  ✓ Cached mask loaded: {ocean_mask.sum():,} ocean pixels")
        return ocean_mask
    
    # Create mask if not cached
    from cartopy.io import shapereader
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely.prepared import prep
    
    print_status("  Creating ocean mask (first-time, will be cached)...")
    
    # Load land polygons from Natural Earth (110m for speed)
    land_shp = shapereader.natural_earth(resolution='110m', 
                                         category='physical', 
                                         name='land')
    land_geoms = list(shapereader.Reader(land_shp).geometries())
    # Combine all land polygons into one prepared geometry for fast queries
    land = prep(unary_union(land_geoms))
    
    # Create 2D mesh
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    
    # Create mask (subsample for speed - check every Nth pixel)
    ny, nx = lon_2d.shape
    ocean_mask = np.ones((ny, nx), dtype=bool)
    
    # Subsample rate: check every 20th pixel, then interpolate
    step = 20
    print_status(f"    Checking {(ny//step) * (nx//step):,} points...")
    
    for i in range(0, ny, step):
        for j in range(0, nx, step):
            point = Point(lon_2d[i, j], lat_2d[i, j])
            if land.contains(point):
                # Mark surrounding area as land
                ocean_mask[max(0, i-step):min(ny, i+step), 
                          max(0, j-step):min(nx, j+step)] = False
    
    # Save mask for future use
    try:
        np.save(mask_file, ocean_mask)
        print_status(f"  ✓ Ocean mask saved to {mask_file}")
    except Exception as e:
        print_status(f"  WARNING: Could not save mask: {e}")
    
    print_status(f"  ✓ Ocean mask created: {ocean_mask.sum():,} ocean pixels, {(~ocean_mask).sum():,} land pixels")
    return ocean_mask

def plot_seasonal_year(year, data_type='GeoNO2', base_dir=None, output_dir=None):
    """
    Create publishable seasonal plot for one year
    
    Args:
        year (int): Year to plot
        data_type (str): Type of data ('GeoNO2' or 'gchp_NO2')
        base_dir (str): Base directory for data
        output_dir (str): Directory to save plots
    """
    start_time = datetime.now()
    print_status(f"Creating seasonal plot for {year} - {data_type}")
    
    # Load coordinate arrays
    try:
        lon = np.load(os.path.join(base_dir, 'tSATLON_global.npy'))
        lat = np.load(os.path.join(base_dir, 'tSATLAT_global.npy'))
        print_status(f"  Loaded coordinates: lon {lon.shape}, lat {lat.shape}")
    except Exception as e:
        print_status(f"ERROR: Could not load coordinate arrays: {e}")
        return
    
    # Create or load ocean mask ONCE for this figure
    try:
        ocean_mask = create_ocean_mask(lon, lat, base_dir)
    except Exception as e:
        print_status(f"  WARNING: Could not create ocean mask: {e}")
        print_status(f"  Continuing without ocean masking...")
        ocean_mask = None  # Will skip masking if it fails
    
    # Get file paths for monthly data (don't load yet - memory optimization)
    monthly_files = load_monthly_data(year, data_type, base_dir)
    
    if len(monthly_files) < 12:
        print_status(f"WARNING: Only {len(monthly_files)}/12 months available for {year}")
        if len(monthly_files) == 0:
            print_status(f"SKIPPING: No data available for {year}")
            return
    
    # Define seasonal order: DJF, MAM, JJA, SON (4 rows x 3 cols)
    # Row 1 (DJF): Dec, Jan, Feb
    # Row 2 (MAM): Mar, Apr, May
    # Row 3 (JJA): Jun, Jul, Aug
    # Row 4 (SON): Sep, Oct, Nov
    month_order = [12, 1, 2,  # DJF
                   3, 4, 5,   # MAM
                   6, 7, 8,   # JJA
                   9, 10, 11] # SON
    
    month_names = ['Dec', 'Jan', 'Feb',
                   'Mar', 'Apr', 'May',
                   'Jun', 'Jul', 'Aug',
                   'Sep', 'Oct', 'Nov']
    
    season_labels = ['DJF', 'MAM', 'JJA', 'SON']
    
    # Setup colormap (matching the provided style)
    cmap = plt.cm.RdYlBu_r
    cmap.set_bad(alpha=0)  # Make NaN values transparent (ocean will show white through)
    cmap.set_under(alpha=0)  # Also make values below vmin transparent
    norm = mcolors.Normalize(vmin=0, vmax=15)
    
    # Create figure with 4 rows x 3 columns - reduce height for tighter spacing
    fig = plt.figure(figsize=(15, 12))  # Further reduced to 12 for even tighter vertical spacing
    
    extent = [-180, 180, -60, 70]
    
    for idx, (month, month_name) in enumerate(zip(month_order, month_names)):
        row = idx // 3
        col = idx % 3
        
        # Create subplot with PlateCarree projection
        ax = fig.add_subplot(4, 3, idx + 1, projection=ccrs.PlateCarree())
        
        # Check if data exists for this month
        if month not in monthly_files:
            ax.text(0.5, 0.5, f'No data\n{month_name} {year}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            continue
        
        # MEMORY OPTIMIZATION: Load one month at a time, then release
        print_status(f"    Loading month {month:02d}...")
        NO2 = np.load(monthly_files[month])
        print_status(f"    Loaded shape: {NO2.shape}, mem: {NO2.nbytes / 1024**2:.1f} MB")
        
        # Apply ocean mask - set ocean values to NaN
        if ocean_mask is not None:
            NO2 = NO2.copy()  # Don't modify cached data
            NO2[ocean_mask] = np.nan
            print_status(f"    Applied ocean mask")
        
        # Set extent and aspect
        ax.set_aspect(1.25)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        # Set white background first (shows through NaN values)
        ax.set_facecolor('white')
        
        # Add minimal features - only add on FIRST subplot to avoid redundant downloads
        if idx == 0:  # Only first subplot triggers the download
            print_status(f"    Downloading cartopy features (one-time, cached)...")
            try:
                # These will download once and be cached
                ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.3, zorder=4)
                ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.3, zorder=5)
                print_status(f"    ✓ Features added")
            except Exception as e:
                print_status(f"    WARNING: Could not add features: {e}")
        else:
            # Reuse cached features for subsequent subplots
            try:
                ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.3, zorder=4)
                ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.3, zorder=5)
            except Exception as e:
                pass  # Silently skip if features fail
        
        # Plot data (zorder=2, between ocean and borders)
        # NaN values will be transparent, showing white ocean underneath
        pcm = ax.pcolormesh(lon, lat, NO2, transform=ccrs.PlateCarree(),
                           cmap=cmap, norm=norm, rasterized=True, zorder=2)  # rasterized=True saves memory
        
        # Add title for each subplot - minimal padding
        title = f'{month_name} {year}'
        ax.set_title(title, fontsize=11, fontweight='bold', pad=0)  # No padding for tightest spacing
        
        # Add season label on the left side of first column
        if col == 0:
            ax.text(-0.15, 0.5, season_labels[row], 
                   transform=ax.transAxes,
                   fontsize=14, fontweight='bold', 
                   rotation=90, va='center', ha='center')
        
        # MEMORY OPTIMIZATION: Release NO2 data immediately after plotting
        del NO2
        if idx % 3 == 2:  # Every 3 plots (end of row), force garbage collection
            gc.collect()
    
    # Add main title - closer to plots
    spec_name = 'GeoNO$_{2}$' if data_type == 'GeoNO2' else 'GCHP NO$_{2}$'
    fig.suptitle(f'{spec_name} Monthly Distribution - {year}', 
                fontsize=16, fontweight='bold', y=0.985)  # Slightly higher to avoid overlap with compressed plots
    
    # Adjust layout - extremely tight spacing
    plt.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.06, 
                       hspace=-0.30, wspace=0.08)  # Even more negative hspace for tighter vertical spacing
    
    # Add colorbar at the bottom - closer to plots
    cbar_ax = fig.add_axes([0.25, 0.03, 0.5, 0.012])  # [left, bottom, width, height]
    cbar = plt.colorbar(pcm, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(r'NO$_{2}$ (ppb)', size=12, fontweight='bold')
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_png = os.path.join(output_dir, f'{data_type}_{year}_seasonal.png')
    else:
        out_png = f'{data_type}_{year}_seasonal.png'
    
    print_status(f"  Saving figure to: {out_png}")
    # Reduce DPI to 200 (from 500) to save massive amounts of memory
    # 500 DPI creates ~20MB files, 200 DPI creates ~5MB files but still publication quality
    fig.savefig(out_png, dpi=200, bbox_inches='tight', transparent=False)
    file_size_mb = os.path.getsize(out_png) / 1024**2
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print_status(f"  ✓ Figure saved: {file_size_mb:.2f} MB (took {duration:.1f}s)")
    
    plt.close(fig)
    gc.collect()  # Force garbage collection to free memory

def main():
    """
    Main function to create publishable seasonal plots
    """
    parser = argparse.ArgumentParser(
        description='Create publishable seasonal NO2 plots (4x3 grid, DJF-MAM-JJA-SON order)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all years for both data types
  %(prog)s --years 2005 2006 2007 --data_type both
  
  # Plot specific year for GeoNO2 only
  %(prog)s --years 2019 --data_type GeoNO2
  
  # Plot all years from 2005-2023
  %(prog)s --years $(seq 2005 2023) --data_type both
        """
    )
    parser.add_argument('--base_dir', type=str, 
                       default='/my-projects2/1.project/NO2_DL_global/input_variables/',
                       help='Base directory containing input data')
    parser.add_argument('--data_type', choices=['GeoNO2', 'gchp_NO2', 'both'], 
                       default='both',
                       help='Type of data to plot')
    parser.add_argument('--output_dir', type=str,
                       default='/my-projects2/1.project/NO2_DL_global/input_variables/plots/',
                       help='Directory to save plots')
    parser.add_argument('--years', nargs='+', type=int, required=True,
                       help='Years to plot (e.g., 2005 2006 2007 or $(seq 2005 2023))')
    
    args = parser.parse_args()
    
    print_status("="*80)
    print_status("Publishable Seasonal NO2 Plotter")
    print_status("="*80)
    print(f"Base directory: {args.base_dir}")
    print(f"Data type: {args.data_type}")
    print(f"Years: {min(args.years)}-{max(args.years)} ({len(args.years)} years)")
    print(f"Output directory: {args.output_dir}")
    print_status("="*80)
    
    if not os.path.exists(args.base_dir):
        print_status(f"ERROR: Base directory not found: {args.base_dir}")
        return
    
    # Determine data types to process
    data_types = ['GeoNO2', 'gchp_NO2'] if args.data_type == 'both' else [args.data_type]
    
    total_plots = len(args.years) * len(data_types)
    current_plot = 0
    
    for data_type in data_types:
        print_status(f"\nProcessing {data_type} data")
        print_status("-"*80)
        
        for year in sorted(args.years):
            current_plot += 1
            print_status(f"\n[{current_plot}/{total_plots}] Year {year}")
            
            try:
                plot_seasonal_year(year, data_type=data_type, 
                                 base_dir=args.base_dir, 
                                 output_dir=args.output_dir)
                
                # MEMORY OPTIMIZATION: Clean up between data types
                gc.collect()
                print_status(f"  Memory cleanup completed")
                
            except Exception as e:
                print_status(f"ERROR processing {year} {data_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print_status("="*80)
    print_status(f"Completed! Generated {current_plot} seasonal plots")
    print_status(f"Output directory: {args.output_dir}")
    print_status("="*80)

if __name__ == "__main__":
    main()
