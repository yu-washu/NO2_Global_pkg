import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import os
import sys
from datetime import datetime

def process_file(file_path, output_path, title, use_xarray=True):
    """
    Process and plot NetCDF file with all variables except lat, lon, time
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return False
        
        print(f"Loading file: {file_path}")
        
        # Load the NetCDF file
        if use_xarray:
            ds = xr.open_dataset(file_path)
        else:
            import netCDF4 as nc
            ds = nc.Dataset(file_path, 'r')
            print("Warning: netCDF4 mode not fully implemented, using xarray instead")
            ds.close()
            ds = xr.open_dataset(file_path)
        
        # Get all variables except lat, lon, time
        exclude_vars = ['lat', 'lon', 'time','gchp_HNO3', 'gchp_PAN', 'gchp_alkylnitrates', 'gchp_NO2col', 'omi_NO2_tot', 'tropomi_NO2_tot', 'omi_NO2_tot_gcshape', 'tropomi_NO2_tot_gcshape']
        plot_vars = [var for var in ds.data_vars if var not in exclude_vars]
        
        print(f"Variables to plot: {plot_vars}")
        print(f"Total variables: {len(plot_vars)}")
        
        if len(plot_vars) == 0:
            print("No variables to plot!")
            return False
        
        # Set up the figure and subplots
        n_vars = len(plot_vars)
        n_cols = 3  # Number of columns
        n_rows = int(np.ceil(n_vars / n_cols))  # Calculate rows needed
        
        # Create figure with cartopy projection
        fig = plt.figure(figsize=(20, 6 * n_rows))
        
        # Get coordinate bounds for consistent plotting
        lat = ds.lat.values
        lon = ds.lon.values
        lat_min, lat_max = lat.min(), lat.max()
        lon_min, lon_max = lon.min(), lon.max()
        
        print(f"Data extent: lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]")
        
        # Create subplots for each variable
        for i, var_name in enumerate(plot_vars):
            print(f"Processing variable {i+1}/{n_vars}: {var_name}")
            
            # Create subplot with cartopy projection
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection=ccrs.PlateCarree())
            
            # Get the data
            data = ds[var_name].values
            
            # Calculate vmax as 0.8 * max(data), handling NaN values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                vmax = 0.8 * np.max(valid_data)
                vmin = np.min(valid_data)
                # Ensure vmin is not negative for log scale if needed
                if vmin <= 0:
                    vmin = np.min(valid_data[valid_data > 0]) if np.any(valid_data > 0) else 1e-10
            else:
                vmax = 1.0
                vmin = 1e-10
                print(f"Warning: No valid data for {var_name}")
            
            # Create the map plot
            im = ax.pcolormesh(lon, lat, data, 
                               transform=ccrs.PlateCarree(),
                               cmap='RdYlBu_r',
                               vmin=vmin,
                               vmax=20,
                               shading='nearest')
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)
            ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
            cbar.set_label(f'{var_name}', fontsize=10)
            
            # Set title
            ax.set_title(f'{var_name}\n(vmax = {vmax:.2e})', fontsize=12, pad=20)
        
        # Adjust layout
        plt.tight_layout()
        plt.suptitle(f'NO2 Variables - {title}', fontsize=16, y=0.98)
        
        # Save the figure
        print(f"Saving plot to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print some statistics
        print("\nVariable Statistics:")
        print("-" * 70)
        for var_name in plot_vars:
            data = ds[var_name].values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(f"{var_name:25} | Min: {np.min(valid_data):.2e} | Max: {np.max(valid_data):.2e} | vmax (0.8*max): {0.8*np.max(valid_data):.2e}")
            else:
                print(f"{var_name:25} | No valid data")
        
        # Close the dataset
        ds.close()
        return True
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Process geophysical NO2 data')
    parser.add_argument('-year', type=int, required=True, help='Year to plot')
    parser.add_argument('-mon', type=int, required=False, help='Month to plot')
    parser.add_argument('--use-xarray', action='store_true', default=True,
                       help='Use xarray instead of netCDF4 (default: True)')
    
    args = parser.parse_args()
    
    base_dir = f'/my-projects2/1.project/GeoNO2/{args.year}/'
    
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.mon is None:
            print("Error: Month (-mon) is required")
            sys.exit(1)
        
        print(f"Starting processing for {args.year}-{args.mon:02d}")
        file_path = os.path.join(base_dir, f"1x1km.GeoNO2.{args.year}{args.mon:02d}.MonMean.nc")
        output_path = os.path.join(base_dir, f"GeoNO2_{args.year}{args.mon:02d}_MonMean.png")
        title = f"{args.year}-{args.mon:02d}"
        
        success = process_file(file_path, output_path, title, args.use_xarray)
        
        if success:
            print(f"✓ Processing completed successfully")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"✗ Processing failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Processing interrupted")
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()