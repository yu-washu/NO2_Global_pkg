import os
import xarray as xr
import argparse
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import netCDF4 as nc
from typing import Tuple, Optional, Union
import gc

class TropomiProcessor:
    
    def __init__(self, coord_dir: str = '/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables'):
        self.coord_dir = coord_dir
        self.coord_cache = {} 
        
    def load_coordinates(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if 'x' in self.coord_cache and 'y' in self.coord_cache:
            return self.coord_cache['x'], self.coord_cache['y']
            
        try:
            x_path = os.path.join(self.coord_dir, 'tSATLON_global_MAP.npy')
            y_path = os.path.join(self.coord_dir, 'tSATLAT_global_MAP.npy')
            
            x = np.load(x_path)
            y = np.load(y_path)
            
            self.coord_cache['x'] = x
            self.coord_cache['y'] = y
            
            print(f"✓ Loaded coordinate arrays: x{x.shape}, y{y.shape}")
            return x, y
            
        except FileNotFoundError:
            print("[WARN] Grid coordinate files not found")
            return None, None
    
    def validate_netcdf_file(self, filepath: str) -> bool:
        try:
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                print(f"[ERROR] File is empty: {filepath}")
                return False
            
            print(f"File size: {file_size / (1024*1024):.2f} MB")
            
            with nc.Dataset(filepath, 'r') as ds:
                print(f"Dimensions: {list(ds.dimensions.keys())}")
                print(f"Variables: {list(ds.variables.keys())}")
                for var_name in list(ds.variables.keys())[:3]:
                    var = ds.variables[var_name]
                    if len(var.dimensions) > 0:
                        try:
                            if var.ndim == 1:
                                _ = var[:min(10, var.shape[0])]
                            elif var.ndim == 2:
                                _ = var[:min(10, var.shape[0]), :min(10, var.shape[1])]
                            print(f"  {var_name}: shape {var.shape}, read test passed")
                        except Exception as e:
                            print(f"  {var_name}: read test failed - {e}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] File validation failed: {str(e)}")
            return False
    
    def calculate_subsample_params(self, lat_size: int, lon_size: int, 
                                 max_elements: int = 2000000) -> Tuple[int, int]:
        total_elements = lat_size * lon_size
        
        if total_elements <= max_elements:
            return 1, 1 
        
        reduction_factor = np.sqrt(total_elements / max_elements)
        lat_step = max(1, int(reduction_factor))
        lon_step = max(1, int(reduction_factor))
        
        print(f"Subsampling: {lat_size}x{lon_size} -> {lat_size//lat_step}x{lon_size//lon_step}")
        return lat_step, lon_step
    
    def plot_tropomi_data(self, dataset: Union[nc.Dataset, xr.Dataset], 
                         title: str, out_png: str, 
                         variables: list = None) -> bool:
        
        if variables is None:
            variables = ['NO2_tot', 'NO2_tot_gcshape']
        
        x_full, y_full = self.load_coordinates()
        use_external_coords = x_full is not None and y_full is not None
        
        n_vars = min(len(variables), 2)
        fig, axes = plt.subplots(n_vars, 1, figsize=(15, 6*n_vars),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        
        if n_vars == 1:
            axes = [axes]
        
        success_count = 0
        try:        
            for i, var in enumerate(variables[:n_vars]):
                ax = axes[i]
                
                try:
                    success = self._plot_single_variable(
                        dataset, var, ax, title, 
                        x_full, y_full, use_external_coords
                    )
                    if success:
                        success_count += 1
                    else:
                        ax.set_visible(False)
                        
                except Exception as e:
                    print(f"[ERROR] Failed to plot {var}: {str(e)}")
                    ax.set_visible(False)
            
            if success_count > 0:
                plt.tight_layout()
                try:
                    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"✓ Saved plot to {out_png}")
                    return True
                except Exception as e:
                    print(f"[ERROR] Failed to save plot: {str(e)}")
                    return False
            else:
                print(f"[ERROR] No variables successfully plotted")
            return False
        
        finally:
            plt.close(fig)
            gc.collect()
    
    def _plot_single_variable(self, dataset, var_name: str, ax, title: str,
                            x_full, y_full, use_external_coords: bool) -> bool:
        
        if isinstance(dataset, nc.Dataset):
            if var_name not in dataset.variables:
                print(f"[WARN] Variable {var_name} not found in dataset")
                return False
            var_obj = dataset.variables[var_name]
        else:
            if var_name not in dataset.data_vars:
                print(f"[WARN] Variable {var_name} not found in dataset")
                return False
            var_obj = dataset[var_name]
        
        print(f"Processing variable: {var_name}")
        
        if isinstance(dataset, nc.Dataset):
            shape = var_obj.shape
            lat_step, lon_step = self.calculate_subsample_params(shape[0], shape[1])
            
            if lat_step > 1 or lon_step > 1:
                indices = (slice(0, shape[0], lat_step), slice(0, shape[1], lon_step))
                v = var_obj[indices]
                if use_external_coords:
                    x_plot = x_full[indices]
                    y_plot = y_full[indices]
                else:
                    x_plot, y_plot = self._get_dataset_coordinates(dataset, indices)
            else:
                v = var_obj[:]
                x_plot, y_plot = (x_full, y_full) if use_external_coords else self._get_dataset_coordinates(dataset)
        
        else:  # xarray
            v = var_obj.values
            x_plot, y_plot = (x_full, y_full) if use_external_coords else self._get_xarray_coordinates(dataset)
        
        v = self._preprocess_data(v)
        
        valid_count = np.sum(~np.isnan(v))
        if valid_count == 0:
            print(f"[WARN] Variable {var_name} has no valid data")
            return False
        
        self._setup_map_features(ax)
        
        vmax = self._calculate_color_scale(v)
        
        try:
            mesh = ax.pcolormesh(x_plot, y_plot, v,
                               transform=ccrs.PlateCarree(),
                               cmap='RdYlBu_r',
                               vmin=0, vmax=1e16,
                               shading='nearest')
            
            cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                               pad=0.08, fraction=0.06)
            cbar.set_label(f'{var_name} (molecules/cm²)', fontsize=10)
            ax.set_title(f"{title}: {var_name}", pad=15, fontsize=12)
            
            print(f"  ✓ Successfully plotted {var_name}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create pcolormesh for {var_name}: {str(e)}")
            return False
    
    def _get_dataset_coordinates(self, dataset, indices=None):
        try:
            lon_vals = dataset.variables['lon'][:]
            lat_vals = dataset.variables['lat'][:]
            
            if indices:
                lon_vals = lon_vals[indices[1]]
                lat_vals = lat_vals[indices[0]]
            
            return np.meshgrid(lon_vals, lat_vals)
        except KeyError:
            print("[ERROR] Cannot find lon/lat coordinates in dataset")
            return None, None
    
    def _get_xarray_coordinates(self, dataset):
        lon_coord = lat_coord = None
        for coord in dataset.coords:
            if coord.lower() in ['lon', 'longitude']:
                lon_coord = coord
            elif coord.lower() in ['lat', 'latitude']:
                lat_coord = coord
        
        if lon_coord and lat_coord:
            return np.meshgrid(dataset[lon_coord].values, dataset[lat_coord].values)
        return None, None
    
    def _preprocess_data(self, v):
        if hasattr(v, 'filled'):
            v = v.filled(np.nan)
        elif hasattr(v, 'mask'):
            v = np.ma.filled(v, np.nan)
        return v
    
    def _setup_map_features(self, ax):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    def _calculate_color_scale(self, v):
        valid_data = v[~np.isnan(v)]
        try:
            p95 = np.nanpercentile(valid_data, 95)
            return min(p95, 1e16)  # 设置上限
        except:
            return 1e16


def process_file(file_path: str, output_path: str, title: str, 
                processor: TropomiProcessor, use_xarray: bool = False) -> bool:
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False
    
    print(f"Processing: {file_path}")
    print(f"Output: {output_path}")
    
    try:
        if use_xarray:
            print("Opening with xarray...")
            with xr.open_dataset(file_path, chunks={'lat': 1000, 'lon': 2000}) as ds:
                return processor.plot_tropomi_data(ds, title, output_path)
        else:
            print("Opening with netCDF4...")
            with nc.Dataset(file_path, 'r') as ds:
                return processor.plot_tropomi_data(ds, title, output_path)
                
    except Exception as e:
        print(f"[ERROR] Processing failed with {e}")
        if not use_xarray:
            print("Trying xarray fallback...")
            try:
                with xr.open_dataset(file_path, chunks={'lat': 1000, 'lon': 2000}) as ds:
                    return processor.plot_tropomi_data(ds, title, output_path)
            except Exception as e2:
                print(f"[ERROR] Fallback also failed: {e2}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Process geophysical NO2 data')
    parser.add_argument('-year', type=int, required=True, help='Year to plot')
    parser.add_argument('-mon', type=int, required=False, help='Month to plot')
    parser.add_argument('--yearly-only', action='store_true', 
                       help='Only create yearly average')
    parser.add_argument('--use-xarray', action='store_true', 
                       help='Use xarray instead of netCDF4')
    
    args = parser.parse_args()
    
    CloudFraction_max, sza_max, QAlim = 0.1, 75, 0.75
    qcstr = f'CF{int(CloudFraction_max * 100):03d}-SZA{sza_max}-QA{int(QAlim * 100)}'
    
    base_dir = '/my-projects2/1.project/NO2_col/TROPOMI/'
    # base_dir = f'/my-projects2/1.project/GeoNO2/{args.year}/'
    
    processor = TropomiProcessor()
    
    
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.yearly_only:
            print(f"Starting processing for {args.year}")
            yearly_dir = os.path.join(base_dir, "yearly")
            file_path = os.path.join(yearly_dir, f"Tropomi_Regrid_{args.year}_{qcstr}.nc")
            output_path = os.path.join(yearly_dir, f"Tropomi_Regrid_{args.year}_plot.png")
            title = str(args.year)
        else:
            print(f"Starting processing for {args.year}-{args.mon:02d}")
            monthly_dir = os.path.join(base_dir, "monthly")
            file_path = os.path.join(monthly_dir, f"Tropomi_Regrid_{args.year}{args.mon:02d}_Monthly_{qcstr}.nc")
            output_path = os.path.join(monthly_dir, f"Tropomi_Regrid_{args.year}{args.mon:02d}_plot.png")
            title = f"{args.year}-{args.mon:02d}"
        
        success = process_file(file_path, output_path, title, processor, args.use_xarray)
        
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