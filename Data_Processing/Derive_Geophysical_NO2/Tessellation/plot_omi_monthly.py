import os
import xarray as xr
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_omi(ds, title, out_png):
    """Plot OMI-MINDS NO2 data"""
    # Load grid arrays
    try:
        x = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
        y = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    except FileNotFoundError:
        print("[ERROR] Grid coordinate files not found")
        return False

    fig, axes = plt.subplots(2, 1, figsize=(8, 8),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    vars_no2 = [v for v in ds.data_vars if 'NO2' in v]
    
    if len(vars_no2) == 0:
        print("[ERROR] No NO2 variables found in dataset")
        return False
    
    for ax, var in zip(axes, vars_no2):
        v = ds[var].values
        if v.size == 0 or np.all(np.isnan(v)):
            print(f"[WARN] variable {var} has no valid data; skipping")
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
                             vmin=0, vmax=5e+16)
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.05, fraction=0.05)
        cbar.set_label(var)
        ax.set_title(f"{title}: {var}", pad=10)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✓ Saved plot to {out_png}")
    return True

def main():
    """Plot existing monthly or yearly OMI-MINDS NO2 data"""
    parser = argparse.ArgumentParser(description='Plot OMI-MINDS NO2 monthly or yearly averages')
    parser.add_argument('year', type=int, help='Year to plot (e.g., 2009)')
    parser.add_argument('--month', type=int, metavar='MONTH', choices=range(1, 13),
                       help='Month to plot (1-12). If not specified, plots yearly average.')
    args = parser.parse_args()
    
    year = args.year
    CloudFraction_max, sza_max, QAlim = 0.1, 75, 0
    qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
        int(CloudFraction_max * 100),
        sza_max,
        int(QAlim * 100)
    )
    
    base_dir = '/my-projects2/1.project/NO2_col/OMI/'
    
    if args.month:
        # Plot monthly data
        monthly_dir = os.path.join(base_dir, f"{year}/monthly")
        fname = f"OMI-MINDS_Regrid_{year}{args.month:02d}_Monthly_{qcstr}.nc"
        fpath = os.path.join(monthly_dir, fname)
        out_png = os.path.join(monthly_dir, f"OMI-MINDS_Regrid_{year}{args.month:02d}_plot.png")
        title = f"{year}-{args.month:02d}"
        
        if not os.path.exists(fpath):
            print(f"[ERROR] Monthly file not found: {fpath}")
            return 1
            
        print(f"Loading monthly data from {fpath}")
        ds = xr.open_dataset(fpath)
        
    else:
        # Plot yearly data
        yearly_dir = os.path.join(base_dir, f"{year}/yearly")
        fname = f"OMI-MINDS_Regrid_{year}_{qcstr}.nc"
        fpath = os.path.join(yearly_dir, fname)
        out_png = os.path.join(yearly_dir, f"OMI-MINDS_Regrid_{year}_plot.png")
        title = f"{year} Yearly"
        
        if not os.path.exists(fpath):
            print(f"[ERROR] Yearly file not found: {fpath}")
            return 1
            
        print(f"Loading yearly data from {fpath}")
        ds = xr.open_dataset(fpath)
    
    # Create plot
    success = plot_omi(ds, title, out_png)
    
    if success:
        print(f"✓ Successfully created plot for {title}")
        return 0
    else:
        print(f"✗ Failed to create plot for {title}")
        return 1

if __name__ == "__main__":
    exit(main())
