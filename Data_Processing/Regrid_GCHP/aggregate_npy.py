#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import psutil

def print_status(stage):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[{stage}] - Memory: {mem:.2f} GB")

# must match the variables you saved daily
VARS = [
    'gchp_NH3','gchp_O3','gchp_OH','gchp_NO',
    'gchp_NO3','gchp_N2O5','gchp_HO2','gchp_H2O2','gchp_CO'
]

# where your daily .npy files live
OUT_CNN_ROOT = '/my-projects2/1.project/NO2_DL_global/input_variables/GCHP_input'

def aggregate_and_cleanup(year: int, mon: int):
    basedir = os.path.join(OUT_CNN_ROOT, str(year))
    if not os.path.isdir(basedir):
        raise FileNotFoundError(f"No directory for year {year}: {basedir}")

    print_status("Start")

    # Define global coordinate arrays
    lon_km = np.round(np.linspace(-179.995, 179.995, 36000), 5)
    lat_km_full = np.round(np.linspace(-89.995, 89.995, 18000), 5)

    # Define trimmed latitude range
    lat_mask = (lat_km_full >= -59.995) & (lat_km_full <= 69.995)
    lat_indices = np.where(lat_mask)[0]

    for var in VARS:
        pattern = os.path.join(
            basedir,
            f"{var}_001x001_Global_map_{year}{mon:02d}??.npy"
        )
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[WARN] No daily files found for {var} {year}-{mon:02d}")
            continue

        arrs = []
        for fn in files:
            arr = np.load(fn)
            # Crop to desired lat range
            arr_cropped = arr[lat_indices, :]
            arrs.append(arr_cropped)

        monthly = np.mean(arrs, axis=0).astype(np.float32)

        # Save monthly file
        outname = os.path.join(
            basedir,
            f"{var}_001x001_Global_map_{year}{mon:02d}.npy"
        )
        np.save(outname, monthly)
        print(f"WROTE {outname}")

        for fn in files:
            os.remove(fn)
        print(f"REMOVED {len(files)} daily files for {var}")
        print_status ("finish average for one var")
    
    print_status("finishe avergae for all vars")


def plot_multiple_geophy(YEAR, MONTH, vars_to_plot):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # base directory for CNN input .npy files
    base_dir = os.path.join(OUT_CNN_ROOT, str(YEAR))

    lon = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLON_global.npy')
    lat = np.load('/my-projects2/1.project/NO2_DL_global/input_variables/tSATLAT_global.npy')


    n = len(vars_to_plot)
    # grid layout up to 9 plots
    if n <= 3:
        nrows, ncols = 1, n
    else:
        nrows, ncols = min(3, n), min(3, n)

    fig = plt.figure(figsize=(ncols*4, nrows*3))
    cmap = plt.cm.get_cmap('RdYlBu_r')

    for i, var in enumerate(vars_to_plot):
        ax = fig.add_subplot(nrows, ncols, i+1, projection=ccrs.PlateCarree())
        fn = f"{var}_001x001_Global_map_{YEAR}{MONTH}.npy"
        arr = np.load(os.path.join(base_dir, fn))
        mesh = ax.pcolormesh(lon, lat, arr,
                             transform=ccrs.PlateCarree(), cmap=cmap,
                             vmin=0, vmax=np.nanmax(arr)*0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.set_extent([-180,180,-90,90], ccrs.PlateCarree())
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
        cbar.set_label(var)
        ax.set_title(var, fontsize=10)

    plt.suptitle(f'Geophysical Fields {YEAR}{MONTH}', fontsize=14, y=0.98)
    plt.tight_layout()
    out_png = os.path.join(base_dir, f'Geophy_{YEAR}{MONTH}.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print_status("figure saving")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Aggregate daily .npy maps into monthly means and delete daily files."
    )
    p.add_argument('--year', type=int, required=True, help="e.g. 2018")
    p.add_argument('--mon',  type=int, required=True, help="01–12")
    args = p.parse_args()

    aggregate_and_cleanup(args.year, args.mon)
    # plot_multiple_geophy(args.year, f"{args.mon:02d}",
    #                  vars_to_plot = ['gchp_NH3','gchp_O3','gchp_OH'])