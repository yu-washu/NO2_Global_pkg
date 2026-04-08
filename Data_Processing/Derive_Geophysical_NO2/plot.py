import gc
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — much faster in batch jobs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import os


def _load_lonlat():
    """Load common global lon/lat grids."""
    x = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
    y = np.load('/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    return x, y


def _get_first_var(ds, candidates):
    for name in candidates:
        if name in ds:
            return name
    return None


def plot_no2_combo_month(year, month, geono2_dir, tropomi_monthly_dir, qcstr, out_png):
    """
    For a given month, make ONE figure with 3 panels:
      1) NO2_trop (TROPOMI)
      2) NO2_tot_gcshap (TROPOMI)
      3) Geo_NO2 (GeoNO2 gap field)
    """
    month_label = {1: "January", 7: "July"}.get(month, f"{month:02d}")

    # Build file paths
    trop_path = os.path.join(
        tropomi_monthly_dir,
        f"Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc",
    )
    geo_path = os.path.join(
        geono2_dir,
        f"1x1km.GeoNO2.{year}{month:02d}.MonMean.nc",
    )

    if not os.path.exists(trop_path):
        print(f"[WARN] TROPOMI file not found: {trop_path}", flush=True)
        return
    if not os.path.exists(geo_path):
        print(f"[WARN] GeoNO2 file not found: {geo_path}", flush=True)
        return

    print(f"[INFO] Loading TROPOMI file: {trop_path}", flush=True)
    print(f"[INFO] Loading GeoNO2 file: {geo_path}", flush=True)

    ds_trop = xr.open_dataset(trop_path)
    ds_geo = xr.open_dataset(geo_path)

    try:
        x, y = _load_lonlat()
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot", flush=True)
        ds_trop.close()
        ds_geo.close()
        return

    # Variable name fallbacks to be robust to different naming
    var_trop = _get_first_var(ds_trop, ["NO2_trop"])
    var_tot_gc = _get_first_var(ds_trop, ["NO2_tot_gcshape"])
    var_geo = _get_first_var(ds_geo, ["gap_GeoNO2"])

    if var_trop is None or var_tot_gc is None or var_geo is None:
        print(f"[WARN] Missing expected variables for {year}-{month:02d}:", flush=True)
        print(f"       NO2_trop candidates -> found: {var_trop}", flush=True)
        print(f"       NO2_tot_gcshap candidates -> found: {var_tot_gc}", flush=True)
        print(f"       Geo_NO2 candidates -> found: {var_geo}", flush=True)
        ds_trop.close()
        ds_geo.close()
        return

    # Define panels as (dataset, varname, label, vmax) — load data one at a time
    panel_defs = [
        (ds_trop, var_trop,  f"{var_trop} (TROPOMI)",  1e16),
        (ds_trop, var_tot_gc, f"{var_tot_gc} (TROPOMI)", 2e16),
        (ds_geo,  var_geo,   f"{var_geo} (GeoNO2)",    15.0),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 15),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    try:
        for ax, (ds_panel, varname, label, vmax_fixed) in zip(axes, panel_defs):
            print(f"[INFO] Plotting {label} ...", flush=True)
            data = ds_panel[varname].values

            if data.size == 0 or np.all(np.isnan(data)):
                print(f"[WARN] {label} has no valid data for {year}-{month:02d}", flush=True)
                del data
                gc.collect()
                ax.set_visible(False)
                continue

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

            mesh = ax.pcolormesh(
                x,
                y,
                data,
                transform=ccrs.PlateCarree(),
                cmap="RdYlBu_r",
                vmin=0,
                vmax=vmax_fixed,
                rasterized=True,  # rasterize the mesh — avoids rendering millions of vector polygons
            )
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                orientation="horizontal",
                pad=0.05,
                fraction=0.05,
            )
            cbar.set_label(label)
            ax.set_title(f"{year}-{month:02d} {label}", pad=10)

            # Free the array immediately after it has been handed to matplotlib
            del data
            gc.collect()

        plt.tight_layout()
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved {month_label} combo plot to: {out_png}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create combo plot for {year}-{month:02d}: {e}", flush=True)
    finally:
        ds_trop.close()
        ds_geo.close()
        del x, y
        gc.collect()

def plot_no2_filled_combo_month(year, month, geono2_dir, tropomi_monthly_dir, qcstr, out_png,
                                tropomi_filled_dir=None):
    """
    Two-panel figure: (1) NO2_trop_filled (TROPOMI), (2) filled_GeoNO2 (GeoNO2).
    """
    month_label = {1: "January", 7: "July"}.get(month, f"{month:02d}")
    if tropomi_filled_dir is None:
        tropomi_filled_dir = f"/my-projects2/1.project/NO2_col/TropNO2col/{year}"

    trop_filled_path = os.path.join(
        tropomi_filled_dir,
        f"1x1km.TROPOMI.TropNO2col_filled.{year}{month:02d}.MonMean.nc",
    )
    geo_filled_path = os.path.join(
        geono2_dir,
        f"1x1km.GeoNO2.{year}{month:02d}.MonMean.nc",
    )

    if not os.path.exists(trop_filled_path):
        print(f"[WARN] TROPOMI filled file not found: {trop_filled_path}", flush=True)
        return
    if not os.path.exists(geo_filled_path):
        print(f"[WARN] GeoNO2 file not found: {geo_filled_path}", flush=True)
        return

    print(f"[INFO] Loading TROPOMI filled: {trop_filled_path}", flush=True)
    print(f"[INFO] Loading GeoNO2: {geo_filled_path}", flush=True)

    ds_trop = xr.open_dataset(trop_filled_path)
    ds_geo = xr.open_dataset(geo_filled_path)

    try:
        x, y = _load_lonlat()
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot", flush=True)
        ds_trop.close()
        ds_geo.close()
        return

    var_trop = _get_first_var(ds_trop, ["NO2_trop_filled"])
    var_geo = _get_first_var(ds_geo, ["filled_GeoNO2"])

    if var_trop is None or var_geo is None:
        print(f"[WARN] Missing variables for {year}-{month:02d}: NO2_trop_filled={var_trop}, filled_GeoNO2={var_geo}", flush=True)
        ds_trop.close()
        ds_geo.close()
        return

    panel_defs = [
        (ds_trop, var_trop, f"{var_trop} (TROPOMI)", 1e16),
        (ds_geo, var_geo, f"{var_geo} (GeoNO2)", 15),
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    try:
        for ax, (ds_panel, varname, label, vmax_fixed) in zip(axes, panel_defs):
            print(f"[INFO] Plotting {label} ...", flush=True)
            data = ds_panel[varname].values

            if data.size == 0 or np.all(np.isnan(data)):
                print(f"[WARN] {label} has no valid data for {year}-{month:02d}", flush=True)
                del data
                gc.collect()
                ax.set_visible(False)
                continue

            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

            mesh = ax.pcolormesh(
                x,
                y,
                data,
                transform=ccrs.PlateCarree(),
                cmap="RdYlBu_r",
                vmin=0,
                vmax=vmax_fixed,
                rasterized=True,
            )
            cbar = plt.colorbar(
                mesh,
                ax=ax,
                orientation="horizontal",
                pad=0.05,
                fraction=0.05,
            )
            cbar.set_label(label)
            ax.set_title(f"{year}-{month:02d} {label}", pad=10)

            del data
            gc.collect()

        plt.tight_layout()
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved {month_label} filled combo plot to: {out_png}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create filled combo plot for {year}-{month:02d}: {e}", flush=True)
    finally:
        ds_trop.close()
        ds_geo.close()
        del x, y
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot NO2 combo figures for Jan and July. Use --combo for 3-panel "
            "(NO2_trop, NO2_tot_gcshap, gap_GeoNO2), --filled-combo for 2-panel "
            "(NO2_trop_filled, filled_GeoNO2). If neither is set, both are produced."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to plot (e.g., 2019).",
    )
    parser.add_argument(
        "--geono2-dir",
        type=str,
        default=None,
        help=(
            "Base directory for GeoNO2 files "
            "(default: /my-projects2/1.project/GeoNO2-v2/[year]/)."
        ),
    )
    parser.add_argument(
        "--tropomi-dir",
        type=str,
        default=None,
        help=(
            "Directory containing TROPOMI monthly files "
            "(default: /my-projects2/1.project/NO2_col/TROPOMI/[year]/monthly)."
        ),
    )
    parser.add_argument(
        "--qcstr",
        type=str,
        default=None,
        help="Quality-control string used in TROPOMI filenames (default: SZA80-QA75).",
    )
    parser.add_argument(
        "--combo",
        action="store_true",
        help="Plot 3-panel combo (NO2_trop, NO2_tot_gcshap, gap_GeoNO2) for Jan and Jul.",
    )
    parser.add_argument(
        "--filled-combo",
        action="store_true",
        help="Plot 2-panel filled combo (NO2_trop_filled, filled_GeoNO2) for Jan and Jul.",
    )

    args = parser.parse_args()

    year = args.year
    geono2_dir = args.geono2_dir or f"/my-projects2/1.project/GeoNO2-v2/{year}/"
    tropomi_monthly_dir = args.tropomi_dir or f"/my-projects2/1.project/NO2_col/TROPOMI/{year}/monthly"
    qcstr = args.qcstr or "SZA80-QA75"

    # If neither flag given, run both; otherwise run only the selected one(s)
    run_both = not args.combo and not args.filled_combo
    do_combo = args.combo or run_both
    do_filled_combo = args.filled_combo or run_both

    print(f"[INFO] GeoNO2 dir: {geono2_dir}", flush=True)
    print(f"[INFO] TROPOMI monthly dir: {tropomi_monthly_dir}", flush=True)
    print(f"[INFO] Using qcstr: {qcstr}", flush=True)
    print(f"[INFO] 3-panel combo: {do_combo}, 2-panel filled combo: {do_filled_combo}", flush=True)

    if do_combo:
        jan_png = os.path.join(geono2_dir, f"NO2_Jan_combo_{year}01.png")
        jul_png = os.path.join(geono2_dir, f"NO2_Jul_combo_{year}07.png")
        plot_no2_combo_month(year, 1, geono2_dir, tropomi_monthly_dir, qcstr, jan_png)
        plot_no2_combo_month(year, 7, geono2_dir, tropomi_monthly_dir, qcstr, jul_png)

    if do_filled_combo:
        tropomi_filled_dir = f"/my-projects2/1.project/NO2_col/TropNO2col/{year}"
        jan_filled_png = os.path.join(geono2_dir, f"NO2_Jan_filled_combo_{year}01.png")
        jul_filled_png = os.path.join(geono2_dir, f"NO2_Jul_filled_combo_{year}07.png")
        plot_no2_filled_combo_month(year, 1, geono2_dir, tropomi_monthly_dir, qcstr, jan_filled_png,
                                   tropomi_filled_dir=tropomi_filled_dir)
        plot_no2_filled_combo_month(year, 7, geono2_dir, tropomi_monthly_dir, qcstr, jul_filled_png,
                                   tropomi_filled_dir=tropomi_filled_dir)