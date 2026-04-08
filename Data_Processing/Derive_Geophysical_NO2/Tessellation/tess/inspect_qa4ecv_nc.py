#!/usr/bin/env python3
"""
Inspect QA4ECV OMI NO2 netCDF structure and compare to PSD v1.1.
Run from env that has netCDF4 (e.g. same as tess_OMI_KNMI_v2):
  python inspect_qa4ecv_nc.py <path_to_QA4ECV_L2_NO2_OMI_*.nc>
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        # Default: file from user (rvmartin path exists)
        path = "/storage1/fs1/rvmartin/Active/yany1/1.project/OMI_KNMI/2005/omi_no2_qa4ecv_20050119/QA4ECV_L2_NO2_OMI_20050119T100200_o02740_fitB_v1.nc"
        alt  = "/storage1/fs1/rvmartin2/Active/yany1/1.project/OMI_KNMI/2005/omi_no2_qa4ecv_20050119/QA4ECV_L2_NO2_OMI_20050119T100200_o02740_fitB_v1.nc"
        if os.path.isfile(path):
            pass
        elif os.path.isfile(alt):
            path = alt
        else:
            # Use first .nc in 20050119 dir if available
            for base in [path, alt]:
                d = os.path.dirname(base)
                if os.path.isdir(d):
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".nc"):
                            path = os.path.join(d, f)
                            break
                    break
            else:
                print("Usage: python inspect_qa4ecv_nc.py [path_to_QA4ECV_L2_NO2_OMI_*.nc]")
                sys.exit(1)
    else:
        path = sys.argv[1]

    if not os.path.isfile(path):
        print("File not found:", path)
        sys.exit(1)

    try:
        import netCDF4 as nc
    except ImportError:
        print("Need netCDF4: pip install netCDF4 (or use your tessellation env)")
        sys.exit(1)

    print("=" * 70)
    print("QA4ECV NO2 PSD v1.1 expects: time=1, scanline=UNLIMITED, ground_pixel=60")
    print("Pixel variables: (time, scanline, ground_pixel); corners: (..., corner=4)")
    print("=" * 70)
    print("File:", path)
    print()

    ds = nc.Dataset(path, "r")

    def dims_and_shape(var):
        if hasattr(var, "dimensions") and hasattr(var, "shape"):
            return var.dimensions, var.shape
        return None, None

    def walk(g, prefix=""):
        for name in sorted(g.groups.keys() if hasattr(g, "groups") else []):
            walk(g.groups[name], prefix + name + "/")
        for name in sorted(g.variables.keys() if hasattr(g, "variables") else []):
            v = g.variables[name]
            d, s = dims_and_shape(v)
            if d is not None:
                print(f"  {prefix}{name}: dims={d} shape={s}")

    print("Dimensions (file level):")
    for dim_name, dim in ds.dimensions.items():
        print(f"  {dim_name}: size={dim.size if not dim.isunlimited() else 'UNLIMITED'}")

    print("\nPRODUCT variables (pixel-level):")
    if "PRODUCT" in ds.groups:
        for name in sorted(ds["PRODUCT"].variables.keys()):
            v = ds["PRODUCT"].variables[name]
            d, s = dims_and_shape(v)
            if d is not None:
                print(f"  PRODUCT/{name}: dims={d} shape={s}")

    print("\nPRODUCT/SUPPORT_DATA/GEOLOCATIONS (corners):")
    if "PRODUCT" in ds.groups and "SUPPORT_DATA" in ds["PRODUCT"].groups:
        geo = ds["PRODUCT"]["SUPPORT_DATA"]["GEOLOCATIONS"]
        for name in ["latitude_bounds", "longitude_bounds"]:
            if name in geo.variables:
                v = geo.variables[name]
                d, s = dims_and_shape(v)
                print(f"  {name}: dims={d} shape={s}")

    print("\nPRODUCT/SUPPORT_DATA/DETAILED_RESULTS (no2_tot_vc, scd_no2, etc.):")
    if "PRODUCT" in ds.groups and "SUPPORT_DATA" in ds["PRODUCT"].groups:
        det = ds["PRODUCT"]["SUPPORT_DATA"]["DETAILED_RESULTS"]
        for name in ["summed_no2_total_vertical_column", "scd_no2", "cloud_radiance_fraction_no2"]:
            if name in det.variables:
                v = det.variables[name]
                d, s = dims_and_shape(v)
                print(f"  {name}: dims={d} shape={s}")

    print("\nPRODUCT/SUPPORT_DATA/INPUT_DATA (flags):")
    if "PRODUCT" in ds.groups and "SUPPORT_DATA" in ds["PRODUCT"].groups:
        inp = ds["PRODUCT"]["SUPPORT_DATA"]["INPUT_DATA"]
        for name in ["snow_ice_flag", "omi_xtrack_flags", "surface_albedo"]:
            if name in inp.variables:
                v = inp.variables[name]
                d, s = dims_and_shape(v)
                print(f"  {name}: dims={d} shape={s}")

    # Reader check: run read_OMI_KNMI and show shapes after _ensure_omi_dims
    print("\n--- Reader output (Tess_func.read_OMI_KNMI) ---")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from Tess_func import read_OMI_KNMI
        omi = read_OMI_KNMI(path)
        for key in ["no2_tot_vc", "no2_tot_sc", "Latitude", "Longitude", "CF", "no2_trop_vc",
                    "CornerLatitude", "CornerLongitude", "AMFtot", "AvKtot"]:
            if key in omi:
                arr = omi[key]
                print(f"  {key}: shape={getattr(arr, 'shape', 'N/A')} ndim={getattr(arr, 'ndim', 'N/A')}")
    except Exception as e:
        print("  (Reader check failed:", e, ")")

    ds.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
