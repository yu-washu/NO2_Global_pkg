import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc4
import os
import gc

# Parameters
res         = '1x1km'
species     = 'NO2'
year        = 2023
obs_version = 'v6'
geono2_version = 'v5.11'

GCHP_Dir = f'/my-projects2/1.project/GeoNO2-v5.11/{year}/'
OutDir   = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
Obs_Dir  = '/my-projects2/1.project/NO2_ground_compiled/monthly/global/'

MODEL_VARS = [
    'gchp_no2',
    'geophysical_no2_tot', 'geophysical_no2_trop', 'geophysical_no2_trop_TM5',
    'gap_SatColNO2_trop_gcshape', 'gap_SatColNO2_tot_gcshape',
    'gap_SatColNO2_trop',         'gap_SatColNO2_tot',
    'gchp_NO2col_tot',  'gchp_NO2col_trop',
    'gchp_eta_tot',     'gchp_eta_trop',     'gchp_eta_trop_clamped',
]

# ---------------------------------------------------------------------------
def get_nearest_grid_index(site_lon, site_lat, grid_lon_min, grid_lat_min, resolution=0.01):
    index_lon = np.round((site_lon - grid_lon_min) / resolution).astype(np.int32)
    index_lat = np.round((site_lat - grid_lat_min) / resolution).astype(np.int32)
    return index_lon, index_lat

def apply_cf(base, obs, alkylnitrates, hno3, pan, alpha=0.15, beta=0.95):
    """Correct observed NO2 to remove HNO3/PAN/alkylnitrate contributions."""
    denom = base + alkylnitrates + alpha * hno3 + beta * pan
    denom_safe = np.where(denom == 0, 1, denom)
    return obs * (base / denom_safe)

# ---------------------------------------------------------------------------
def load_obs():
    print("Loading observation data...")
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(Obs_Dir, 'combined_global_no2_2005-2023_v6_filtered.csv'),
        chunksize=100000, low_memory=False,
        dtype={'no2': float, 'year': int}
    ):
        yr = chunk[chunk['year'] == year]
        if len(yr) > 0:
            chunks.append(yr)
    obs = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"Loaded {len(obs)} monthly observations for {year}")
    return obs

def load_gchp_coords():
    print("Loading GCHP coordinates...")
    for month in range(1, 13):
        fp = os.path.join(GCHP_Dir, f'{res}.GeoNO2.{year}{month:02d}.MonMean.nc')
        if os.path.exists(fp):
            with nc4.Dataset(fp, 'r') as ds:
                lat = np.array(ds.variables['lat'][:], dtype=np.float32)
                lon = np.array(ds.variables['lon'][:], dtype=np.float32)
            return {'lat': lat, 'lon': lon,
                    'lat_min': lat.min(), 'lon_min': lon.min(),
                    'lat_max': lat.max(), 'lon_max': lon.max()}
    raise FileNotFoundError("No GeoNO2 files found in " + GCHP_Dir)

# ---------------------------------------------------------------------------
def process_monthly(obs_data, gchp_coords):
    print("Processing monthly data...")

    # Map every observation to its nearest GeoNO2 pixel
    idx_lon, idx_lat = get_nearest_grid_index(
        obs_data['lon'].values, obs_data['lat'].values,
        gchp_coords['lon_min'], gchp_coords['lat_min']
    )
    idx_lat = np.clip(idx_lat, 0, len(gchp_coords['lat']) - 1)
    idx_lon = np.clip(idx_lon, 0, len(gchp_coords['lon']) - 1)

    obs_indexed = obs_data.copy()
    obs_indexed['px_lat'] = idx_lat
    obs_indexed['px_lon'] = idx_lon

    # Average observations that fall in the same pixel-month
    print("Grouping observations by pixel-month...")
    grouped = (obs_indexed
               .groupby(['px_lat', 'px_lon', 'mon'])
               .agg(lat=('lat', 'mean'), lon=('lon', 'mean'),
                    no2=('no2', 'mean'), year=('year', 'first'))
               .reset_index())
    n_sites = (obs_indexed
               .groupby(['px_lat', 'px_lon', 'mon'])
               .size()
               .reset_index(name='num_sites'))
    grouped = grouped.merge(n_sites, on=['px_lat', 'px_lon', 'mon'])

    print("Extracting model values...")
    monthly_rows = []

    for month in sorted(grouped['mon'].unique()):
        mdf = grouped[grouped['mon'] == month].copy()
        fp  = os.path.join(GCHP_Dir, f'{res}.GeoNO2.{year}{int(month):02d}.MonMean.nc')

        if not os.path.exists(fp):
            print(f"[WARNING] Missing GeoNO2 file for month {month}")
            continue

        print(f"  Processing month {month}...")

        li = mdf['px_lat'].values
        lj = mdf['px_lon'].values

        # Load one full variable at a time, extract station points, free immediately
        # Peak memory: ~2.6 GB (one 18000x36000 grid) instead of 15 at once
        def _extract_nc4(ds, var):
            arr = np.array(ds.variables[var][:], dtype=np.float32)
            vals = arr[li, lj].copy()
            del arr
            return vals

        with nc4.Dataset(fp, 'r') as ds:
            v_gchp_no2   = _extract_nc4(ds, 'gchp_NO2')
            v_geo_tot    = _extract_nc4(ds, 'filled_GeoNO2_tot')
            v_geo_trop   = _extract_nc4(ds, 'filled_GeoNO2_trop')
            v_geo_trop_TM5 = _extract_nc4(ds, 'filled_GeoNO2_trop_TM5')
            v_sat_trop_gc = _extract_nc4(ds, 'gap_SatColNO2_trop_gcshape')
            v_sat_tot_gc  = _extract_nc4(ds, 'gap_SatColNO2_tot_gcshape')
            v_sat_trop   = _extract_nc4(ds, 'gap_SatColNO2_trop')
            v_sat_tot    = _extract_nc4(ds, 'gap_SatColNO2_tot')
            v_col_tot    = _extract_nc4(ds, 'gchp_NO2col_tot')
            v_col_trop   = _extract_nc4(ds, 'gchp_NO2col_trop')
            v_eta_tot    = _extract_nc4(ds, 'gchp_eta_tot')
            v_eta_trop   = _extract_nc4(ds, 'gchp_eta_trop')
            v_eta_trop_c = _extract_nc4(ds, 'gchp_eta_trop_clamped')
            v_alkyl      = _extract_nc4(ds, 'gchp_alkylnitrates')
            v_hno3       = _extract_nc4(ds, 'gchp_HNO3')
            v_pan        = _extract_nc4(ds, 'gchp_PAN')
        gc.collect()

        obs_cf = apply_cf(v_gchp_no2, mdf['no2'].values, v_alkyl, v_hno3, v_pan)

        monthly_rows.append(pd.DataFrame({
            'px_lat': mdf['px_lat'].values,
            'px_lon': mdf['px_lon'].values,
            'lat':   mdf['lat'].values,
            'lon':   mdf['lon'].values,
            'year':  mdf['year'].values,
            'month': month,
            'obs_no2':                   obs_cf,
            'gchp_no2':                  v_gchp_no2,
            'geophysical_no2_tot':       v_geo_tot,
            'geophysical_no2_trop':      v_geo_trop,
            'geophysical_no2_trop_TM5':  v_geo_trop_TM5,
            'gap_SatColNO2_trop_gcshape': v_sat_trop_gc,
            'gap_SatColNO2_tot_gcshape':  v_sat_tot_gc,
            'gap_SatColNO2_trop':         v_sat_trop,
            'gap_SatColNO2_tot':          v_sat_tot,
            'gchp_NO2col_tot':           v_col_tot,
            'gchp_NO2col_trop':          v_col_trop,
            'gchp_eta_tot':              v_eta_tot,
            'gchp_eta_trop':             v_eta_trop,
            'gchp_eta_trop_clamped':     v_eta_trop_c,
            'num_sites':                 mdf['num_sites'].values,
        }))
        gc.collect()

    return pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()

# ---------------------------------------------------------------------------
def main():
    global year, GCHP_Dir, OutDir

    parser = argparse.ArgumentParser(description='Global NO2 evaluation (GeoNO2 v5.4)')
    parser.add_argument('--year', type=int, required=True)
    args = parser.parse_args()

    year     = args.year
    GCHP_Dir = f'/my-projects2/1.project/GeoNO2-v5.11/{year}/'
    OutDir   = f'/my-projects2/1.project/Evaluation/obs{obs_version}/'
    os.makedirs(OutDir, exist_ok=True)

    print(f"Starting global NO2 evaluation (GeoNO2 v5.4) for year {year}...")

    obs_data    = load_obs()
    gchp_coords = load_gchp_coords()

    # Monthly output
    monthly_df = process_monthly(obs_data, gchp_coords)
    monthly_out = os.path.join(OutDir, f'{species}_monthly_{year}_obs{obs_version}_geono2-{geono2_version}.csv')
    monthly_df.to_csv(monthly_out, index=False)
    print(f"Saved monthly data: {monthly_out} ({len(monthly_df)} rows)")

    # Annual output: average across months per pixel (group by pixel index, not float coords)
    agg_dict = {v: 'mean' for v in ['obs_no2', 'lat', 'lon'] + MODEL_VARS}
    agg_dict['num_sites'] = 'sum'
    annual = (monthly_df
              .groupby(['px_lat', 'px_lon'])
              .agg(agg_dict)
              .reset_index()
              .drop(columns=['px_lat', 'px_lon']))
    annual['year'] = year

    annual_out = os.path.join(OutDir, f'{species}_annual_{year}_obs{obs_version}_geono2-{geono2_version}.csv')
    annual.to_csv(annual_out, index=False)
    print(f"Saved annual data: {annual_out} ({len(annual)} rows)")

    print("\nCompleted successfully!")

if __name__ == "__main__":
    main()
