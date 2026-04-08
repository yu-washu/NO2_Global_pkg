import numpy as np
import netCDF4 as nc

def ncsave(sfname, no2, tlat, tlon):
    
    """
    Save NO2 data to NetCDF format.
    """
    with nc.Dataset(sfname, 'w', format='NETCDF4') as ds:
        ds.createDimension('lat', len(tlat))
        ds.createDimension('lon', len(tlon))
        
        lat_var = ds.createVariable('lat', 'f4', ('lat',))
        lon_var = ds.createVariable('lon', 'f4', ('lon',))
        no2_var = ds.createVariable('no2', 'f4', ('lat', 'lon'))
        
        lat_var[:] = tlat
        lon_var[:] = tlon
        no2_var[:, :] = no2
        
        lat_var.units = 'degrees_north'
        lon_var.units = 'degrees_east'
        no2_var.units = 'molec/cm2'
        no2_var.long_name = 'NO2 Vertical Column Density'
    
    print(f"Saved: {sfname}")

def read_OMI_MINDS(filename):
    with nc.Dataset(filename, 'r') as ds:
        science_data = ds['SCIENCE_DATA']
        ancillary_data = ds['ANCILLARY_DATA']
        geolocation_data = ds['GEOLOCATION_DATA']
        
        scd_tot = science_data['SlantColumnAmountNO2'][:]
        vcd_tot = science_data['ColumnAmountNO2'][:]
        scattering_weight = science_data['ScatteringWeight'][:]
        
        VcdQualityFlags = science_data['VcdQualityFlags'][:]
        
        ECF = ancillary_data['CloudFraction'][:] # ECF: Effective Cloud Fraction
        XTrackQualityFlags = ancillary_data['XTrackQualityFlags'][:] #RowAnomalyFlag, 0: not affected by Row Anomaly, 1: affected by RowAnomalyFlag
        
        latitude = geolocation_data['Latitude'][:]
        longitude = geolocation_data['Longitude'][:]
        scattering_wt_pressure = geolocation_data['ScatteringWeightPressure'][:]
        
        solar_zenith_angle = geolocation_data['SolarZenithAngle'][:]

        return {
            'filename': filename,
            'QualityFlag': VcdQualityFlags,
            'RowAnomalyFlag':XTrackQualityFlags ,
            'ECF': ECF,
            'CornerLatitude': geolocation_data['FoV75CornerLatitude'][:],
            'CornerLongitude': geolocation_data['FoV75CornerLongitude'][:],
            'Latitude': latitude,
            'Longitude': longitude,
            'sza': solar_zenith_angle,
            'no2_tot_sc': scd_tot,
            'no2_tot_vc': vcd_tot,
            'sw': scattering_weight,
            'swp': scattering_wt_pressure,
        }

def read_OMI_KNMI(filename):
    with nc.Dataset(filename, 'r') as ds:
        tm5a = ds['PRODUCT']['tm5_pressure_level_a'][:]
        tm5b = ds['PRODUCT']['tm5_pressure_level_b'][:]
        # QA4ECV PSD §3.3: use tm5_surface_pressure (meteorologically-corrected) for
        # pressure-level reconstruction, NOT surface_pressure from the cloud product.
        # tm5_surface_pressure in QA4ECV is in hPa; multiply by 100 to match the
        # Pa units of tm5_pressure_level_a (same convention as TROPOMI surface_pressure).
        ps = ds['PRODUCT']['tm5_surface_pressure'][:] * 100.0
        p_bottom = tm5a[:, 0][np.newaxis, np.newaxis, np.newaxis, :] + tm5b[:, 0][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis]
        p_top = tm5a[:, 1][np.newaxis, np.newaxis, np.newaxis, :] + tm5b[:, 1][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis]
        p =  0.5*(p_bottom + p_top) * 0.01 #Pressure in hPa

        AvKtot = ds['PRODUCT']['averaging_kernel'][:]
        AMFtot = ds['PRODUCT']['amf_total'][:]
        AMFtrop = ds['PRODUCT']['amf_trop'][:]
        tpp = ds['PRODUCT']['layer'][:]
        trop_layer = ds['PRODUCT']['tm5_tropopause_layer_index'][:]  # (1, nscan, npix) per-pixel

        no2_trop_vc = ds['PRODUCT']['tropospheric_no2_vertical_column'][:]
        no2_trop_sc = no2_trop_vc * AMFtrop
        scale      = (AMFtot / AMFtrop)[..., np.newaxis]
        AvKtrop    = AvKtot * scale

        return {
            'filename': filename,
            'CornerLatitude': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['latitude_bounds'][:],
            'CornerLongitude': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['longitude_bounds'][:],
            'Latitude': ds['PRODUCT']['latitude'][:],
            'Longitude': ds['PRODUCT']['longitude'][:],
            'scanIndex': ds['PRODUCT']['scanline'][:],
            'ground_pixel': ds['PRODUCT']['ground_pixel'][:],
            'AMFtrop': AMFtrop,
            'AMFtot': AMFtot,
            'AvKtot': AvKtot,
            'AvKtrop': AvKtrop,
            'tm5a': tm5a,
            'tm5b': tm5b,
            'tpp': tpp,
            'p': p,
            'QualityFlag': ds['PRODUCT']['processing_error_flag'][:],
            'SnowIceFlag': ds['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['snow_ice_flag'][:],
            'SurfaceAlbedo': ds['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['surface_albedo'][:],
            'RowAnomalyFlag': ds['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['omi_xtrack_flags'][:],
            'sza': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['solar_zenith_angle'][:],
            'CF': ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['cloud_radiance_fraction_no2'][:],
            'no2_trop_vc': no2_trop_vc,
            'no2_trop_sc': no2_trop_sc,
            'no2_tot_vc': ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['summed_no2_total_vertical_column'][:],
            'no2_tot_sc' : ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['scd_no2'][:],

        }

def read_TROPOMI(filename):
    # import h5netcdf.legacyapi as nc
    with nc.Dataset(filename, 'r') as ds:
        tm5a = ds['PRODUCT']['tm5_constant_a'][:]
        tm5b = ds['PRODUCT']['tm5_constant_b'][:]
        ps = ds['PRODUCT']['SUPPORT_DATA']['INPUT_DATA']['surface_pressure'][:]
        p_bottom = tm5a[:, 0][np.newaxis, np.newaxis, np.newaxis, :] + tm5b[:, 0][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis]
        p_top = tm5a[:, 1][np.newaxis, np.newaxis, np.newaxis, :] + tm5b[:, 1][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis]
        p =  0.5*(p_bottom + p_top) * 0.01 #Pressure in hPa

        AvKtot = ds['PRODUCT']['averaging_kernel'][:]
        AMFtot = ds['PRODUCT']['air_mass_factor_total'][:]
        AMFtrop = ds['PRODUCT']['air_mass_factor_troposphere'][:]
        tpp = ds['PRODUCT']['layer'][:]
        trop_layer = ds['PRODUCT']['tm5_tropopause_layer_index'][:]  # (1, nscan, npix) per-pixel

        no2_trop_vc = ds['PRODUCT']['nitrogendioxide_tropospheric_column'][:] * ds['PRODUCT']['nitrogendioxide_tropospheric_column'].multiplication_factor_to_convert_to_molecules_percm2
        no2_trop_sc = no2_trop_vc * AMFtrop

        # Derive AvKtrop from AvKtot following ATBD section 6.4.5:
        #   AvKtrop = AvKtot * (AMFtot / AMFtrop),  stratospheric layers = 0
        # trop_layer is per-pixel (1, nscan, npix); layers >= trop_layer are stratospheric.
        scale   = (AMFtot / AMFtrop)[..., np.newaxis]                        # (1, nscan, npix, 1)
        AvKtrop = AvKtot * scale                                              # (1, nscan, npix, 34)

        return {
            'filename': filename,
            'CornerLatitude': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['latitude_bounds'][:],
            'CornerLongitude': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['longitude_bounds'][:],
            'Latitude': ds['PRODUCT']['latitude'][:],
            'Longitude': ds['PRODUCT']['longitude'][:],
            'scanIndex': ds['PRODUCT']['scanline'][:],
            'ground_pixel': ds['PRODUCT']['ground_pixel'][:],
            'QualityFlag': ds['PRODUCT']['qa_value'][:],
            'sza': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['solar_zenith_angle'][:],
            'CF': ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['cloud_radiance_fraction_nitrogendioxide_window'][:],
            # 'CF': ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['cloud_fraction_crb_nitrogendioxide_window'][:],
            'vaa': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['viewing_azimuth_angle'][:],
            'vza': ds['PRODUCT']['SUPPORT_DATA']['GEOLOCATIONS']['viewing_zenith_angle'][:],
            'no2_trop_vc': no2_trop_vc,
            'no2_trop_sc': no2_trop_sc,
            'no2_tot_vc': ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_total_column'][:] * ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_total_column'].multiplication_factor_to_convert_to_molecules_percm2,
            'no2_tot_sc' : ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_slant_column_density'][:]* ds['PRODUCT']['SUPPORT_DATA']['DETAILED_RESULTS']['nitrogendioxide_total_column'].multiplication_factor_to_convert_to_molecules_percm2,
            'AMFtrop': AMFtrop,
            'AMFtot': AMFtot,
            'AvKtot': AvKtot,
            'AvKtrop': AvKtrop,
            'tpp': tpp,
            'layer': tpp,
            'tm5a': tm5a,
            'tm5b': tm5b,
            'p': p,
        }

def write_tessellation_input_grid_file(para_filename, filename_datain, filename_dataout, lat, lon, latlim=None, lonlim=None):
    # WRITE_TESSELLATION_INPUT_GRID_FILE setup tessellation input grid
    #
    # WRITE_TESSELLATION_INPUT_GRID
    # If LAT and Lon are scalars, the grid is setup on a regular grid using LAT
    # and LON as the grid-box spacing. If they are vectors, they must be a
    # series of predefined latitude and longitude edges for the grid box (ie,
    # model-defined boxes). These should be on a regular grid, but the edges can be irregular. 
    # LATLIM and LONLIM are optional arguments which can
    # be used to define the limits of the grid when the scalar grid spacing is
    # provided.
    #
    # Caroline Nowlan, 19-Oct-2012

    # Regular grid over entire world

    if np.isscalar(lat) and np.isscalar(lon):
        dlat = lat
        dlon = lon
        if latlim is None or lonlim is None:
            # Whole world
            irgrid_flag = 0
            latlim = [-90, 90]
            lonlim = [-180, 180]
        else:
            # Defined limits --> create lat/lon grid
            irgrid_flag = 1
            latgrid = np.arange(latlim[0], latlim[1] + dlat, dlat)
            longrid = np.arange(lonlim[0], lonlim[1] + dlon, dlon)
            nlat = len(latgrid) - 1
            nlon = len(longrid) - 1
    else:
        irgrid_flag = 1
        dlat = lat[1] - lat[0]
        dlon = lon[1] - lon[0]
        nlat = len(lat)-1
        nlon = len(lon)-1
        latgrid = lat
        longrid = np.concatenate((lon, np.arange(lon[-1] + dlon, 360 + dlon, dlon)))

    with open(para_filename, 'w') as fid:
        line1 = f"{nlon:5d}  {nlat:5d}  {irgrid_flag:1d}\n"
        fid.write(line1)
        fid.write(f"{filename_datain}\n")
        fid.write(f"{filename_dataout}\n")

        if irgrid_flag == 0:
            unitarea = 100
        else:
            fid.write(' '.join(f"{val:10.3f}" for val in longrid) + '\n')
            fid.write(' '.join(f"{val:10.3f}" for val in latgrid) + '\n')
            unitarea = (dlat * dlon).item()

        fid.write(f"{unitarea:9.4f}\n")

def load_and_save_OMI_KNMI_to_nc(in_filename, out_filename, nvar, tlat, tlon):
    import xarray as xr
    """
    Load ASCII tessellation output file, process NO2 data, and save to NetCDF.

    Handles:
    - Fortran-style 'D' exponent notation
    - Skips inconsistent header lines
    - Enforces uniform column count using `usecols`
    - Saves to NetCDF format

    Parameters:
    - in_filename: str, input ASCII file path
    - out_filename: str, output NetCDF file path
    - tlat: np.array, latitude grid centers
    - tlon: np.array, longitude grid centers

    Returns:
    - ds: xr.array, processed NO2 Data Array
    """
    import datetime
    sfname = out_filename
    ncols = len(tlon)
    with open(in_filename, 'r') as f:
        raw_lines = [line.replace('D', 'E').strip() for line in f if line.strip()]
        padded = []
        for line in raw_lines:
            vals = line.split()
            if len(vals) < ncols:
                vals += ['0'] * (ncols - len(vals))
            elif len(vals) > ncols:
                vals = vals[:ncols]
            padded.append(' '.join(vals))
        data = np.loadtxt(padded)

        nrows, nlonc = data.shape
        # Fortran tessellation output: nvar data blocks + 1 area block + 1 pixel-count block
        nlatc = nrows // (2 + nvar)
        if nlonc != len(tlon):
            raise ValueError(f"Longitude grids do not match: {nlonc} vs {len(tlon)}")
        if nlatc != len(tlat):
            raise ValueError(f"Latitude grids do not match: {nlatc} vs {len(tlat)}")

        no2_trop = data[:nlatc, :]
        no2_trop[no2_trop == 0] = np.nan

        no2_trop_gcshape = data[nlatc:2*nlatc, :]
        no2_trop_gcshape[no2_trop_gcshape == 0] = np.nan

        no2_tot = data[2*nlatc:3*nlatc, :]
        no2_tot[no2_tot == 0] = np.nan

        no2_tot_gcshape = data[3*nlatc:4*nlatc, :]
        no2_tot_gcshape[no2_tot_gcshape == 0] = np.nan


        ds = xr.Dataset(
                data_vars={
                'NO2_trop': (['lat', 'lon'], 
                             no2_trop, 
                             {'units': 'molec cm-2', 'long_name': 'OMI-KNMI Tropospheric NO2 Vertical Column'}
                             ),
                'NO2_trop_gcshape': (['lat', 'lon'], 
                                    no2_trop_gcshape, 
                                    {'units': 'molec cm-2', 'long_name': 'OMI-KNMI Tropospheric NO2 Column Reconstructed with GEOS-Chem Shape Factor'}
                                    ),
                'NO2_tot':(['lat', 'lon'], 
                           no2_tot, 
                           {'units': 'molec cm-2', 'long_name': 'OMI-KNMI Total NO2 Vertical Column'}
                           ),
                'NO2_tot_gcshape': (['lat', 'lon'], 
                                    no2_tot_gcshape, 
                                    {'units': 'molec cm-2', 'long_name': 'OMI-KNMI Total NO2 Column Reconstructed with GEOS-Chem Shape Factor'}
                                    ),
                },
                coords={'lat': tlat, 'lon': tlon},
                attrs={
                        "title": "Global OMI-KNMI NO2 Tessellated Product",
                        "institution": "WashU ACAG",
                        "source": "OMI-KNMI L2 NO2 and GCHP c180 shape factors",
                        "history": f"Created on {datetime.datetime.now().isoformat()}",
                        "description": "Tessellated and regridded NO2 product at 0.01° resolution using Rob Spurr's tessellation code.",
                        "contact": "Yu Yan <yany1@wustl.edu>"
                    }
        )
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        encoding['lat'] = {'_FillValue': None}
        encoding['lon'] = {'_FillValue': None}
        ds.to_netcdf(sfname, encoding=encoding)

        return ds

def load_and_save_OMI_KNMI_AMFtot_to_nc(in_filename, out_filename, nvar, tlat, tlon):
    import xarray as xr
    """
    Load ASCII tessellation output file, process NO2 data, and save to NetCDF.

    Handles:
    - Fortran-style 'D' exponent notation
    - Skips inconsistent header lines
    - Enforces uniform column count using `usecols`
    - Saves to NetCDF format

    Parameters:
    - in_filename: str, input ASCII file path
    - out_filename: str, output NetCDF file path
    - tlat: np.array, latitude grid centers
    - tlon: np.array, longitude grid centers

    Returns:
    - ds: xr.array, processed NO2 Data Array
    """
    import datetime
    sfname = out_filename
    ncols = len(tlon)
    with open(in_filename, 'r') as f:
        raw_lines = [line.replace('D', 'E').strip() for line in f if line.strip()]
        padded = []
        for line in raw_lines:
            vals = line.split()
            if len(vals) < ncols:
                vals += ['0'] * (ncols - len(vals))
            elif len(vals) > ncols:
                vals = vals[:ncols]
            padded.append(' '.join(vals))
        data = np.loadtxt(padded)

        nvar = nvar
        nrows, nlonc = data.shape

        nlatc = nrows // (2 + nvar)
        if nlonc != len(tlon):
            raise ValueError(f"Longitude grids do not match: {nlonc} vs {len(tlon)}")
        if nlatc != len(tlat):
            raise ValueError(f"Latitude grids do not match: {nlatc} vs {len(tlat)}")
        
        # in_filename is the output file from fortran tessellation software   
        #Block 1-3: Variable 1-3 (e.g. no2_trop)
        # Row 1: lon1 lon2 lon3 ... lonN
        # Row 2: ...
        # ...
        # Row nlat
        # 
        # ...
        # Block 4: Total area
        # ...
        # Block 5: Number of contributing pixels (integer)
        
        # Extract the first block
        amf_trop = data[:nlatc, :]
        amf_trop[amf_trop == 0] = np.nan
        
        # # Extract the second block
        amf_trop_gcshape = data[nlatc:2*nlatc,:]
        amf_trop_gcshape[amf_trop_gcshape == 0] = np.nan
        
        # Extract the third block
        amf_tot = data[2*nlatc:3*nlatc, :]
        amf_tot[amf_tot == 0] = np.nan
        
        # Extract the fourth block
        amf_tot_gcshape = data[3*nlatc:4*nlatc, :]
        amf_tot_gcshape[amf_tot_gcshape == 0] = np.nan

        ds = xr.Dataset(
                data_vars={
                'amf_trop': (['lat', 'lon'], 
                             amf_trop, 
                             {'units': '1', 'long_name': 'OMI-KNMI Total AMF'}
                             ),
                'amf_trop_gcshape': (['lat', 'lon'], 
                                    amf_trop_gcshape, 
                                    {'units': '1', 'long_name': 'GCHP-Profile-Shaped OMI-KNMI Tropospheric AMF'}
                                    ),
                'amf_tot':(['lat', 'lon'], 
                           amf_tot, 
                           {'units': '1', 'long_name': 'OMI-KNMI Total AMF'}
                           ),
                'amf_tot_gcshape': (['lat', 'lon'], 
                                    amf_tot_gcshape, 
                                    {'units': '1', 'long_name': 'GCHP-Profile-Shaped OMI-KNMI Total AMF'}
                                    ),
                },
                coords={'lat': tlat, 'lon': tlon},
                attrs={
                        "title": "Global OMI-KNMI AMFtot + AMF_gcshape Tessellated Product",
                        "institution": "WashU ACAG",
                        "source": "OMI-KNMI L2 AMFtot and GCHP-Profile-Shaped AMF",
                        "history": f"Created on {datetime.datetime.now().isoformat()}",
                        "description": "Tessellated and regridded AMFtot and GCHP-Profile-Shaped AMF at 0.01° resolution using Rob Spurr's tessellation code.",
                        "contact": "Yu Yan <yany1@wustl.edu>"
                    }
        )
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        encoding['lat'] = {'_FillValue': None}
        encoding['lon'] = {'_FillValue': None}
        ds.to_netcdf(sfname, encoding=encoding)

        return ds

def load_and_save_TROPOMI_to_nc(in_filename, out_filename, nvar, tlat, tlon):
    import xarray as xr
    """
    Load ASCII tessellation output file, process NO2 data, and save to NetCDF.

    Handles:
    - Fortran-style 'D' exponent notation
    - Skips inconsistent header lines
    - Enforces uniform column count using `usecols`
    - Saves to NetCDF format

    Parameters:
    - in_filename: str, input ASCII file path
    - out_filename: str, output NetCDF file path
    - tlat: np.array, latitude grid centers
    - tlon: np.array, longitude grid centers

    Returns:
    - ds: xr.array, processed NO2 Data Array
    """
    import datetime
    sfname = out_filename
    with open(in_filename, 'r') as f:
        lines = [line.replace('D', 'E').strip() for line in f if line.strip()]
        data = np.loadtxt(lines)

        nvar = nvar
        nrows, nlonc = data.shape

        nlatc = nrows // (2 + nvar)
        if nlonc != len(tlon):
            raise ValueError(f"Longitude grids do not match: {nlonc} vs {len(tlon)}")
        if nlatc != len(tlat):
            raise ValueError(f"Latitude grids do not match: {nlatc} vs {len(tlat)}")
        
        # in_filename is the output file from fortran tessellation software   
        #Block 1-3: Variable 1-3 (e.g. no2_trop)
        # Row 1: lon1 lon2 lon3 ... lonN
        # Row 2: ...
        # ...
        # Row nlat
        # 
        # ...
        # Block 4: Total area
        # ...
        # Block 5: Number of contributing pixels (integer)
        
        # Extract the first block
        no2_trop = data[:nlatc, :]
        no2_trop[no2_trop == 0] = np.nan
        
        # # Extract the second block
        no2_trop_gcshape = data[nlatc:2*nlatc,:]
        no2_trop_gcshape[no2_trop_gcshape == 0] = np.nan
        
        # #Extract the third block
        no2_tot = data[2*nlatc:3*nlatc, :]
        no2_tot[no2_tot == 0] = np.nan
        
        # Extract the fourth block
        no2_tot_gcshape = data[3*nlatc:4*nlatc, :]
        no2_tot_gcshape[no2_tot_gcshape == 0] = np.nan
        
        # Extract the second block
        # no2_tot = data[:nlatc, :]
        # no2_tot[no2_tot == 0] = np.nan
        
        # #Extract the third block
        # no2_tot_gcshape = data[nlatc:2*nlatc,:]
        # no2_tot_gcshape[no2_tot_gcshape == 0] = np.nan

        ds = xr.Dataset(
                data_vars={
                'NO2_trop': (['lat', 'lon'], 
                             no2_trop, 
                             {'units': 'molec cm-2', 'long_name': 'TROPOMI Tropospheric NO2 Vertical Column'}
                             ),
                'NO2_trop_gcshape': (['lat', 'lon'], 
                                    no2_trop_gcshape, 
                                    {'units': 'molec cm-2', 'long_name': 'TROPOMI Tropospheric NO2 Column Reconstructed with GEOS-Chem Shape Factor'}
                                    ),
                'NO2_tot':(['lat', 'lon'], 
                           no2_tot, 
                           {'units': 'molec cm-2', 'long_name': 'TROPOMI Total NO2 Vertical Column'}
                           ),
                'NO2_tot_gcshape': (['lat', 'lon'], 
                                    no2_tot_gcshape, 
                                    {'units': 'molec cm-2', 'long_name': 'TROPOMI Total NO2 Column Reconstructed with GEOS-Chem Shape Factor'}
                                    ),
                },
                coords={'lat': tlat, 'lon': tlon},
                attrs={
                        "title": "Global TROPOMI NO2 Tessellated Product",
                        "institution": "WashU ACAG",
                        "source": "TROPOMI L2 NO2 and GCHP c180 shape factors",
                        "history": f"Created on {datetime.datetime.now().isoformat()}",
                        "description": "Tessellated and regridded NO2 product at 0.01° resolution using Rob Spurr's tessellation code.",
                        "contact": "Yu Yan <yany1@wustl.edu>"
                    }
        )
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        encoding['lat'] = {'_FillValue': None}
        encoding['lon'] = {'_FillValue': None}
        ds.to_netcdf(sfname, encoding=encoding)

        return ds
    
def load_and_save_TROPOMI_AMFtot_to_nc(in_filename, out_filename, nvar, tlat, tlon):
    import xarray as xr
    import datetime
    with open(in_filename, 'r') as f:
        lines = [line.replace('D', 'E').strip() for line in f if line.strip()]
        data = np.loadtxt(lines)

    nrows, nlonc = data.shape
    nlatc = nrows // (2 + nvar)
    if nlonc != len(tlon):
        raise ValueError(f"Longitude grids do not match: {nlonc} vs {len(tlon)}")
    if nlatc != len(tlat):
        raise ValueError(f"Latitude grids do not match: {nlatc} vs {len(tlat)}")

    amf_trop = data[:nlatc, :]
    amf_trop[amf_trop == 0] = np.nan

    amf_trop_gcshape = data[nlatc:2 * nlatc, :]
    amf_trop_gcshape[amf_trop_gcshape == 0] = np.nan

    amf_tot = data[2*nlatc:3*nlatc, :]
    amf_tot[amf_tot == 0] = np.nan

    amf_tot_gcshape = data[3*nlatc:4*nlatc, :]
    amf_tot_gcshape[amf_tot_gcshape == 0] = np.nan

    ds = xr.Dataset(
        data_vars={
            'amf_trop': (['lat', 'lon'],
                        amf_trop,
                        {'units': '1',
                         'long_name': 'TROPOMI Total Air Mass Factor'}),
            'amf_trop_gcshape': (['lat', 'lon'],
                            amf_trop_gcshape,
                            {'units': '1',
                             'long_name': 'GCHP-Profile-Shaped Total AMF '
                                          '(sum(prof_tm5) / (AMFtot * sum(prof_tm5 * AvK)))'}),
            'amf_tot': (['lat', 'lon'],
                        amf_tot,
                        {'units': '1',
                         'long_name': 'TROPOMI Total Air Mass Factor'}),
            'amf_tot_gcshape': (['lat', 'lon'],
                            amf_tot_gcshape,
                            {'units': '1',
                             'long_name': 'GCHP-Profile-Shaped Total AMF '
                                          '(sum(prof_tm5) / (AMFtot * sum(prof_tm5 * AvK)))'}),
        },
        coords={'lat': tlat, 'lon': tlon},
        attrs={
            "title": "Global TROPOMI AMFtot + AMF_gcshape Tessellated Product",
            "institution": "WashU ACAG",
            "source": "TROPOMI L2 NO2 + GCHP c180 shape factors",
            "history": f"Created on {datetime.datetime.now().isoformat()}",
            "description": "Tessellated and regridded AMFtot and GCHP-shaped AMF at 0.01° resolution.",
            "contact": "Yu Yan <yany1@wustl.edu>"
        }
    )
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
    encoding['lat'] = {'_FillValue': None}
    encoding['lon'] = {'_FillValue': None}
    ds.to_netcdf(out_filename, encoding=encoding)
    return ds

def load_and_save_OMI_MINDS_to_nc(in_filename, out_filename, nvar, tlat, tlon):
    import xarray as xr
    """
    Load ASCII tessellation output file, process NO2 data, and save to NetCDF.

    Handles:
    - Fortran-style 'D' exponent notation
    - Skips inconsistent header lines
    - Enforces uniform column count using `usecols`
    - Saves to NetCDF format

    Parameters:
    - in_filename: str, input ASCII file path
    - out_filename: str, output NetCDF file path
    - tlat: np.array, latitude grid centers
    - tlon: np.array, longitude grid centers

    Returns:
    - ds: xr.array, processed NO2 Data Array
    """
    import datetime
    sfname = out_filename
    with open(in_filename, 'r') as f:
        lines = [line.replace('D', 'E').strip() for line in f if line.strip()]
        data = np.loadtxt(lines)

        nvar = nvar
        nrows, nlonc = data.shape

        nlatc = nrows // (2 + nvar)
        if nlonc != len(tlon):
            raise ValueError(f"Longitude grids do not match: {nlonc} vs {len(tlon)}")
        if nlatc != len(tlat):
            raise ValueError(f"Latitude grids do not match: {nlatc} vs {len(tlat)}")
        
        # in_filename is the output file from fortran tessellation software   
        #Block 1-3: Variable 1-3 (e.g. no2_trop)
        # Row 1: lon1 lon2 lon3 ... lonN
        # Row 2: ...
        # ...
        # Row nlat
        # 
        # ...
        # Block 4: Total area
        # ...
        # Block 5: Number of contributing pixels (integer)
        
        # Extract the first block
        # no2_trop = data[:nlatc, :]
        # no2_trop[no2_trop == 0] = np.nan
        
        # # Extract the second block
        # no2_tot = data[nlatc:2*nlatc,:]
        # no2_tot[no2_tot == 0] = np.nan
        
        # #Extract the third block
        # no2_tot_gcshape = data[2*nlatc:3*nlatc, :]
        # no2_tot_gcshape[no2_tot_gcshape == 0] = np.nan
        
        # Extract the second block
        no2_tot = data[:nlatc, :]
        no2_tot[no2_tot == 0] = np.nan
        
        #Extract the third block
        no2_tot_gcshape = data[nlatc:2*nlatc,:]
        no2_tot_gcshape[no2_tot_gcshape == 0] = np.nan

        ds = xr.Dataset(
                data_vars={
                # 'NO2_trop': (['lat', 'lon'], 
                #              no2_trop, 
                #              {'units': 'molec cm-2', 'long_name': 'TROPOMI Tropospheric NO2 Vertical Column'}
                #              ),
                'NO2_tot':(['lat', 'lon'], 
                           no2_tot, 
                           {'units': 'molec cm-2', 'long_name': 'OMI MINDS Total NO2 Vertical Column'}
                           ),
                'NO2_tot_gcshape': (['lat', 'lon'], 
                                    no2_tot_gcshape, 
                                    {'units': 'molec cm-2', 'long_name': 'OMI MINDS Total NO2 Column Reconstructed with GEOS-Chem Shape Factor'}
                                    ),
                },
                coords={'lat': tlat, 'lon': tlon},
                attrs={
                        "title": "Global OMI-MINDS NO2 Tessellated Product",
                        "institution": "WashU ACAG",
                        "source": "OMI-MINDS L2 NO2 and GCHP c180 shape factors",
                        "history": f"Created on {datetime.datetime.now().isoformat()}",
                        "description": "Tessellated and regridded NO2 product at 0.01° resolution using Rob Spurr's tessellation code.",
                        "contact": "Yu Yan <yany1@wustl.edu>"
                    }
        )
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
        encoding['lat'] = {'_FillValue': None}
        encoding['lon'] = {'_FillValue': None}
        ds.to_netcdf(sfname, encoding=encoding)

        return ds
