from data_func.iostream import load_RawObs_training_data,load_RawObservation, load_GeoLatLon, load_GeoLatLon_Map, load_BLISCO_data,load_mahalanobis_distance_data
from data_func.calculation import calculate_covariance_matrix, invert_matrix,calculate_mahalanobis_distance
from data_func.utils import neighbors_haversine_indices,Obs_version
from map_uncertainty_func.iostream import save_absoulute_uncertainty_map,load_absolute_uncertainty_map,load_estimation_map_data,load_rRMSE_map,save_rRMSE_map,load_bins_LOWESS_values,save_mahalanobis_distance_map,load_mahalanobis_distance_map,get_landtype, save_pixel_nearby_sites_index_map,load_pixels_nearest_sites_indices_map, save_local_reference_map, load_local_reference_map, load_mapdata
from map_uncertainty_func.utils import inputfiles_table
from map_uncertainty_func.data_func import Get_Mahalanobis_Distances
import numpy as np
from data_func.iostream import load_TrainingVariables

def Get_absolute_uncertainty_map(species,version,special_name,YYYY,MM,obs_version,nearby_sites_number,
                                 map_estimation_special_name,map_estimation_version):
    '''
    Get absolute uncertainty map for given species and date.
    '''
    print(f"Calculating absolute uncertainty map for species: {species}, version: {version}, date: {YYYY}-{MM}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    rRMSE_uncertainty_map = load_rRMSE_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                           obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    
    if MM != 12:
        map_data, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[MM], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
        padded_map = np.zeros(rRMSE_uncertainty_map.shape,dtype=np.float64)
        
        # Determine the padding offset dynamically based on the shape difference
        rh, rw = rRMSE_uncertainty_map.shape
        h, w = map_data.shape
        pad_h = (rh - h) // 2
        pad_w = (rw - w) // 2
        padded_map[pad_h:pad_h+h, pad_w:pad_w+w] = map_data
    else:
        padded_map = np.zeros(rRMSE_uncertainty_map.shape,dtype=np.float64)
        rh, rw = rRMSE_uncertainty_map.shape
        for m in range(12):
            monthly_map, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[m], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
            h, w = monthly_map.shape
            pad_h = (rh - h) // 2
            pad_w = (rw - w) // 2
            padded_map[pad_h:pad_h+h, pad_w:pad_w+w] += monthly_map
        padded_map = padded_map / 12.0
    absolute_uncertainty_map = padded_map * rRMSE_uncertainty_map 
    save_absoulute_uncertainty_map(absolute_uncertainty_map=absolute_uncertainty_map,species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY=YYYY,MM=MONTH[MM],
                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    return

def Get_longterm_average_absolute_uncertainty_map(species,version,special_name,YYYY_list:np.int32,MM:np.int32,obs_version,nearby_sites_number,
                                 map_estimation_special_name,map_estimation_version):
    '''
    Get longterm average absolute uncertainty map for given species and years/months.
    '''
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Calculating longterm average absolute uncertainty map for species: {species}, version: {version}, month: {MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    else:
        print(f"Calculating longterm average absolute uncertainty map for species: {species}, version: {version}, all months, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    longterm_average_absolute_uncertainty_map = None
    count = 0
    for YYYY in YYYY_list:
        if MM != 12:
            temp_absolute_uncertainty_map = load_absolute_uncertainty_map(species=species,version=version,special_name=special_name,YYYY=YYYY,MM=MONTH[MM],
                                                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
        else:
            temp_absolute_uncertainty_map = np.zeros((13000,36000),dtype=np.float64)
            for m in range(12):
                monthly_absolute_uncertainty_map = load_absolute_uncertainty_map(species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY=YYYY,MM=MONTH[m],
                                                                       obs_version=obs_version,nearby_sites_number=nearby_sites_number)
                temp_absolute_uncertainty_map += monthly_absolute_uncertainty_map
            temp_absolute_uncertainty_map = temp_absolute_uncertainty_map / 12.0
        if longterm_average_absolute_uncertainty_map is None:
            longterm_average_absolute_uncertainty_map = np.zeros_like(temp_absolute_uncertainty_map)
        longterm_average_absolute_uncertainty_map += temp_absolute_uncertainty_map
        count += 1
    longterm_average_absolute_uncertainty_map = longterm_average_absolute_uncertainty_map / count
    save_absoulute_uncertainty_map(absolute_uncertainty_map=longterm_average_absolute_uncertainty_map,
                                   species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    return

def Convert_mahalanobis_distance_map_to_uncertainty(species,version,special_name,
                                                    Obs_version,nearby_sites_number,YYYY_list:np.int32,MM_list:np.int32):
    '''
    Convert Mahalanobis distance map to uncertainty map for given species and years/months.
    '''
    Mahalanobis_distance_bin_centers,WINTER_LOWESS_values,SPRING_LOWESS_values,SUMMER_LOWESS_values,AUTUMN_LOWESS_values,ALL_LOWESS_values = load_bins_LOWESS_values(species=species,version=version,special_name=special_name,nearby_sites_number=nearby_sites_number)
    
    ## Get Four Seasons and All data, and plot five lines in one figure

    WINTER_MONTHS = ['Jan', 'Feb',  'Dec']
    SPRING_MONTHS = ['Mar', 'Apr', 'May']
    SUMMER_MONTHS = ['Jun','Jul','Aug']
    AUTUMN_MONTHS = ['Sep','Oct','Nov']
    ALL_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # MONTHS = ['Jun','Jul','Aug']
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    
    for YYYY in YYYY_list:
        for MM in MM_list:
            print(f"Calculating rRMSE uncertainty map for species: {species}, version: {version}, date: {YYYY}-{MONTH[MM]}, obs_version: {Obs_version}, nearby_sites_number: {nearby_sites_number}")
            mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                             obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
            mahalanobis_distance_map = np.log(mahalanobis_distance_map + 1)
            map_uncertainty = np.zeros(mahalanobis_distance_map.shape,dtype=np.float64)
            
            if MM in [0,1,11]: # Winter
                LOWESS_values = WINTER_LOWESS_values
            elif MM in [2,3,4]: # Spring
                LOWESS_values = SPRING_LOWESS_values
            elif MM in [5,6,7]: # Summer
                LOWESS_values = SUMMER_LOWESS_values
            elif MM in [8,9,10]: # Autumn
                LOWESS_values = AUTUMN_LOWESS_values
            elif MM == 12:
                LOWESS_values = ALL_LOWESS_values
            valid_LOWESS_index = np.where(~np.isnan(LOWESS_values))[0]
            LOWESS_values = LOWESS_values[valid_LOWESS_index]
            temp_Mahalanobis_distance_bin_centers = [Mahalanobis_distance_bin_centers[i] for i in valid_LOWESS_index]
            for iradius in range(len(temp_Mahalanobis_distance_bin_centers)-1):
                d_left  = temp_Mahalanobis_distance_bin_centers[iradius]
                d_right = temp_Mahalanobis_distance_bin_centers[iradius+1]
                rRMSE_left  = LOWESS_values[iradius]
                rRMSE_right = LOWESS_values[iradius+1]
                pixels_index = np.where((mahalanobis_distance_map >= d_left) & (mahalanobis_distance_map < d_right))
                print('d_left: {}, d_right: {}, rRMSE_left: {}, rRMSE_right: {}'.format(d_left,d_right,rRMSE_left,rRMSE_right))
                slope = (rRMSE_right - rRMSE_left) / (d_right - d_left)
                map_uncertainty[pixels_index] = (mahalanobis_distance_map[pixels_index]-d_left)*slope + rRMSE_left

            d_left  = temp_Mahalanobis_distance_bin_centers[0]
            d_right = temp_Mahalanobis_distance_bin_centers[-1]
            rRMSE_left  = LOWESS_values[0]
            rRMSE_right = LOWESS_values[-1]
            outrange_pixels_index = np.where(mahalanobis_distance_map >= temp_Mahalanobis_distance_bin_centers[-1])
            
            mask_low = np.where(mahalanobis_distance_map < d_left)
            
            if LOWESS_values[0] <= LOWESS_values[1]:
                slope = abs(LOWESS_values[1]-LOWESS_values[0])/(temp_Mahalanobis_distance_bin_centers[1]-temp_Mahalanobis_distance_bin_centers[0])
            else:
                slope = 0.05
            map_uncertainty[mask_low] = slope*(mahalanobis_distance_map[mask_low]-temp_Mahalanobis_distance_bin_centers[0])+LOWESS_values[0]

            if LOWESS_values[-1] >= LOWESS_values[-2]:
                slope = abs(LOWESS_values[-1]-LOWESS_values[-2])/(temp_Mahalanobis_distance_bin_centers[-1]-temp_Mahalanobis_distance_bin_centers[-2])
                map_uncertainty[outrange_pixels_index] = slope*(mahalanobis_distance_map[outrange_pixels_index]-temp_Mahalanobis_distance_bin_centers[-1])+LOWESS_values[-1]
            else:
                #slope,intercept = m, b = np.polyfit(Mahalanobis_distance_bin_centers,LOWESS_values,1)#abs(BLCO_rRMSE_LOWESS_values[-1]-BLCO_rRMSE_LOWESS_values[0])/(distances_bins_array[-1]-distances_bins_array[0])
                map_uncertainty[outrange_pixels_index] = 0.1*(mahalanobis_distance_map[outrange_pixels_index]-temp_Mahalanobis_distance_bin_centers[-1])+LOWESS_values[-1]
                #map_uncertainty[outrange_pixels_index] = rRMSE_right #(map_distances[outrange_pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
            save_rRMSE_map(rRMSE_uncertainty_map=map_uncertainty,species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    return

def Get_nearby_sites_indices_map(species,version,nearby_sites_number,YYYY:np.int32,MM:np.int32):
    obs_data, obs_lat, obs_lon = load_RawObservation(species)
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    original_nearby_sites_number = nearby_sites_number
    if MM != 12:
        temp_obs_data = obs_data[:,(YYYY-2005)*12 + MM]
        nonan_index = np.where(~np.isnan(temp_obs_data))
        obs_index = np.arange(obs_lat.shape[0])
        if obs_lat[nonan_index].shape[0] < nearby_sites_number:
            nearby_sites_number = obs_lat[nonan_index].shape[0]
            print('The nearby sites number is set to {} due to the limited valid observations.'.format(nearby_sites_number))
        GeoLAT_MAP, GeoLON_MAP = load_GeoLatLon_Map()
        nearby_sites_training_data_indices = np.zeros((GeoLAT_MAP.shape[0],GeoLAT_MAP.shape[1],nearby_sites_number), dtype=int)
        landtype = get_landtype(extent=[-59.995,69.995,-179.995,179.995])
        for ix in range(GeoLAT_MAP.shape[0]):
            land_index = np.where(landtype[ix,:] != 0)
            print('It is procceding ' + str(np.round(100*(ix/GeoLAT_MAP.shape[0]),2))+'%.' )
            if len(land_index[0]) == 0:
                print('No lands.')
                None
            else:
                idx = neighbors_haversine_indices(
                            obs_lat[nonan_index], obs_lon[nonan_index], GeoLAT_MAP[ix,land_index[0]], GeoLON_MAP[ix,land_index[0]], nearby_sites_number
                        )
                idx = np.array(idx)
                original_idx = obs_index[nonan_index][idx]
                nearby_sites_training_data_indices[ix,land_index[0],:] = original_idx
                
        save_pixel_nearby_sites_index_map(nearby_sites_training_data_indices=nearby_sites_training_data_indices,
                                        species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=original_nearby_sites_number)
    
    return

def Get_local_reference_for_channels(channel_lists,species,version,
                                   Obs_version,nearby_sites_number,YYYY:np.int32,MM:np.int32):
    if MM != 12:
        MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
        width_nc, height_nc, sites_number, start_YYYY_training, _ = load_TrainingVariables(channel_lists)
        RawObs_training_data = load_RawObs_training_data(channel_lists=channel_lists)

        nearby_sites_training_data_indices = load_pixels_nearest_sites_indices_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                                                obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
        H, W, K = nearby_sites_training_data_indices.shape

        GeoLAT_MAP, GeoLON_MAP = load_GeoLatLon_Map()
        local_reference_for_channels_map = {}
        for channel in channel_lists:
            print('Processing {} - {} - {} local reference for channel: {}'.format(species,YYYY,MONTH[MM],channel))
            temp_training_data = RawObs_training_data[channel][((YYYY-2005)*12 + MM)*sites_number : ((YYYY-2005)*12 + MM +1)*sites_number]
            temp_training_data_reference_map = np.zeros((GeoLAT_MAP.shape[0],GeoLAT_MAP.shape[1]), dtype=np.float32)
            local_reference_for_channels_map[channel] = np.zeros((GeoLAT_MAP.shape[0],GeoLAT_MAP.shape[1]), dtype=np.float32)
            for ik in range(K):
                temp_training_data_reference_map += temp_training_data[nearby_sites_training_data_indices[:, :, ik]]
            temp_training_data_reference_map = temp_training_data_reference_map / K
            local_reference_for_channels_map[channel] = temp_training_data_reference_map
        save_local_reference_map(local_reference_for_channels_map=local_reference_for_channels_map,
                                species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    return

 
def Calculate_Mahalanobis_distance(channel_lists,species,version,
                                   Obs_version,nearby_sites_number,YYYY:np.int32,MM:np.int32,
                                   longterm_average:bool=False):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']

    if MM != 12:
        RawObs_training_data = load_RawObs_training_data(channel_lists=channel_lists)
        Mahalanobis_distance_data = {}
        # Calculate Mahalanobis distance for each channel
        total_channels_training_site_data_list = []
        for channel in channel_lists:
            total_channels_training_site_data_list.append(RawObs_training_data[channel])
        total_channels_training_site_data_list = np.stack(total_channels_training_site_data_list, axis=1)
        covariance_matrix = calculate_covariance_matrix(total_channels_training_site_data_list)
        inverted_covariance_matrix = invert_matrix(covariance_matrix)
        
        local_reference_for_channels_map = load_local_reference_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                                    obs_version=Obs_version,nearby_sites_number=nearby_sites_number)

        Training_Map_data = np.zeros((len(channel_lists),local_reference_for_channels_map[channel_lists[0]].shape[0],local_reference_for_channels_map[channel_lists[0]].shape[1]), dtype=np.float32)
        Total_local_reference_for_channels_map = np.zeros((len(channel_lists),local_reference_for_channels_map[channel_lists[0]].shape[0],local_reference_for_channels_map[channel_lists[0]].shape[1]), dtype=np.float32)
        input_file_dic = inputfiles_table(YYYY=YYYY,MM=MONTH[MM])
        for ichannel, channel in enumerate(channel_lists):
            print('Processing {} - {} - {} Mahalanobis distance for channel: {}'.format(species,YYYY,MONTH[MM],channel))
            infile = input_file_dic[channel]
            temp_mapdata = load_mapdata(infile)
            Training_Map_data[ichannel, :, :] = temp_mapdata
            Total_local_reference_for_channels_map[ichannel, :, :] = local_reference_for_channels_map[channel]
        Mahalanobis_distance_data_map = calculate_mahalanobis_distance(data=Training_Map_data,
                                                                    mean_vector=Total_local_reference_for_channels_map,
                                                                    inverted_covariance_matrix=inverted_covariance_matrix)

        save_mahalanobis_distance_map(mahalanobis_distance_map=Mahalanobis_distance_data_map,
                                    species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    elif MM == 12:
        local_reference_for_channels_map = load_local_reference_map(species=species,version=version,YYYY=YYYY,MM=MONTH[0],
                                                                    obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
        Mahalanobis_distance_data = np.zeros((local_reference_for_channels_map[channel_lists[0]].shape[0],local_reference_for_channels_map[channel_lists[0]].shape[1]), dtype=np.float32)
        for imonth in range(12):
            temp_Mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[imonth],
                                                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
            Mahalanobis_distance_data += temp_Mahalanobis_distance_map
        Mahalanobis_distance_data = Mahalanobis_distance_data / 12.0
        save_mahalanobis_distance_map(mahalanobis_distance_map=Mahalanobis_distance_data,
                                    species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)

        
                
    return Mahalanobis_distance_data

def get_longterm_average_mahalanobis_distance_map(species,version,Obs_version,nearby_sites_number,YYYY_list:np.int32,MM:np.int32):
    '''
    Get longterm average Mahalanobis distance map for given species and years/months.
    '''
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Calculating longterm average Mahalanobis distance map for species: {species}, version: {version}, month: {MONTH[MM]}, obs_version: {Obs_version}, nearby_sites_number: {nearby_sites_number}")
        longterm_average_mahalanobis_distance_map = None
        count = 0
        for YYYY in YYYY_list:
            temp_mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
            if longterm_average_mahalanobis_distance_map is None:
                longterm_average_mahalanobis_distance_map = np.zeros_like(temp_mahalanobis_distance_map)
            longterm_average_mahalanobis_distance_map += temp_mahalanobis_distance_map
            count += 1
        longterm_average_mahalanobis_distance_map = longterm_average_mahalanobis_distance_map / count
        save_mahalanobis_distance_map(mahalanobis_distance_map=longterm_average_mahalanobis_distance_map,
                                    species=species,version=version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    elif MM == 12:
        print(f"Calculating longterm average Mahalanobis distance map for species: {species}, version: {version}, month: All Months, obs_version: {Obs_version}, nearby_sites_number: {nearby_sites_number}")
        longterm_average_mahalanobis_distance_map = None
        count = 0
        for YYYY in YYYY_list:
            for imonth in range(12):
                temp_mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[imonth],
                                                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
                if longterm_average_mahalanobis_distance_map is None:
                    longterm_average_mahalanobis_distance_map = np.zeros_like(temp_mahalanobis_distance_map)
                longterm_average_mahalanobis_distance_map += temp_mahalanobis_distance_map
                count += 1
        longterm_average_mahalanobis_distance_map = longterm_average_mahalanobis_distance_map / count
        save_mahalanobis_distance_map(mahalanobis_distance_map=longterm_average_mahalanobis_distance_map,
                                    species=species,version=version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    return