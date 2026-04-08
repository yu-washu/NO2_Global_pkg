from map_uncertainty_func.Assemble_func import get_longterm_average_mahalanobis_distance_map,Get_longterm_average_absolute_uncertainty_map,Get_absolute_uncertainty_map,Convert_mahalanobis_distance_map_to_uncertainty,Calculate_Mahalanobis_distance,Get_nearby_sites_indices_map, Get_local_reference_for_channels
from data_func.utils import Get_typeName, Obs_version
from visualization_pkg.Assemble_func import plot_longterm_average_absolute_uncertainty_map,plot_longterm_average_mahalanobis_distance_map,plot_longterm_average_map_estimation_data,plot_mahalanobis_distance_map, plot_rRMSE_uncertainty_map,plot_absolute_uncertainty_map,plot_map_estimation_data
import argparse
import numpy as np

desire_year_list = [2023]

### Main functions Switches

Get_the_nearby_sites_indices_map_Switch = False # Run once to get the nearby sites indices map, then set it to False. Only set it True if you change to a new Observation version or nearby sites number.
Get_local_reference_map_Switch = False # Run once to get the local reference map, then set it to False. Only set it True if you change to a new Observation version, include new variables, or nearby sites number.

Get_mahalanobis_distance_map_Switch = False # Run once to get the Mahalanobis distance map, then set it to False. Only set it True if you change to a new Observation version, include new variables, or nearby sites number.
Plot_mahalanobis_distance_map_switch = True # After getting the Mahalanobis distance map, set it to True to plot the figures.

Get_uncertainty_rRMSE_map_Switch = False # Before getting the map, run the file mahalanobis_distance_uncertainty_test.ipynb to get the relationship between Mahalanobis distance and rRMSE.
Plot_rRMSE_uncertainty_map_Switch = False # After getting the rRMSE uncertainty map, set it to True to plot the figures.

Get_absolute_uncertainty_map_Switch = False# Multiple the rRMSE uncertainty map with the estimation map to get the absolute uncertainty map.
Plot_absolute_uncertainty_map_Switch = False # After getting the absolute uncertainty map, set it to True to plot the figures.

Plot_Map_estimation_Switch = False # Plot the map estimation data.

plot_months = [12,0,6]
#[0,1,2,3,4,5,6,7,8,9,10,11,12]  # The months to plot the rRMSE uncertainty map. 12 is annual.

#################################################################################################################
#### Longterm average maps Switches and parameters
#################################################################################################################
# must get all months prior running longterm average maps

get_longterm_average_mahalanobis_distance_map_Switch = False # Get longterm average Mahalanobis distance map
get_longterm_average_absolute_uncertainty_map_Switch = False # Get longterm average absolute uncertainty map

plot_longterm_average_mahalanobis_distance_map_Switch = False # Plot longterm average Mahalanobis distance map
plot_longterm_average_absolute_uncertainty_map_Switch = False # Plot longterm average absolute uncertainty map
plot_longterm_average_map_estimation_data_Switch = False # Plot longterm average map estimation data
longterm_months = [12] # The months to plot the longterm average maps.

species = 'NO2'
version = 'v2'
special_name = '_cf_v6_filtered'
typeName = Get_typeName(bias=False, normalize_bias=False, normalize_species=True, absolute_species=False, log_species=False, species=species)
startyear = 2023
endyear = 2023
nchannel = 26
width = 5
height = 5
buffer_radius_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
BLCO_kfold = 10
BLCO_seeds_number = 5
channel_lists = ['GeoNO2', 'GCHP_NO2','Population',
                'NO_emi','Total_DM', 
                'NDVI','ISA',
                'TSW', 'USTAR', 'V10M', 'U10M', 'T2M','RH','PBLH','TP','PS', 
                'elevation', 'lat','lon', 
                'log_major_roads','log_minor_roads_new', 
                'forests_density', 'shrublands_distance', 'croplands_distance', 'urban_builtup_lands_buffer-6500', 'water_bodies_distance'
                ]

local_nearby_sites_number = 30
vmin_list = [0, 0, 0, 0, 0]
vmax_list = [5, 5, 10, 5, 5]

absolute_uncertainty_vmin_list = [0,0,0,0,0]
absolute_uncertainty_vmax_list = [5,5,5,5,5]
    
#############################################################################################################
### Get local reference map
#############################################################################################################
if Get_the_nearby_sites_indices_map_Switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            Get_nearby_sites_indices_map(species=species,version=version,nearby_sites_number=local_nearby_sites_number,
                                            YYYY=YYYY,MM=MM)
if Get_local_reference_map_Switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            Get_local_reference_for_channels(channel_lists=channel_lists,
                                            species=species,version=version,
                                            Obs_version=Obs_version,
                                            nearby_sites_number=local_nearby_sites_number,
                                            YYYY=YYYY,MM=MM)
#############################################################################################################
### Get and plot Mahalanobis distance map
#############################################################################################################
if Get_mahalanobis_distance_map_Switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            Calculate_Mahalanobis_distance(channel_lists=channel_lists,
                                            species=species,version=version,
                                            Obs_version=Obs_version,
                                            nearby_sites_number=local_nearby_sites_number,
                                            YYYY=YYYY,MM=MM)
if Plot_mahalanobis_distance_map_switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            plot_mahalanobis_distance_map(species=species,version=version,
                                            YYYY=YYYY,MM=MM,
                                            obs_version=Obs_version,
                                            nearby_sites_number=local_nearby_sites_number)
            
#############################################################################################################
### Get and plot rRMSE uncertainty map
#############################################################################################################
            
if Get_uncertainty_rRMSE_map_Switch:
    Convert_mahalanobis_distance_map_to_uncertainty(
                                            species=species,version=version,special_name=special_name,
                                            Obs_version=Obs_version,
                                            nearby_sites_number=local_nearby_sites_number,
                                            YYYY_list=desire_year_list,MM_list=plot_months)


if Plot_rRMSE_uncertainty_map_Switch:
    vmin_list = [0.30 , 0.35, 0.30, 0.30]
    vmax_list = [0.50 , 0.50, 0.50, 0.50]
    for YYYY in desire_year_list:
        for MM in plot_months:
            if MM in [0,1,11]:
                vmin = vmin_list[0]
                vmax = vmax_list[0]
            elif MM in [2,3,4]:
                vmin = vmin_list[1]
                vmax = vmax_list[1]
            elif MM in [5,6,7]:
                vmin = vmin_list[2]
                vmax = vmax_list[2]
            else:
                vmin = vmin_list[3]
                vmax = vmax_list[3]
            plot_rRMSE_uncertainty_map(species=species,version=version,
                                        YYYY=YYYY,MM=MM,
                                        obs_version=Obs_version,
                                        nearby_sites_number=local_nearby_sites_number,
                                        vmin=vmin,vmax=vmax)

#############################################################################################################
### Get and plot absolute uncertainty map
#############################################################################################################
if Get_absolute_uncertainty_map_Switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            Get_absolute_uncertainty_map(species=species,version=version,special_name=special_name,
                                        obs_version=Obs_version,
                                        nearby_sites_number=local_nearby_sites_number,
                                        YYYY=YYYY,MM=MM,map_estimation_special_name=special_name,
                                        map_estimation_version=version)
if Plot_absolute_uncertainty_map_Switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            
            if MM in [0,1,11]:
                vmin = absolute_uncertainty_vmin_list[0]
                vmax = absolute_uncertainty_vmax_list[0]
            elif MM in [2,3,4]:
                vmin = absolute_uncertainty_vmin_list[1]
                vmax = absolute_uncertainty_vmax_list[1]
            elif MM in [5,6,7]:
                vmin = absolute_uncertainty_vmin_list[2]
                vmax = absolute_uncertainty_vmax_list[2]
            elif MM in [8,9,10]:
                vmin = absolute_uncertainty_vmin_list[3]
                vmax = absolute_uncertainty_vmax_list[3]
            else:
                vmin = absolute_uncertainty_vmin_list[4]
                vmax = absolute_uncertainty_vmax_list[4]
            plot_absolute_uncertainty_map(species=species,map_estimation_version=version,map_estimation_special_name=special_name,
                                            YYYY=YYYY,MM=MM,
                                            obs_version=Obs_version,
                                            nearby_sites_number=local_nearby_sites_number,
                                            vmin=vmin,vmax=vmax)
############################################################################################################
### Plot map estimation data
############################################################################################################
if Plot_Map_estimation_Switch:
    for YYYY in desire_year_list:
        for MM in plot_months:
            plot_map_estimation_data(species=species,map_estimation_version=version,YYYY=YYYY,MM=MM,map_estimation_special_name=special_name)
        

##################################################################################################################
### Longterm average maps
##################################################################################################################
if get_longterm_average_mahalanobis_distance_map_Switch:
    for MM in longterm_months:
        get_longterm_average_mahalanobis_distance_map(species=species,version=version,
                                                Obs_version=Obs_version,
                                                nearby_sites_number=local_nearby_sites_number,
                                                YYYY_list=desire_year_list,MM=MM)
if get_longterm_average_absolute_uncertainty_map_Switch:
    for MM in longterm_months:
        Get_longterm_average_absolute_uncertainty_map(species=species,version=version,special_name=special_name,
                                                obs_version=Obs_version,
                                                nearby_sites_number=local_nearby_sites_number,
                                                YYYY_list=desire_year_list,MM=MM,
                                                map_estimation_special_name=special_name,
                                                map_estimation_version=version)
if plot_longterm_average_mahalanobis_distance_map_Switch:
    for MM in longterm_months:
        plot_longterm_average_mahalanobis_distance_map(species=species,version=version,
                                                YYYY_list=desire_year_list,MM=MM,
                                                obs_version=Obs_version,
                                                nearby_sites_number=local_nearby_sites_number)
        
if plot_longterm_average_absolute_uncertainty_map_Switch:
    for MM in longterm_months:
        if MM in [0,1,11]:
            vmin = absolute_uncertainty_vmin_list[0]
            vmax = absolute_uncertainty_vmax_list[0]
        elif MM in [2,3,4]:
            vmin = absolute_uncertainty_vmin_list[1]
            vmax = absolute_uncertainty_vmax_list[1]
        elif MM in [5,6,7]:
            vmin = absolute_uncertainty_vmin_list[2]
            vmax = absolute_uncertainty_vmax_list[2]
        elif MM in [8,9,10]:
            vmin = absolute_uncertainty_vmin_list[3]
            vmax = absolute_uncertainty_vmax_list[3]
        else:
            vmin = absolute_uncertainty_vmin_list[4]
            vmax = absolute_uncertainty_vmax_list[4]
        plot_longterm_average_absolute_uncertainty_map(species=species,map_estimation_version=version,map_estimation_special_name=special_name,
                                                YYYY_list=desire_year_list,MM=MM,
                                                obs_version=Obs_version,
                                                nearby_sites_number=local_nearby_sites_number,
                                                vmin=vmin,vmax=vmax
                                        )
if plot_longterm_average_map_estimation_data_Switch:
    for MM in longterm_months:
        plot_longterm_average_map_estimation_data(species=species,map_estimation_version=version,
                                                YYYY_list=desire_year_list,MM=MM,
                                                map_estimation_special_name=special_name)