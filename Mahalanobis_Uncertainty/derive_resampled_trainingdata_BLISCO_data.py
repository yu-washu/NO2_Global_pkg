import numpy as np
import os
from data_func.Assemble_func import derive_corresponding_training_data_BLISCO_data
from data_func.utils import Get_typeName

species = 'NO2'
version = 'v2'
typeName = Get_typeName(bias=False, normalize_bias=False, normalize_species=True, absolute_species=False, log_species=False, species=species)
startyear = 2023
endyear = 2023
nchannel = 26
special_name = '_cf_v6_filtered'
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
desire_year_list = ['2023']
derive_corresponding_training_data_BLISCO_data(typeName=typeName, species=species, version=version,
                                                startyear=startyear, endyear=endyear, nchannel=nchannel,
                                                special_name=special_name, width=width, height=height,
                                                buffer_radius_list=buffer_radius_list, BLCO_kfold=BLCO_kfold,
                                                BLCO_seeds_number=BLCO_seeds_number, channel_lists=channel_lists,
                                                desire_year_list=desire_year_list)