#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200GB
#SBATCH --job-name="asia"
#SBATCH --output=job_output/slurm-%j.txt
#SBATCH --container-image=1yuyan/python-gfortran:latest
#SBATCH --container-mounts=/storage1/fs1/rvmartin2/Active/yany1:/my-projects2
#SBATCH --container-workdir=/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/Data_processing/Get_OpenStreetMap_Input

ulimit -s unlimited

mkdir -p job_output

declare -A Region_list
#afghanistan anhui armenia azerbaijan bangladesh beijing bhutan cambodia central-fed-district central-zone \
#chongqing chubu chugoku crimean-fed-district eastern-zone east-timor far-eastern-fed-district fujian gansu \
#gcc-states
Region_list["asia"]="guangdong guangxi guizhou hainan hebei heilongjiang henan hokkaido hong-kong hubei hunan \
                    inner-mongolia iran iraq israel-and-palestine java jiangsu jiangxi jilin jordan kalimantan kaliningrad \
                    kansai kanto kazakhstan kyrgyzstan kyushu laos lebanon liaoning macau malaysia-singapore-brunei maldives \
                    maluku mongolia myanmar nepal ningxia north-caucasus-fed-district north-eastern-zone northern-zone north-korea \
                    northwestern-fed-district nusa-tenggara pakistan papua philippines qinghai shaanxi shandong shanghai shanxi \
                    shikoku siberian-fed-district sichuan southern-zone south-fed-district south-korea sri-lanka sulawesi sumatra \
                    syria taiwan tajikistan thailand tianjin tibet tohoku turkmenistan ural-fed-district uzbekistan vietnam \
                    volga-fed-district western-zone xinjiang yemen yunnan zhejiang"

entries=(primary secondary trunk motorway tertiary residential unclassified)

for region in ${Region_list["asia"]}
do
    for entry in "${entries[@]}"
    do
    echo "Processing asia, region: $region, entry: $entry"
    python main.py \
        --Continent "asia" \
        --regions_list "$region" \
        --Entry_List_forRegional_RoadDensityMap "$entry"
    done
done

# python main.py

# for entry in "${entries[@]}"
# do
#     echo "Processing entry: $entry"
#     python3 main_nearest_roaddensity_pixels.py -R "$entry"
# done

# # Aggregate the road density map and the nearest road density pixels for each pixel.
# python3 aggreagte_road_variables.py