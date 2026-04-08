#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200GB
#SBATCH --job-name="france"
#SBATCH --output=job_output/slurm-%j.txt
#SBATCH --container-image=1yuyan/python-gfortran:latest
#SBATCH --container-mounts=/storage1/fs1/rvmartin2/Active/yany1:/my-projects2
#SBATCH --container-workdir=/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/Data_processing/Get_OpenStreetMap_Input

ulimit -s unlimited

mkdir -p job_output
declare -A Region_list

#alsace
Region_list["france"]="corse midi-pyrenees aquitaine franche-comte nord-pas-de-calais auvergne guyane pays-de-la-loire \
    basse-normandie haute-normandie picardie bourgogne ile-de-france poitou-charentes bretagne languedoc-roussillon \
    provence-alpes-cote-d-azur centre limousin rhone-alpes champagne-ardenne lorraine"

entries=(primary secondary trunk motorway tertiary residential unclassified)

for region in ${Region_list["france"]}
do
    for entry in "${entries[@]}"
    do
        echo "Processing france region: $region, entry: $entry"
        python main.py \
            --Continent "france" \
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