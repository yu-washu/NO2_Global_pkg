#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200GB
#SBATCH --job-name="europe"
#SBATCH --output=job_output/slurm-%j.txt
#SBATCH --container-image=1yuyan/python-gfortran:latest
#SBATCH --container-mounts=/storage1/fs1/rvmartin2/Active/yany1:/my-projects2
#SBATCH --container-workdir=/my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/Data_processing/Get_OpenStreetMap_Input
ulimit -s unlimited

mkdir -p job_output

declare -A Region_list

Region_list["europe"]="albania georgia montenegro andorra austria greece norway azores guernsey-jersey belarus hungary \
                        portugal belgium iceland romania bosnia-herzegovina ireland-and-northern-ireland bulgariaisle-of-man \
                        serbia croatia slovakia cyprus kosovo slovenia czech-republic latvia denmark liechtenstein sweden lithuania \
                        switzerland luxembourg turkey estonia macedonia ukraine finland moldova monaco"

entries=(primary secondary trunk motorway tertiary residential unclassified)

for region in ${Region_list["europe"]}
do
    for entry in "${entries[@]}"
    do
        echo "Processing europe, region: $region, entry: $entry"
        python main.py \
            --Continent "europe" \
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