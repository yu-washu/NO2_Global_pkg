#!/bin/bash

# Create output directory for job logs
mkdir -p job_output

# Define all regions in different continents
declare -A Region_list

# Asia regions
#Region_list["asia"]="eastern-zone central-zone heilongjiang hokkaido inner-mongolia iran kalimantan kazakhstan malaysia-singapore-brunei mongolia myanmar northern-zone northwestern-fed-district pakistan philippines qinghai siberian-fed-district sichuan southern-zone south-fed-district sumatra thailand tibet turkmenistan ural-fed-district uzbekistan vietnam volga-fed-district western-zone xinjiang"
Region_list["canada"]="quebec saskatchewan manitoba nunavut yukon new-brunswick ontario prince-edward-island"
# Road entry types
entries=(motorway primary secondary trunk tertiary residential unclassified)

# Directory for temporary job scripts
mkdir -p temp_job_scripts

# Loop through continents and regions to create and submit individual job scripts
for region in ${Region_list["canada"]}
do
    # Create a job script for this region
    job_script="temp_job_scripts/canada_${region}.bsub"
    
    cat > "$job_script" << EOL
#!/bin/bash
#BSUB -q rvmartin
#BSUB -n 1
#BSUB -W 499:00
#BSUB -R "rusage[mem=30GB] span[hosts=1] select[port8543=1]"
#BSUB -a 'docker(1yuyan/python-gfortran:latest)'
#BSUB -J "${region}"
#BSUB -N
#BSUB -G compute-rvmartin
#BSUB -g /yany1/jobs30
#BSUB -u yany1@wustl.edu
#BSUB -o job_output/canada_${region}-%J-output.txt

. /opt/conda/bin/activate
cd /my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/Data_processing/Get_OpenStreetMap_Input/

entries=(motorway primary secondary trunk tertiary residential unclassified)

for entry in "\${entries[@]}"
do
    echo "Processing ${region}, entry: \$entry"
    python main.py \\
        --Continent "canada" \\
        --regions_list "${region}" \\
        --Entry_List_forRegional_RoadDensityMap "\$entry"
done
EOL

        # Make the job script executable
        chmod +x "$job_script"
        
        # Submit the job
        echo "Submitting job for ${region}"
        bsub < "$job_script"
done

echo "All jobs submitted successfully!"
