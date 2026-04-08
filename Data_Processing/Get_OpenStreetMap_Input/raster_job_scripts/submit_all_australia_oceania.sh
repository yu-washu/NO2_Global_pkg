#!/bin/bash

# Create output directory for job logs
mkdir -p job_output

# Define all regions in different continents
declare -A Region_list

# argentina bolivia chile colombia guyana paraguay suriname uruguay venezuela vanuatu fiji ile-de-clipperton pitcairn-islands tokelau niue palau wallis-et-futuna marshall-islands kiribati papua-new-guinea

Region_list["australia-oceania"]="samoa"

# Road entry types
entries=(motorway primary secondary trunk tertiary residential unclassified)

# Directory for temporary job scripts
mkdir -p temp_job_scripts

# Loop through continents and regions to create and submit individual job scripts
for region in ${Region_list["australia-oceania"]}
do
    # Create a job script for this region
    job_script="temp_job_scripts/australia-oceania_${region}.bsub"
    
    cat > "$job_script" << EOL
#!/bin/bash
#BSUB -q rvmartin
#BSUB -n 2
#BSUB -W 499:00
#BSUB -R "rusage[mem=200GB] span[hosts=1] select[port8543=1]"
#BSUB -a 'docker(1yuyan/python-gfortran:latest)'
#BSUB -J "${region}"
#BSUB -N
#BSUB -G compute-rvmartin
#BSUB -g /yany1/jobs15
#BSUB -u yany1@wustl.edu
#BSUB -o job_output/australia-oceania_${region}-%J-output.txt

. /opt/conda/bin/activate
cd /my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/Data_processing/Get_OpenStreetMap_Input/

entries=(motorway primary secondary trunk tertiary residential unclassified)

for entry in "\${entries[@]}"
do
    echo "Processing ${region}, entry: \$entry"
    python main.py \\
        --Continent "australia-oceania" \\
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