#!/bin/bash

# Create output directory for job logs
mkdir -p job_output

# Define all regions in different continents
declare -A Region_list

#andorra austria greece norway azores guernsey-jersey belarus hungary portugal belgium iceland romania bosnia-herzegovina ireland-and-northern-ireland bulgaria 

Region_list["europe"]="isle-of-man serbia croatia slovakia cyprus kosovo\
                        slovenia czech-republic latvia denmark liechtenstein sweden lithuania switzerland luxembourg turkey estonia\
                        macedonia ukraine finland moldova monaco malta faroe-islands"
# Road entry types
entries=(motorway primary secondary trunk tertiary residential unclassified)

# Directory for temporary job scripts
mkdir -p temp_job_scripts

# Loop through continents and regions to create and submit individual job scripts
for region in ${Region_list["europe"]}
do
    # Create a job script for this region
    job_script="temp_job_scripts/europe_${region}.bsub"
    
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
#BSUB -g /yany1/jobs50
#BSUB -u yany1@wustl.edu
#BSUB -o job_output/europe_${region}-%J-output.txt

. /opt/conda/bin/activate
cd /my-projects2/1.project/NO2_DL_global_2019/NO2_global_pkg/Data_processing/Get_OpenStreetMap_Input/

entries=(motorway primary secondary trunk tertiary residential unclassified)

for entry in "\${entries[@]}"
do
    echo "Processing ${region}, entry: \$entry"
    python main.py \\
        --Continent "europe" \\
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