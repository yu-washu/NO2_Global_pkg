#!/bin/bash

# Define the range for the loop
variables=('GeoNO2'  'GCHP_NO2' 
            'NO_anthro_emi'  'NMVOC_anthro_emi' 
            'Total_DM'
            'major_roads' 'minor_roads' 
            'Urban_Builtup_Lands' 'Crop_Nat_Vege_Mos'  'Water_Bodies' 
            'V10M'  'U10M'  'T2M'  'RH'  'PBLH'  'PRECTOT' 
            'Lat'  'Lon'  'elevation' 
            'Population'
            )


# Job script file
job_script="run_gpu.bsub"

# Print the total number of iterations
total_iterations=${#variables[@]}
echo "Total number of iterations: $total_iterations"

# Loop through the variables
for ((i=0; i<total_iterations; i++)); do
    var=${variables[i]}

    # Print the current iteration number
    echo "Iteration $((i+1)) of $total_iterations: Processing variable $var"
    
    # Create a temporary modified script
    modified_script="modified_job_script_${i}.bsub"
    cp $job_script $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % 50 .*/pause_time=\$((RANDOM % 10 + (${i} * 120)))/" $modified_script
    # Use sed to replace variables in the script (exclusion test)
    sed -i "s/^var=.*/var=${var}/" $modified_script
    #sed -i "s/^Exclude_Variables_Sensitivity_Test_Switch=.*/Exclude_Variables_Sensitivity_Test_Switch=false/" $modified_script
    #sed -i "s/^Exclude_Variables_Sensitivity_Test_Variables=.*/Exclude_Variables_Sensitivity_Test_Variables=[['${var}']]/" $modified_script
    sed -i "s/^#BSUB -J .*/#BSUB -J \"v0.0.9 exclude ${var}\"/" $modified_script

    # Submit the modified script using bsub
    echo "Submitting job for variable $var..."
    bsub < $modified_script

    # Pause for 2 seconds before the next submission
    echo "Waiting for 2 seconds before the next job..."
    sleep 2

    # Clean up temporary script after submission
    rm $modified_script
done