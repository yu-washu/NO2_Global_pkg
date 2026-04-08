#!/bin/bash

# Define the range for the loop
start_radius=24
end_radius=26
radius_bin=1

# Job script file
job_script="run_cpu.bsub"

# Loop through the years
for (( radius=$start_radius; radius<=$end_radius; radius+=$radius_bin )); do
    # Update beginyears_endyears and Estimation_years dynamically
    Buffer_size="[$radius]"

    # Create a temporary modified script
    modified_script="modified_job_script_${Buffer_size}.bsub"
    cp $job_script $modified_script

    # Use sed to replace variables in the script
    sed -i "s/^Buffer_size=.*/Buffer_size=${Buffer_size}/" $modified_script
    sed -i "s/^#BSUB -J .*/#BSUB -J \"${radius}\"/" $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % 50 .*/pause_time=\$((RANDOM % 30 + (${radius} - ${start_radius}) * 120))/" $modified_script

    # Submit the modified script using bsub
    echo "Submitting job for radius $radius..."
    bsub < $modified_script
    
    # Pause for 90 seconds before the next submission
    echo "Waiting for 30 seconds before the next job..."
    sleep 30

    rm $modified_script
done