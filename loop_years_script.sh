#!/bin/bash

# Define the range for the loop
start_year=2021
end_year=2021

# Job script file
job_script="run_gpu.bsub"

# Loop through the years
for year in $(seq $start_year $end_year); do
    # Update beginyears_endyears and Estimation_years dynamically
    beginyears_endyears="[$year]"
    Estimation_years="[[$year]]"
    Wait_time=

    # Create a temporary modified script
    modified_script="modified_job_script_${year}.bsub"
    cp $job_script $modified_script

    # Use sed to replace variables in the script
    sed -i "s/^beginyears_endyears=.*/beginyears_endyears=${beginyears_endyears}/" $modified_script
    sed -i "s/^Estimation_years=.*/Estimation_years=${Estimation_years}/" $modified_script
    sed -i "s/^#BSUB -J .*/#BSUB -J \"Estiamtion 1.8.0 PM ${year}\"/" $modified_script

    # Update the pause_time calculation
    sed -i "s/^pause_time=\$((RANDOM % 50 .*/pause_time=\$((RANDOM % 10 + (${year} - ${start_year}) * 150))/" $modified_script

    # Submit the modified script using bsub
    echo "Submitting job for year $year..."
    bsub < $modified_script

    # Optional: Clean up temporary script after submission
    
    # Pause for 2 seconds before the next submission
    echo "Waiting for 2 seconds before the next job..."
    sleep 2

    rm $modified_script
done
