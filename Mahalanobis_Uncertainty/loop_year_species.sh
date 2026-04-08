#!/bin/bash

# Define the tracers and years
#tracers=("PM25" "NO3" "SO4" "NH4" "BC" "OM" "DUST" "SS")
tracers=("NH4")
YEAR=(2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023)
#1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 

# Job script file
job_script="run_cpu.bsub"

# Loop through each tracer and year
for tracer in "${tracers[@]}"; do
    for year in "${YEAR[@]}"; do
        # Create a temporary bsub script
        modified_script="modified_job_script_${tracer}_${year}.bsub"
        cp $job_script $modified_script

        # Modify the python3 main.py line
        sed -i "s/^python3 derive_map_mahalanobis_uncertainty.py  .*/python3 derive_map_mahalanobis_uncertainty.py  --SPECIES_list '$tracer' --desire_year_list $year/" $modified_script
        sed -i "s/^#BSUB -J .*/#BSUB -J \"${tracer}_${year}\"/"  $modified_script
        # Submit the job
        bsub < $modified_script
        # Remove the temporary bsub script
        rm $modified_script
    done
done
