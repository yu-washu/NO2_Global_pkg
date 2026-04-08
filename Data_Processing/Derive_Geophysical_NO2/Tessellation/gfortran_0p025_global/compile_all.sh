#!/bin/bash

months=(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec)

for month in "${months[@]}"; do
    echo "Compiling for $month..."
    gfortran -mcmodel=large tessellate_variablenpix.f90 tesselation_software.f -o tessellate_variablenpix_"$month" &
done

wait
echo "All compilations finished!"

