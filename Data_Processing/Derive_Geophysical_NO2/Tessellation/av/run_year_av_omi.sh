#!/usr/bin/env bash
set -x
set -euo pipefail
ulimit -c 0                  # coredumpsize
ulimit -u 50000              # maxproc

# —————————————————————————————————————————————————————————————
# Submit yearly TROPOMI averaging jobs
# Run this AFTER all monthly jobs are complete
# —————————————————————————————————————————————————————————————

# Configuration
START_YEAR=2005
END_YEAR=2005
MEM=300000
SCRIPT="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av/omi_KNMI_average.py"
LOGDIR="logs_av"; mkdir -p "$LOGDIR"

GROUP="/yany1/omi-yearly"
# Create a smaller job group for yearly jobs (fewer jobs)
bgmod -L "10" "${GROUP}" 2>/dev/null || bgadd -L "10" "${GROUP}"
echo "Using job‐group ${GROUP} (limit=10)"

# —————————————————————————————————————————————————————————————
# Loop over each year, submit one yearly job per year
# —————————————————————————————————————————————————————————————
for year in $(seq $START_YEAR $END_YEAR); do
  JOB_NAME="OmiYear${year}"
  
  bsub -q rvmartin \
       -J "${JOB_NAME}" \
       -g "$GROUP" \
       -n 1 \
       -W 499:00 \
       -Q "all ~0" \
       -u yany1@wustl.edu -G compute-rvmartin \
       -R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" \
       -R "select[port8543=1]" \
       -R "rusage[mem=${MEM}]" \
       -a "docker(1yuyan/netcdf-mpi:latest)" \
       -o "${LOGDIR}/OmiYear_${year}.out" \
       bash -lc $'\n'"\
. /opt/conda/bin/activate && \
cd /my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av && \
echo \"[\$(hostname)] Processing TROPOMI yearly average: ${year}\"
echo \"Start time: \$(date)\"
echo \"Job ID: \$LSB_JOBID\"
ulimit -s unlimited
python3 ${SCRIPT} ${year} --yearly-only
echo \"End time: \$(date)\"
echo \"Completed yearly average: ${year}\"
"
done

echo "Submitted yearly jobs for years $START_YEAR → $END_YEAR into group $GROUP."
echo ""
echo "These jobs will wait for all monthly jobs to complete before starting."
echo ""
echo "Monitor jobs with:"
echo "  bjobs | grep TropomiYear"
echo "  bjobs -g $GROUP"
echo ""
echo "Check logs:"
echo "  tail -f ${LOGDIR}/TropomiYear_2018.out"