#!/usr/bin/env bash
set -x
set -euo pipefail
ulimit -c 0                  # coredumpsize
ulimit -u 50000              # maxproc

# —————————————————————————————————————————————————————————————
# 1) Auto-detect how many 72/64-core exec nodes are up
# —————————————————————————————————————————————————————————————
read N72 N64 <<< $(
  bhosts | awk '
    NR>1 && $2=="ok" && $1~/^compute1-exec/ {
      if ($4==72)    n72++;
      else if ($4==64) n64++;
    }
    END { print n72+0, n64+0 }
  ')
(( N72+N64 )) || { echo "No exec nodes!" >&2; exit 1; }
echo "Found $N72×72-core nodes, $N64×64-core nodes"

# —————————————————————————————————————————————————————————————
# 2) Compute total free cores (reserve ≥16 per node)
# —————————————————————————————————————————————————————————————
PTILE=48             # leave 16 cores free on both 64-core & 72-core boxes
TOTAL_FREE=$(( PTILE*(N72+N64) ))
echo "→ Will allow up to $TOTAL_FREE concurrent one-core jobs"

GROUP="/yany1/omi-monthly"
# if it exists, modify; otherwise add
bgmod -L "${TOTAL_FREE}" "${GROUP}" 2>/dev/null || bgadd -L "${TOTAL_FREE}" "${GROUP}"
echo "Using job‐group ${GROUP} (limit=${TOTAL_FREE})"

# —————————————————————————————————————————————————————————————
# 3) Configuration for TROPOMI processing
# —————————————————————————————————————————————————————————————
START_YEAR=2005
END_YEAR=2005
MEM=150000           # in MB (≈150 GB)
SCRIPT="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av/omi_KNMI_average.py"
LOGDIR="logs_av"; mkdir -p "$LOGDIR"

# —————————————————————————————————————————————————————————————
# 4) Loop over each year and month, submit one job per month
# —————————————————————————————————————————————————————————————
for year in $(seq $START_YEAR $END_YEAR); do
  for month in {1..1}; do
    MONTH_STR=$(printf "%02d" $month)
    JOB_NAME="Momi${year}${MONTH_STR}"
    
    echo "DEBUG: submitting ${JOB_NAME} (${year}-${MONTH_STR})"
    
    bsub -q rvmartin \
         -J "${JOB_NAME}" \
         -g "$GROUP" \
         -n 1 \
         -W 499:00 \
         -u yany1@wustl.edu -G compute-rvmartin \
         -R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" \
         -R "select[hname!='compute1-exec-16.ris.wustl.edu']" \
         -R "select[hname!='compute1-exec-17.ris.wustl.edu']" \
         -R "select[hname!='compute1-exec-29.ris.wustl.edu']" \
         -R "select[port8543=1]" \
         -R "span[ptile=${PTILE}]" \
         -R "rusage[mem=${MEM}]" \
         -a "docker(1yuyan/netcdf-mpi:latest)" \
         -o "${LOGDIR}/Momi_${year}_${MONTH_STR}.out" \
         bash -lc $'\n'"\
. /opt/conda/bin/activate && \
ulimit -s unlimited && \
/bin/bash && \
cd /my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av && \
echo \"[\$(hostname)] Processing OMI-MINDS monthly average: ${year}-${MONTH_STR}\" && \
echo \"Start time: \$(date)\" && \
echo \"Job ID: \$LSB_JOBID\" && \
python3 -u ${SCRIPT} ${year} --month ${month} && \
echo \"End time: \$(date)\" && \
echo \"Completed: ${year}-${MONTH_STR}\"
"
  done
done

echo "Submitted monthly jobs for years $START_YEAR → $END_YEAR into group $GROUP (limit=$TOTAL_FREE)."
echo "  bjobs -g $GROUP"