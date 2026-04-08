#!/usr/bin/env bash
set -x
set -euo pipefail

ulimit -c 0                  # coredumpsize
ulimit -u 50000              # maxproc
ulimit -s unlimited

# —————————————————————————————————————————————————————————————
# Configuration
# —————————————————————————————————————————————————————————————
START_YEAR=2005
END_YEAR=2023
MEM=500000          # in MB (≈500 GB)
SCRIPT="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Model_Evaluation_pkg/plot/plot_TrainningData/plot_input_predictors.py"
LOGDIR="logs"; mkdir -p "$LOGDIR"

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
CORES_PER_JOB=2
PTILE=48             # leave 16 cores free on both 64-core & 72-core boxes
TOTAL_FREE=$(( PTILE*(N72+N64) ))
MAX_CONCURRENT=$(( TOTAL_FREE / CORES_PER_JOB ))
echo "→ Will allow up to $TOTAL_FREE cores total"
echo "→ Max concurrent plot jobs (${CORES_PER_JOB} cores each): $MAX_CONCURRENT"

GROUP="/yany1/plot"
# if it exists, modify; otherwise add
bgmod -L "${MAX_CONCURRENT}" "${GROUP}" 2>/dev/null || bgadd -L "${MAX_CONCURRENT}" "${GROUP}"
echo "Using job‐group ${GROUP} (limit=${MAX_CONCURRENT})"

# —————————————————————————————————————————————————————————————
# 3) Loop over each year and submit one job per year
# —————————————————————————————————————————————————————————————

for YEAR in $(seq $START_YEAR $END_YEAR); do
  echo "DEBUG: submitting plot for year ${YEAR}"
  bsub -q rvmartin \
       -J "Plot${YEAR}" \
       -g "$GROUP" \
       -n 2 \
       -W 499:00 \
       -Q "all ~0" \
       -u yany1@wustl.edu -G compute-rvmartin \
       -R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" \
       -R "select[port8543=1]" \
       -R "rusage[mem=${MEM}]" \
       -a "docker(1yuyan/netcdf-mpi:latest)" \
       bash -lc $'\n'"\
. /opt/conda/bin/activate && \
cd /my-projects2/1.project/NO2_DL_global/input_variables && \
exec >\"${LOGDIR}/plot_${YEAR}.log\" 2>&1
echo \"[\$(hostname)] Creating plots for year ${YEAR}\"
echo \"Start time: \$(date)\"
echo \"Job ID: \$LSB_JOBID\"
python3 ${SCRIPT} --years ${YEAR} --data_type both
echo \"End time: \$(date)\"
"
done

echo "Submitted years
 $START_YEAR → $END_YEAR into group $GROUP (limit=$MAX_CONCURRENT)."
