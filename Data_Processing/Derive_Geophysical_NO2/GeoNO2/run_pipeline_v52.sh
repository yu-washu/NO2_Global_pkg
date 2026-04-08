#!/usr/bin/env bash
set -x
set -euo pipefail
ulimit -c 0
ulimit -u 50000

# ==========================================================================
# GeoNO2 v5.2 Full Pipeline (Cooper-lite column-ratio approach)
#   Step 1: Submit GeoNO2 v5.2 monthly jobs (one per year-month)
#   Step 2: Submit evaluation job (depends on all Step 1 jobs finishing)
#   Step 3: Submit plot job (depends on Step 2 finishing)
# ==========================================================================

# ── Configuration ─────────────────────────────────────────────────────────
START_YEAR=2023
END_YEAR=2023
MEM_GEONO2=320000         # MB for GeoNO2 jobs
MEM_EVAL=16000            # MB for evaluation/plot jobs
PTILE=48

BASEDIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing"
GEONO2_SCRIPT="${BASEDIR}/Derive_Geophysical_NO2/GeoNO2/geono2_v5.2.py"
EVAL_SCRIPT="${BASEDIR}/Model_Evaluation_pkg/compr/compr_geo_gchp_global_v52.py"
PLOT_SCRIPT="${BASEDIR}/Model_Evaluation_pkg/compr/plot_scatter_global_v52.py"
LOGDIR="logs_v52"; mkdir -p "$LOGDIR"

GROUP="/yany1/GeoNO2_v52"

# ── Step 0: Auto-detect nodes & set job-group limit ──────────────────────
read N72 N64 <<< $(
  bhosts | awk '
    NR>1 && $2=="ok" && $1~/^compute1-exec/ {
      if ($4==72)    n72++;
      else if ($4==64) n64++;
    }
    END { print n72+0, n64+0 }
  ')
(( N72+N64 )) || { echo "No exec nodes!" >&2; exit 1; }
TOTAL_FREE=$(( PTILE*(N72+N64) ))
echo "Found $N72×72-core + $N64×64-core nodes → limit=$TOTAL_FREE"

bgmod -L "${TOTAL_FREE}" "${GROUP}" 2>/dev/null || bgadd -L "${TOTAL_FREE}" "${GROUP}"

# ── Step 1: Submit GeoNO2 v5.2 monthly jobs ──────────────────────────────
echo ""
echo "========================================"
echo "  Step 1: Submitting GeoNO2 v5.2 jobs"
echo "========================================"

GEONO2_JOB_NAMES=()

for year in $(seq $START_YEAR $END_YEAR); do
  for month in {1..12}; do
    MONTH_STR=$(printf "%02d" $month)
    JOB_NAME="GeoNO2v52_${year}${MONTH_STR}"
    GEONO2_JOB_NAMES+=("${JOB_NAME}")

    echo "  Submitting ${JOB_NAME} (${year}-${MONTH_STR})"

    bsub -q rvmartin \
         -J "${JOB_NAME}" \
         -g "$GROUP" \
         -n 1 \
         -W 499:00 \
         -u yany1@wustl.edu -G compute-rvmartin \
         -R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" \
         -R "select[port8543=1]" \
         -R "span[ptile=${PTILE}]" \
         -R "rusage[mem=${MEM_GEONO2}]" \
         -a "docker(1yuyan/netcdf-mpi:latest)" \
         -o "${LOGDIR}/GeoNO2v52_${year}_${MONTH_STR}.out" \
         bash -lc $'\n'"\
. /opt/conda/bin/activate && \
/bin/bash && \
cd /my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/GeoNO2 && \
echo \"[\$(hostname)] GeoNO2 v5.2: ${year}-${MONTH_STR}\"
echo \"Start: \$(date)  Job: \$LSB_JOBID\"
ulimit -s unlimited
python3 -u ${GEONO2_SCRIPT} ${year} --month ${month} --no-plot
echo \"End: \$(date)  Completed: ${year}-${MONTH_STR}\"
"
  done
done

echo "  → ${#GEONO2_JOB_NAMES[@]} GeoNO2 jobs submitted"

# ── Build dependency string ──────────────────────────────────────────────
DEP_STR=""
for jn in "${GEONO2_JOB_NAMES[@]}"; do
  if [ -z "$DEP_STR" ]; then
    DEP_STR="done(${jn})"
  else
    DEP_STR="${DEP_STR} && done(${jn})"
  fi
done

# ── Step 2: Submit evaluation job (depends on Step 1) ────────────────────
echo ""
echo "========================================"
echo "  Step 2: Submitting evaluation job"
echo "========================================"

for year in $(seq $START_YEAR $END_YEAR); do
  EVAL_JOB="Eval_v52_${year}"
  echo "  Submitting ${EVAL_JOB} (depends on all GeoNO2 jobs)"

  bsub -q rvmartin \
       -J "${EVAL_JOB}" \
       -w "${DEP_STR}" \
       -n 1 \
       -W 4:00 \
       -u yany1@wustl.edu -G compute-rvmartin \
       -R "rusage[mem=${MEM_EVAL}]" \
       -a "docker(1yuyan/netcdf-mpi:latest)" \
       -o "${LOGDIR}/Eval_v52_${year}.out" \
       bash -lc $'\n'"\
. /opt/conda/bin/activate && \
/bin/bash && \
echo \"[\$(hostname)] Evaluation v5.2: ${year}\"
echo \"Start: \$(date)  Job: \$LSB_JOBID\"
python3 -u ${EVAL_SCRIPT} --year ${year}
echo \"End: \$(date)  Evaluation done: ${year}\"
"

  # ── Step 3: Submit plot job (depends on Step 2) ──────────────────────
  echo ""
  echo "========================================"
  echo "  Step 3: Submitting plot job"
  echo "========================================"

  PLOT_JOB="Plot_v52_${year}"
  echo "  Submitting ${PLOT_JOB} (depends on ${EVAL_JOB})"

  bsub -q rvmartin \
       -J "${PLOT_JOB}" \
       -w "done(${EVAL_JOB})" \
       -n 1 \
       -W 1:00 \
       -u yany1@wustl.edu -G compute-rvmartin \
       -R "rusage[mem=${MEM_EVAL}]" \
       -a "docker(1yuyan/netcdf-mpi:latest)" \
       -o "${LOGDIR}/Plot_v52_${year}.out" \
       bash -lc $'\n'"\
. /opt/conda/bin/activate && \
/bin/bash && \
echo \"[\$(hostname)] Plotting v5.2: ${year}\"
echo \"Start: \$(date)  Job: \$LSB_JOBID\"
python3 -u ${PLOT_SCRIPT} --year ${year}
echo \"End: \$(date)  Plotting done: ${year}\"
"
done

echo ""
echo "========================================"
echo "  Pipeline submitted!"
echo "========================================"
echo "  GeoNO2 jobs: ${#GEONO2_JOB_NAMES[@]}"
echo "  Evaluation:  Eval_v52_* (runs after all GeoNO2 jobs finish)"
echo "  Plots:       Plot_v52_* (runs after evaluation finishes)"
echo ""
echo "  Monitor with:  bjobs -g ${GROUP}"
echo "  Or:            bjobs -J 'GeoNO2v52_*'"
echo "  Eval/Plot:     bjobs -J 'Eval_v52_*' ; bjobs -J 'Plot_v52_*'"
echo "  Logs:          ${LOGDIR}/"
