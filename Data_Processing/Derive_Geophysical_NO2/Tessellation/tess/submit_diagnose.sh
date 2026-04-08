#!/usr/bin/env bash
# Submit diagnostic job: GC shape factor trop vs total analysis
# Usage: bash submit_diagnose.sh [YYYY MM DD]
#   default: 2023 7 15

YEAR=${1:-2023}
MONTH=${2:-7}
DAY=${3:-15}

SCRIPT="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess/diagnose_gcshape_trop_vs_tot.py"
LOGDIR="/rdcw/fs2/rvmartin2/Active/yany1/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess/diagnostics"
mkdir -p "$LOGDIR"

LABEL=$(printf "%04d%02d%02d" "$YEAR" "$MONTH" "$DAY")

bsub -q rvmartin \
     -J "DiagGCshape_${LABEL}" \
     -n 1 \
     -W 24:00 \
     -u yany1@wustl.edu -G compute-rvmartin \
     -R "select[port8543=1]" \
     -R "rusage[mem=50000]" \
     -a "docker(1yuyan/netcdf-mpi:latest)" \
     -o "${LOGDIR}/diagnose_gcshape_${LABEL}.out" \
     bash -lc ". /opt/conda/bin/activate && /bin/bash && ulimit -s unlimited && python3 -u ${SCRIPT} ${YEAR} ${MONTH} ${DAY} --max-pixels 50000"

echo "Submitted DiagGCshape_${LABEL} — check ${LOGDIR}/diagnose_gcshape_${LABEL}.out"
