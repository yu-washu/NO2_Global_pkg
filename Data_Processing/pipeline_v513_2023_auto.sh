#!/usr/bin/env bash
###############################################################################
# pipeline_v513_2023_auto.sh
#
# Automated pipeline for 2023 GeoNO2 v5.13 processing.
# Reuses existing NO2col-v3 tessellation/averaging from v5.3.
# Only runs stages 5-7 (GeoNO2 derivation, plot, evaluation).
#
# v5.13 change: eta_trop CLAMPING with v5.7 bounds [1.5e-17, 1.5e-15]
#
# Stages:
#   5. GeoNO2 v5.13 derivation (--no-plot, all 12 months)
#   6. GeoNO2 Jan plot-only (quick check)
#   7. Evaluation (compr + scatter)
#
# Usage:  nohup bash pipeline_v513_2023_auto.sh > pipeline_v513_2023.log 2>&1 &
###############################################################################
set -uo pipefail

YEAR=2023
POLL_INTERVAL=300   # 5 min between checks

# ── Real filesystem paths (for file-existence checks) ──
BASE_FS="/rdcw/fs2/rvmartin2/Active/yany1/1.project"
GEONO2_FS="${BASE_FS}/GeoNO2-v5.13/${YEAR}"

# ── Docker-visible paths (inside container) ──
GEONO2_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/GeoNO2"
COMPR_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Model_Evaluation_pkg/compr"

LOGDIR="${BASE_FS}/NO2_DL_global/NO2_global_pkg/Data_Processing/pipeline_logs_v513_2023"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

###############################################################################
# Helpers
###############################################################################
wait_for_jobs() {
  local pattern="$1"
  local label="$2"
  while true; do
    local n
    n=$(bjobs -w 2>/dev/null | grep -c "$pattern" || true)
    if [[ "$n" -eq 0 ]]; then
      log "All ${label} jobs completed."
      return 0
    fi
    log "Waiting for ${label}: ${n} jobs remaining..."
    sleep "$POLL_INTERVAL"
  done
}

submit_bsub() {
  local job_name="$1" mem="$2" group="$3" log_file="$4" cmd_body="$5"
  local docker="${6:-1yuyan/netcdf-mpi:latest}"
  local queue="${7:-general}"
  bsub -q "${queue}" \
       -J "${job_name}" -g "${group}" -n 1 -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin -N \
       -R "rusage[mem=${mem}] span[hosts=1] select[port8543=1]" \
       -a "docker(${docker})" \
       -o "${log_file}" \
       bash -lc ". /opt/conda/bin/activate && /bin/bash && ulimit -s unlimited && ${cmd_body}"
}

###############################################################################
# STAGE 5: GeoNO2 v5.13 derivation (--no-plot)
###############################################################################
stage5_geono2() {
  log "=== STAGE 5: Submitting GeoNO2 v5.13 processing (--no-plot) ==="

  local group="/yany1/geono2v513"
  bgmod -L 12 "${group}" 2>/dev/null || bgadd -L 12 "${group}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local outfile="${GEONO2_FS}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$outfile" ]]; then
      log "  GeoNO2 month ${mstr}: already exists, skipping"
      continue
    fi
    local job_name="GeoV513_${YEAR}${mstr}"
    local log_file="${LOGDIR}/geono2_v513_${YEAR}_${mstr}.out"
    local cmd="cd ${GEONO2_CD} && python3 -u geono2_v5.13.py ${YEAR} --month ${month} --no-plot"
    submit_bsub "${job_name}" 200000 "${group}" "${log_file}" "${cmd}"
    log "  Submitted GeoNO2 v5.13: ${mstr}"
  done

  wait_for_jobs "GeoV513_${YEAR}" "GeoNO2 v5.13"

  # Verify
  local all_ok=true
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    if [[ ! -f "${GEONO2_FS}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc" ]]; then
      log "  [ERROR] Missing GeoNO2 v5.13 file for ${mstr}!"
      all_ok=false
    fi
  done
  if [[ "$all_ok" != true ]]; then
    log "[WARN] Some GeoNO2 files missing, continuing anyway..."
  fi

  log "=== STAGE 5 COMPLETE ==="
}

###############################################################################
# STAGE 6: GeoNO2 Jan plot-only (quick visual check)
###############################################################################
stage6_geono2_jan_plot() {
  log "=== STAGE 6: GeoNO2 v5.13 Jan plot-only ==="

  local group="/yany1/geono2v513_plot"
  bgmod -L 2 "${group}" 2>/dev/null || bgadd -L 2 "${group}"

  local job_name="GeoPlotV513_${YEAR}01"
  local log_file="${LOGDIR}/geono2_v513_plot_${YEAR}_01.out"
  local cmd="cd ${GEONO2_CD} && python3 -u geono2_v5.13.py ${YEAR} --month 1 --plot-only"
  submit_bsub "${job_name}" 400000 "${group}" "${log_file}" "${cmd}"
  log "  Submitted Jan plot-only"

  wait_for_jobs "GeoPlotV513_" "GeoNO2 Jan plot"
  log "=== STAGE 6 COMPLETE ==="
}

###############################################################################
# STAGE 7: Evaluation (compr + scatter)
###############################################################################
stage7_evaluation() {
  log "=== STAGE 7: Submitting evaluation ==="

  local group="/yany1/evalv513"
  bgmod -L 2 "${group}" 2>/dev/null || bgadd -L 2 "${group}"

  # compr first
  local job_name="ComprV513_${YEAR}"
  local log_file="${LOGDIR}/compr_v513_${YEAR}.out"
  local cmd="cd ${COMPR_CD} && python3 -u compr_geo_gchp_global_v513.py --year ${YEAR}"
  submit_bsub "${job_name}" 200000 "${group}" "${log_file}" "${cmd}" "1yuyan/python-gfortran:latest"
  log "  Submitted compr_geo_gchp_global_v513.py"

  wait_for_jobs "ComprV513_${YEAR}" "evaluation comparison"

  # scatter plot
  local job_name2="ScatterV513_${YEAR}"
  local log_file2="${LOGDIR}/scatter_v513_${YEAR}.out"
  local cmd2="cd ${COMPR_CD} && python3 -u plot_scatter_global_v513.py --year ${YEAR}"
  submit_bsub "${job_name2}" 200000 "${group}" "${log_file2}" "${cmd2}" "1yuyan/python-gfortran:latest"
  log "  Submitted plot_scatter_global_v513.py"

  wait_for_jobs "ScatterV513_${YEAR}" "scatter plot"
  log "=== STAGE 7 COMPLETE ==="
}

###############################################################################
# MAIN
###############################################################################
log "================================================================"
log "  v5.13 PIPELINE FOR TROPOMI NO2 ${YEAR}"
log "  PID: $$"
log "  Change from v5.3: eta_trop clamping [1.5e-17, 1.5e-15] (v5.7 bounds, clamped)"
log "  Reuses NO2col-v3 tessellation/averaging from v5.3"
log "================================================================"

stage5_geono2
stage7_evaluation

log "================================================================"
log "  v5.13 PIPELINE COMPLETE FOR ${YEAR}"
log "================================================================"
