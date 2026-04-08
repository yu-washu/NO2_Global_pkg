#!/usr/bin/env bash
###############################################################################
# pipeline_omi_2005_auto.sh
#
# Full automated pipeline for OMI NO2 2005 with GeoNO2 v5.10 + v5.13 (clamped).
# Runs all stages from GCHP regridding through evaluation.
#
# Stages:
#   1. GCHP regridding (365 days)                        — netcdf-mpi, general+rvmartin
#   2. OMI tessellation NO2col v3 (365 days, split)      — intel-py, rvmartin+general
#   3. Monthly + yearly averages (NO2col, 12+1 months)   — netcdf-mpi, qa queue
#   4. GeoNO2 v5.10 derivation (12 months, OMI)          — netcdf-mpi, general queue
#   5. GeoNO2 v5.13 derivation (12 months, OMI)          — netcdf-mpi, qa queue
#   6. Evaluation v5.10 + v5.13 (compr + scatter)         — python-gfortran
#
# Queue strategy: GCHP regrid on general+rvmartin, tess after regrid on rvmartin+general.
# AMF tessellation skipped (already validated with TROPOMI).
#
# Usage:  nohup bash pipeline_omi_2005_auto.sh > pipeline_omi_2005.log 2>&1 &
###############################################################################
set -uo pipefail

YEAR=2005
POLL_INTERVAL=300   # 5 min between checks
QC_OMI="CF050-SZA80-QA0-RA0-SI110-SI2252-SI3255"

# ── Real filesystem paths (for file-existence checks) ──
BASE_FS="/rdcw/fs2/rvmartin2/Active/yany1/1.project"
BASE_FS1="/rdcw/fs1/rvmartin/Active/yany1/1.project"

# GCHP regrid outputs
GCHP_TESS_FS="${BASE_FS}/gchp-v2/forTessellation/${YEAR}/daily"
GCHP_GEO_FS="${BASE_FS}/gchp-v2/forObservation-Geophysical/${YEAR}/daily"

# OMI tessellation outputs (no AMF — skipped, validated with TROPOMI)
DAILY_OMI_FS="${BASE_FS}/NO2col-v3/OMI_KNMI/${YEAR}/daily"
MONTHLY_OMI_FS="${BASE_FS}/NO2col-v3/OMI_KNMI/${YEAR}/monthly"
YEARLY_OMI_FS="${BASE_FS}/NO2col-v3/OMI_KNMI/${YEAR}/yearly"

# GeoNO2 outputs
GEONO2_V510_FS="${BASE_FS}/GeoNO2-v5.10/${YEAR}"
GEONO2_V513_FS="${BASE_FS}/GeoNO2-v5.13/${YEAR}"

# ── Docker-visible paths (inside container) ──
REGRID_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Regrid_GCHP"
TESS_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess"
AMF_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/amf"
AV_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av"
GEONO2_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/GeoNO2"
COMPR_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Model_Evaluation_pkg/compr"

LOGDIR="${BASE_FS}/NO2_DL_global/NO2_global_pkg/Data_Processing/pipeline_logs_omi_2005"
mkdir -p "$LOGDIR"

# 2005 is not a leap year
declare -A DAYS_IN_MONTH=(
  [1]=31 [2]=28 [3]=31 [4]=30 [5]=31 [6]=30
  [7]=31 [8]=31 [9]=30 [10]=31 [11]=30 [12]=31
)
TOTAL_DAYS=365

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

# Tessellation jobs — intel-py docker
submit_tess() {
  local job_name="$1" mem="$2" group="$3" log_file="$4" cmd_body="$5"
  local queue="${6:-general}"
  bsub -q "${queue}" \
       -J "${job_name}" -g "${group}" -n 1 -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin -N \
       -R "rusage[mem=${mem}] span[hosts=1] select[port8543=1]" \
       -a 'docker(1yuyan/intel-py:202508)' \
       -o "${log_file}" \
       bash -lc ". /opt/conda/bin/activate && /bin/bash && ulimit -s unlimited && ${cmd_body}"
}

# General job submission — netcdf-mpi or python-gfortran docker
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

count_files() {
  local dir="$1" pattern="$2"
  find "$dir" -maxdepth 1 -name "$pattern" 2>/dev/null | wc -l
}

###############################################################################
# STAGE 1: GCHP Regridding (365 days) — general queue only
###############################################################################
stage1_gchp_regrid() {
  log "=== STAGE 1: Submitting GCHP regridding for ${YEAR} ==="

  local group="/yany1/regrid2005"
  bgmod -L 40 "${group}" 2>/dev/null || bgadd -L 40 "${group}"

  local n_submitted=0
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local ndays=${DAYS_IN_MONTH[$month]}
    for day in $(seq 1 "$ndays"); do
      local dstr=$(printf "%02d" "$day")
      local label="${YEAR}${mstr}${dstr}"

      # Check if both outputs already exist
      local tess_out="${GCHP_TESS_FS}/01x01.Hours.13-15.${label}.nc4"
      local geo_out="${GCHP_GEO_FS}/1x1km.Hours.13-15.${label}.nc4"
      if [[ -f "$tess_out" && -f "$geo_out" ]]; then continue; fi

      local job_name="Rgrd_${label}"
      local log_file="${LOGDIR}/regrid_${label}.out"
      local cmd="cd ${REGRID_CD} && python3 -u main_nearest_neighbour.py --year ${YEAR} --mon ${month} --day ${day}"
      submit_bsub "${job_name}" 150000 "${group}" "${log_file}" "${cmd}" "1yuyan/netcdf-mpi:latest" "general"
      n_submitted=$((n_submitted + 1))
    done
  done
  log "  Submitted ${n_submitted} GCHP regridding jobs on general queue"

  wait_for_jobs "Rgrd_${YEAR}" "GCHP regridding"

  # Verify
  local have_tess have_geo
  have_tess=$(count_files "${GCHP_TESS_FS}" "01x01.Hours.13-15.${YEAR}*.nc4")
  have_geo=$(count_files "${GCHP_GEO_FS}" "1x1km.Hours.13-15.${YEAR}*.nc4")
  log "  GCHP regrid complete: forTessellation=${have_tess}/${TOTAL_DAYS}, forObservation=${have_geo}/${TOTAL_DAYS}"

  if [[ "$have_tess" -lt "$TOTAL_DAYS" || "$have_geo" -lt "$TOTAL_DAYS" ]]; then
    log "[WARN] Some GCHP regrid files missing, continuing anyway..."
  fi

  log "=== STAGE 1 COMPLETE ==="
}

###############################################################################
# STAGE 2: OMI Tessellation NO2col v3 (365 days) — split rvmartin + general
###############################################################################
stage2_omi_tess() {
  log "=== STAGE 2: Submitting OMI NO2col tessellation for ${YEAR} ==="

  local group="/yany1/omi_tessv3"
  bgmod -L 40 "${group}" 2>/dev/null || bgadd -L 40 "${group}"

  local n_submitted=0
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local ndays=${DAYS_IN_MONTH[$month]}
    # Split months: Jan-Jun → rvmartin, Jul-Dec → general
    local queue="rvmartin"
    if [[ "$month" -gt 6 ]]; then queue="general"; fi

    for day in $(seq 1 "$ndays"); do
      local dstr=$(printf "%02d" "$day")
      local label="${YEAR}${mstr}${dstr}"
      local outfile="${DAILY_OMI_FS}/OMI_KNMI_Regrid_${label}_${QC_OMI}.nc"
      if [[ -f "$outfile" ]]; then continue; fi

      local job_name="OT3_${label}"
      local log_file="${LOGDIR}/tess_omi_${label}.out"
      local cmd="cd ${TESS_DIR} && python3 -u tess_OMI_KNMI_v3.py --year ${YEAR} --mon ${month} --day ${day}"
      submit_tess "${job_name}" 150000 "${group}" "${log_file}" "${cmd}" "${queue}"
      n_submitted=$((n_submitted + 1))
    done
  done
  log "  Submitted ${n_submitted} OMI NO2col tess jobs (Jan-Jun→rvmartin, Jul-Dec→general)"
}

###############################################################################
# STAGE 3: Monthly averages (NO2col only, no AMF) — qa queue
###############################################################################
stage3_monthly_averages() {
  log "=== STAGE 3: Waiting for tessellation, then monthly averages ==="

  # Wait for all OMI NO2col daily files
  wait_for_jobs "OT3_${YEAR}" "OMI NO2col tessellation"
  local have_omi
  have_omi=$(count_files "${DAILY_OMI_FS}" "OMI_KNMI_Regrid_${YEAR}*_${QC_OMI}.nc")
  log "  OMI NO2col daily: ${have_omi}/${TOTAL_DAYS} files"

  # Submit NO2col monthly averages
  local group="/yany1/omi_monthly"
  bgmod -L 14 "${group}" 2>/dev/null || bgadd -L 14 "${group}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local outfile="${MONTHLY_OMI_FS}/OMI_KNMI_Regrid_${YEAR}${mstr}_Monthly_${QC_OMI}.nc"
    if [[ -f "$outfile" ]]; then
      log "  NO2col month ${mstr}: already exists, skipping"
      continue
    fi
    local job_name="MonOMI_${YEAR}${mstr}"
    local log_file="${LOGDIR}/monthly_omi_${YEAR}_${mstr}.out"
    local cmd="cd ${AV_DIR} && python3 -u omi_KNMI_average_v3.py ${YEAR} --month ${month} --no-plot"
    submit_bsub "${job_name}" 150000 "${group}" "${log_file}" "${cmd}" "1yuyan/netcdf-mpi:latest" "qa"
    log "  Submitted OMI NO2col monthly avg: ${mstr}"
  done

  wait_for_jobs "MonOMI_${YEAR}" "OMI NO2col monthly average"

  # Verify NO2col monthly files
  local all_ok=true
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    if [[ ! -f "${MONTHLY_OMI_FS}/OMI_KNMI_Regrid_${YEAR}${mstr}_Monthly_${QC_OMI}.nc" ]]; then
      log "  [ERROR] Missing OMI NO2col monthly file for ${mstr}!"
      all_ok=false
    fi
  done
  if [[ "$all_ok" != true ]]; then
    log "[FATAL] Not all OMI monthly files created. Aborting."
    exit 1
  fi

  log "=== STAGE 3 COMPLETE ==="
}

###############################################################################
# STAGE 4: Yearly average (NO2col only) — general queue
###############################################################################
stage4_yearly_average() {
  log "=== STAGE 4: Submitting OMI yearly average ==="

  local group="/yany1/omi_yearly"
  bgmod -L 2 "${group}" 2>/dev/null || bgadd -L 2 "${group}"

  local outfile="${YEARLY_OMI_FS}/OMI_KNMI_Regrid_${YEAR}_${QC_OMI}.nc"
  if [[ -f "$outfile" ]]; then
    log "  Yearly file already exists, skipping"
  else
    local job_name="YearOMI_${YEAR}"
    local log_file="${LOGDIR}/yearly_omi_${YEAR}.out"
    local cmd="cd ${AV_DIR} && python3 -u omi_KNMI_average_v3.py ${YEAR} --yearly-only --no-plot"
    submit_bsub "${job_name}" 300000 "${group}" "${log_file}" "${cmd}" "1yuyan/netcdf-mpi:latest" "general"
    log "  Submitted OMI yearly average"

    wait_for_jobs "YearOMI_${YEAR}" "OMI yearly average"
  fi

  log "=== STAGE 4 COMPLETE ==="
}

###############################################################################
# STAGE 5: GeoNO2 v5.10 derivation (OMI, 12 months) — general queue
###############################################################################
stage5_geono2_v510() {
  log "=== STAGE 5: Submitting GeoNO2 v5.10 (OMI) ==="

  local group="/yany1/geo510_omi"
  bgmod -L 12 "${group}" 2>/dev/null || bgadd -L 12 "${group}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local outfile="${GEONO2_V510_FS}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$outfile" ]]; then
      log "  GeoNO2 v5.10 month ${mstr}: already exists, skipping"
      continue
    fi
    local job_name="G510O_${YEAR}${mstr}"
    local log_file="${LOGDIR}/geono2_v510_omi_${YEAR}_${mstr}.out"
    local cmd="cd ${GEONO2_CD} && python3 -u geono2_v5.10.py ${YEAR} --month ${month} --instrument OMI --no-plot"
    submit_bsub "${job_name}" 200000 "${group}" "${log_file}" "${cmd}" "1yuyan/netcdf-mpi:latest" "general"
    log "  Submitted GeoNO2 v5.10 OMI: ${mstr}"
  done

  wait_for_jobs "G510O_${YEAR}" "GeoNO2 v5.10 OMI"

  local all_ok=true
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    if [[ ! -f "${GEONO2_V510_FS}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc" ]]; then
      log "  [ERROR] Missing GeoNO2 v5.10 file for ${mstr}!"
      all_ok=false
    fi
  done
  if [[ "$all_ok" != true ]]; then
    log "[WARN] Some GeoNO2 v5.10 files missing, continuing..."
  fi

  log "=== STAGE 5 COMPLETE ==="
}

###############################################################################
# STAGE 6: GeoNO2 v5.13 derivation (OMI, 12 months) — qa queue
###############################################################################
stage6_geono2_v513() {
  log "=== STAGE 6: Submitting GeoNO2 v5.13 (OMI) ==="

  local group="/yany1/geo513_omi"
  bgmod -L 12 "${group}" 2>/dev/null || bgadd -L 12 "${group}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local outfile="${GEONO2_V513_FS}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$outfile" ]]; then
      log "  GeoNO2 v5.13 month ${mstr}: already exists, skipping"
      continue
    fi
    local job_name="G513O_${YEAR}${mstr}"
    local log_file="${LOGDIR}/geono2_v513_omi_${YEAR}_${mstr}.out"
    local cmd="cd ${GEONO2_CD} && python3 -u geono2_v5.13.py ${YEAR} --month ${month} --instrument OMI --no-plot"
    submit_bsub "${job_name}" 200000 "${group}" "${log_file}" "${cmd}" "1yuyan/netcdf-mpi:latest" "qa"
    log "  Submitted GeoNO2 v5.13 OMI: ${mstr}"
  done

  wait_for_jobs "G513O_${YEAR}" "GeoNO2 v5.13 OMI"

  local all_ok=true
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    if [[ ! -f "${GEONO2_V513_FS}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc" ]]; then
      log "  [ERROR] Missing GeoNO2 v5.13 file for ${mstr}!"
      all_ok=false
    fi
  done
  if [[ "$all_ok" != true ]]; then
    log "[WARN] Some GeoNO2 v5.13 files missing, continuing..."
  fi

  log "=== STAGE 6 COMPLETE ==="
}

###############################################################################
# STAGE 7: Evaluation — v5.10 + v5.13 (compr + scatter)
###############################################################################
stage7_evaluation() {
  log "=== STAGE 7: Submitting evaluation for v5.10 and v5.13 ==="

  local group="/yany1/eval_omi2005"
  bgmod -L 4 "${group}" 2>/dev/null || bgadd -L 4 "${group}"

  # ---- v5.10 evaluation ----
  local job_name="ComprV510_${YEAR}"
  local log_file="${LOGDIR}/compr_v510_omi_${YEAR}.out"
  local cmd="cd ${COMPR_CD} && python3 -u compr_geo_gchp_global_v510.py --year ${YEAR}"
  submit_bsub "${job_name}" 200000 "${group}" "${log_file}" "${cmd}" "1yuyan/python-gfortran:latest" "general"
  log "  Submitted compr v5.10"

  # ---- v5.13 evaluation ----
  local job_name2="ComprV513_${YEAR}"
  local log_file2="${LOGDIR}/compr_v513_omi_${YEAR}.out"
  local cmd2="cd ${COMPR_CD} && python3 -u compr_geo_gchp_global_v513.py --year ${YEAR}"
  submit_bsub "${job_name2}" 200000 "${group}" "${log_file2}" "${cmd2}" "1yuyan/python-gfortran:latest" "qa"
  log "  Submitted compr v5.13"

  wait_for_jobs "ComprV51" "evaluation comparison"

  # ---- Scatter plots ----
  local job_name3="ScatterV510_${YEAR}"
  local log_file3="${LOGDIR}/scatter_v510_omi_${YEAR}.out"
  local cmd3="cd ${COMPR_CD} && python3 -u plot_scatter_global_v510.py --year ${YEAR}"
  submit_bsub "${job_name3}" 200000 "${group}" "${log_file3}" "${cmd3}" "1yuyan/python-gfortran:latest" "general"

  local job_name4="ScatterV513_${YEAR}"
  local log_file4="${LOGDIR}/scatter_v513_omi_${YEAR}.out"
  local cmd4="cd ${COMPR_CD} && python3 -u plot_scatter_global_v513.py --year ${YEAR}"
  submit_bsub "${job_name4}" 200000 "${group}" "${log_file4}" "${cmd4}" "1yuyan/python-gfortran:latest" "qa"
  log "  Submitted scatter plots"

  wait_for_jobs "ScatterV51" "scatter plots"
  log "=== STAGE 7 COMPLETE ==="
}

###############################################################################
# MAIN
###############################################################################
log "================================================================"
log "  OMI 2005 FULL PIPELINE"
log "  PID: $$"
log "  Stages: GCHP regrid → OMI tess → average → GeoNO2 v5.10+v5.13 → eval"
log "  Queues: general+rvmartin (regrid), rvmartin+general (tess), qa (avg+geo)"
log "  GeoNO2: eta_trop clamping (not clip-to-NaN)"
log "  AMF tessellation: SKIPPED (validated with TROPOMI)"
log "================================================================"

# Stage 1: GCHP regridding on qa (must complete before tessellation)
stage1_gchp_regrid

# Stage 2: OMI tessellation (Jan-Jun→rvmartin, Jul-Dec→general)
stage2_omi_tess

# Stage 3: Monthly averages (waits for stage 2)
stage3_monthly_averages

# Stage 4: Yearly average
stage4_yearly_average

# Stages 5+6: GeoNO2 v5.10 (general) and v5.13 (qa) — run in parallel
stage5_geono2_v510
stage6_geono2_v513

# Stage 7: Evaluation for both versions
stage7_evaluation

log "================================================================"
log "  OMI 2005 PIPELINE COMPLETE"
log "================================================================"
