#!/usr/bin/env bash
###############################################################################
# pipeline_2006_2022_auto.sh
#
# Multi-year pipeline for GeoNO2 v5.13 (clamped eta) processing.
#
# Instrument schedule:
#   2006-2017:      OMI only
#   2018 Jan-May:   OMI,  Jun-Dec: TROPOMI
#   2019-2022:      TROPOMI only
#
# Per-year stages (sequential):
#   1. GCHP regridding (daily)
#   2. Tessellation NO2col v3 (daily)
#   3a. GCHP aggregate (monthly + yearly) — parallel with 3b
#   3b. NO2col monthly + yearly average
#   4. GeoNO2 v5.13 --slim (12 months)
#   5. save_npy (extract to npy for ML)
#
# After all years:
#   6. derive_observation_HP.py (training labels)
#   7. sync to S3
#
# Usage:
#   nohup bash pipeline_2006_2022_auto.sh > pipeline_2006_2022.log 2>&1 &
###############################################################################
set -uo pipefail

POLL_INTERVAL=300
QC_TROP="SZA80-QA75"
QC_OMI="CF050-SZA80-QA0-RA0-SI110-SI2252-SI3255"

BASE_FS="/rdcw/fs2/rvmartin2/Active/yany1/1.project"

# Docker paths
REGRID_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Regrid_GCHP"
TESS_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess"
AV_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av"
GEONO2_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/GeoNO2"
LABEL_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Label"

LOGBASE="${BASE_FS}/NO2_DL_global/NO2_global_pkg/Data_Processing/pipeline_logs_multiyear"
mkdir -p "$LOGBASE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

###############################################################################
# Helpers
###############################################################################
wait_for_jobs() {
  local pattern="$1" label="$2"
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

submit_tess() {
  local job_name="$1" mem="$2" group="$3" log_file="$4" cmd_body="$5" queue="${6:-general}"
  bsub -q "${queue}" \
       -J "${job_name}" -g "${group}" -n 1 -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin -N \
       -R "rusage[mem=${mem}] span[hosts=1] select[port8543=1]" \
       -a 'docker(1yuyan/intel-py:202508)' \
       -o "${log_file}" \
       bash -lc ". /opt/conda/bin/activate && /bin/bash && ulimit -s unlimited && ${cmd_body}"
}

submit_bsub() {
  local job_name="$1" mem="$2" group="$3" log_file="$4" cmd_body="$5"
  local docker="${6:-1yuyan/netcdf-mpi:latest}" queue="${7:-general}"
  bsub -q "${queue}" \
       -J "${job_name}" -g "${group}" -n 1 -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin -N \
       -R "rusage[mem=${mem}] span[hosts=1] select[port8543=1]" \
       -a "docker(${docker})" \
       -o "${log_file}" \
       bash -lc ". /opt/conda/bin/activate && /bin/bash && ulimit -s unlimited && ${cmd_body}"
}

days_in_month() {
  local year=$1 month=$2
  cal "$month" "$year" | awk 'NF {DAYS=$NF} END {print DAYS}'
}

get_instrument() {
  local year=$1 month=$2
  if [[ "$year" -le 2017 ]]; then echo "OMI"
  elif [[ "$year" -eq 2018 && "$month" -le 5 ]]; then echo "OMI"
  else echo "TROPOMI"; fi
}

get_tess_script() {
  if [[ "$1" == "OMI" ]]; then echo "tess_OMI_KNMI_v3.py"; else echo "tess_TROPOMI_v3.py"; fi
}

get_av_script() {
  if [[ "$1" == "OMI" ]]; then echo "omi_KNMI_average_v3.py"; else echo "tropomi_average_HP_v3.py"; fi
}

get_qc() {
  if [[ "$1" == "OMI" ]]; then echo "$QC_OMI"; else echo "$QC_TROP"; fi
}

get_daily_prefix() {
  if [[ "$1" == "OMI" ]]; then echo "OMI_KNMI_Regrid"; else echo "Tropomi_Regrid"; fi
}

get_sat_dir() {
  local inst="$1" year="$2"
  if [[ "$inst" == "OMI" ]]; then echo "${BASE_FS}/NO2col-v3/OMI_KNMI/${year}"
  else echo "${BASE_FS}/NO2col-v3/TROPOMI/${year}"; fi
}

###############################################################################
# Process a single year
###############################################################################
process_year() {
  local YEAR=$1
  local LOGDIR="${LOGBASE}/${YEAR}"
  mkdir -p "$LOGDIR"

  log "================================================================"
  log "  YEAR ${YEAR} — START"
  log "================================================================"

  # Determine which instruments are needed
  local inst_jan; inst_jan=$(get_instrument "$YEAR" 1)
  local inst_dec; inst_dec=$(get_instrument "$YEAR" 12)
  local -a instruments=("$inst_jan")
  if [[ "$inst_jan" != "$inst_dec" ]]; then
    instruments+=("$inst_dec")
  fi
  log "  Instruments: ${instruments[*]}"

  # ================================================================
  # STAGE 1: GCHP Regridding
  # ================================================================
  log "  [Stage 1] GCHP regridding..."
  local rg="/yany1/rg${YEAR}"
  bgmod -L 30 "${rg}" 2>/dev/null || bgadd -L 30 "${rg}"

  local n_rg=0
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local ndays; ndays=$(days_in_month "$YEAR" "$month")
    for day in $(seq 1 "$ndays"); do
      local dstr=$(printf "%02d" "$day")
      local label="${YEAR}${mstr}${dstr}"
      local t_out="${BASE_FS}/gchp-v2/forTessellation/${YEAR}/daily/01x01.Hours.13-15.${label}.nc4"
      local g_out="${BASE_FS}/gchp-v2/forObservation-Geophysical/${YEAR}/daily/1x1km.Hours.13-15.${label}.nc4"
      if [[ -f "$t_out" && -f "$g_out" ]]; then continue; fi

      submit_bsub "Rg_${label}" 150000 "${rg}" "${LOGDIR}/rg_${label}.out" \
        "cd ${REGRID_CD} && python3 -u main_nearest_neighbour.py --year ${YEAR} --mon ${month} --day ${day}"
      n_rg=$((n_rg + 1))
    done
  done
  log "  Stage 1: ${n_rg} regrid jobs submitted"
  wait_for_jobs "Rg_${YEAR}" "regrid ${YEAR}"

  # ================================================================
  # STAGE 2: Tessellation
  # ================================================================
  log "  [Stage 2] Tessellation..."
  local ts="/yany1/ts${YEAR}"
  bgmod -L 40 "${ts}" 2>/dev/null || bgadd -L 40 "${ts}"

  local n_ts=0
  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local inst; inst=$(get_instrument "$YEAR" "$month")
    local sat_dir; sat_dir=$(get_sat_dir "$inst" "$YEAR")
    local tess_script; tess_script=$(get_tess_script "$inst")
    local qc; qc=$(get_qc "$inst")
    local prefix; prefix=$(get_daily_prefix "$inst")
    local ndays; ndays=$(days_in_month "$YEAR" "$month")

    for day in $(seq 1 "$ndays"); do
      local dstr=$(printf "%02d" "$day")
      local label="${YEAR}${mstr}${dstr}"
      local outfile="${sat_dir}/daily/${prefix}_${label}_${qc}.nc"
      if [[ -f "$outfile" ]]; then continue; fi

      submit_tess "Ts_${label}" 150000 "${ts}" "${LOGDIR}/ts_${label}.out" \
        "cd ${TESS_DIR} && python3 -u ${tess_script} --year ${YEAR} --mon ${month} --day ${day}"
      n_ts=$((n_ts + 1))
    done
  done
  log "  Stage 2: ${n_ts} tessellation jobs submitted"
  wait_for_jobs "Ts_${YEAR}" "tessellation ${YEAR}"

  # ================================================================
  # STAGE 3a: GCHP aggregate (monthly then yearly)
  # ================================================================
  log "  [Stage 3a] GCHP aggregate..."
  local ag="/yany1/ag${YEAR}"
  bgmod -L 12 "${ag}" 2>/dev/null || bgadd -L 12 "${ag}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local monthly_out="${BASE_FS}/gchp-v2/forObservation-Geophysical/${YEAR}/monthly/1x1km.Hours.13-15.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$monthly_out" ]]; then continue; fi
    submit_bsub "Ag_${YEAR}${mstr}" 150000 "${ag}" "${LOGDIR}/ag_${YEAR}_${mstr}.out" \
      "cd ${REGRID_CD} && python3 -u aggregate.py ${YEAR} --month ${month} --no-plot"
  done

  # ================================================================
  # STAGE 3b: NO2col monthly averages (in parallel with 3a)
  # ================================================================
  log "  [Stage 3b] NO2col monthly averages..."
  local av="/yany1/av${YEAR}"
  bgmod -L 14 "${av}" 2>/dev/null || bgadd -L 14 "${av}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local inst; inst=$(get_instrument "$YEAR" "$month")
    local av_script; av_script=$(get_av_script "$inst")
    submit_bsub "Av_${YEAR}${mstr}" 150000 "${av}" "${LOGDIR}/av_${YEAR}_${mstr}.out" \
      "cd ${AV_DIR} && python3 -u ${av_script} ${YEAR} --month ${month} --no-plot"
  done

  # Wait for both 3a and 3b monthly
  wait_for_jobs "Ag_${YEAR}" "GCHP aggregate ${YEAR}"
  wait_for_jobs "Av_${YEAR}" "NO2col average ${YEAR}"

  # GCHP yearly
  submit_bsub "AgY${YEAR}" 200000 "${ag}" "${LOGDIR}/ag_yearly_${YEAR}.out" \
    "cd ${REGRID_CD} && python3 -u aggregate.py ${YEAR} --yearly-only --no-plot"
  wait_for_jobs "AgY${YEAR}" "GCHP yearly ${YEAR}"

  # NO2col yearly (one per instrument used this year)
  for inst in "${instruments[@]}"; do
    local av_script; av_script=$(get_av_script "$inst")
    local tag; tag=$(echo "$inst" | cut -c1-3)
    submit_bsub "AvY${tag}${YEAR}" 300000 "${av}" "${LOGDIR}/av_yearly_${tag}_${YEAR}.out" \
      "cd ${AV_DIR} && python3 -u ${av_script} ${YEAR} --yearly-only --no-plot"
  done
  wait_for_jobs "AvY" "NO2col yearly ${YEAR}"

  # ================================================================
  # STAGE 4: GeoNO2 v5.13 --slim
  # ================================================================
  log "  [Stage 4] GeoNO2 v5.13 --slim..."
  local ge="/yany1/ge${YEAR}"
  bgmod -L 12 "${ge}" 2>/dev/null || bgadd -L 12 "${ge}"

  for month in $(seq 1 12); do
    local mstr=$(printf "%02d" "$month")
    local inst; inst=$(get_instrument "$YEAR" "$month")
    local outfile="${BASE_FS}/GeoNO2-v5.13/${YEAR}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$outfile" ]]; then continue; fi

    submit_bsub "Ge_${YEAR}${mstr}" 200000 "${ge}" "${LOGDIR}/ge_${YEAR}_${mstr}.out" \
      "cd ${GEONO2_CD} && python3 -u geono2_v5.13.py ${YEAR} --month ${month} --instrument ${inst} --slim --no-plot"
  done
  log "  Stage 4: GeoNO2 jobs submitted"
  wait_for_jobs "Ge_${YEAR}" "GeoNO2 ${YEAR}"

  # ================================================================
  # STAGE 5: save_npy
  # ================================================================
  log "  [Stage 5] Saving npy..."
  submit_bsub "Np_${YEAR}" 100000 "${ge}" "${LOGDIR}/npy_${YEAR}.out" \
    "cd ${GEONO2_CD} && python3 -u save_npy_v5_clamp.py ${YEAR} --version v5.13"
  wait_for_jobs "Np_${YEAR}" "save_npy ${YEAR}"

  log "  YEAR ${YEAR} — COMPLETE"
  log ""
}

###############################################################################
# MAIN
###############################################################################
log "================================================================"
log "  MULTI-YEAR PIPELINE 2006-2022"
log "  GeoNO2 v5.13 (clamped eta, --slim output)"
log "  OMI: 2006-2018 May | TROPOMI: 2018 Jun-2022"
log "  PID: $$"
log "================================================================"

for YEAR in $(seq 2006 2022); do
  process_year "$YEAR"
done

# ==== STAGE 6: Derive observation labels ====
log "=== Stage 6: derive_observation_HP.py ==="
submit_bsub "DeriveObs" 200000 "/yany1/derive" "${LOGBASE}/derive_obs.out" \
  "cd ${LABEL_CD} && python3 -u derive_observation_HP.py" \
  "1yuyan/python-gfortran:latest"
wait_for_jobs "DeriveObs" "derive observation labels"

# ==== STAGE 7: Sync to S3 ====
log "=== Stage 7: sync to S3 ==="
bash "${BASE_FS}/NO2_DL_global/sync_all.sh" 2>&1 | while read -r line; do log "  $line"; done

log "================================================================"
log "  MULTI-YEAR PIPELINE COMPLETE (2006-2022)"
log "================================================================"
