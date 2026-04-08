#!/usr/bin/env bash
###############################################################################
# submit_parallel_2007_2022.sh
#
# Submit ALL remaining years (2007-2022) in parallel using LSF dependencies.
# No bash wait loops — LSF handles ordering via -w "done(...)".
#
# Per-year dependency chain:
#   Regrid → Tess → (GCHP agg + NO2col avg) → GeoNO2 → save_npy
#
# Queue distribution:
#   Regrid:  odd years→general, even years→rvmartin
#   Tess:    odd years→rvmartin, even years→general  (opposite of regrid)
#   Rest:    general
#
# Usage:
#   nohup bash submit_parallel_2007_2022.sh > submit_parallel_2007_2022.log 2>&1 &
###############################################################################
set -uo pipefail

BASE_FS="/rdcw/fs2/rvmartin2/Active/yany1/1.project"
REGRID_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Regrid_GCHP"
TESS_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/tess"
AV_DIR="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/av"
GEONO2_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/GeoNO2"
LABEL_CD="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Label"
LOGBASE="${BASE_FS}/NO2_DL_global/NO2_global_pkg/Data_Processing/pipeline_logs_multiyear"

QC_OMI="CF050-SZA80-QA0-RA0-SI110-SI2252-SI3255"
QC_TROP="SZA80-QA75"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

days_in_month() { cal "$2" "$1" | awk 'NF {DAYS=$NF} END {print DAYS}'; }

get_instrument() {
  local year=$1 month=$2
  if [[ "$year" -le 2017 ]]; then echo "OMI"
  elif [[ "$year" -eq 2018 && "$month" -le 5 ]]; then echo "OMI"
  else echo "TROPOMI"; fi
}

# Leave cores free for others: ptile=48 on 64/72-core nodes
PTILE=48

submit_job() {
  local queue="$1" job_name="$2" group="$3" mem="$4" dep="$5" log_file="$6" docker="$7" cmd="$8"
  local -a dep_args=()
  if [[ -n "$dep" ]]; then dep_args=(-w "$dep"); fi
  bsub -q "${queue}" -J "${job_name}" -g "${group}" -n 1 -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin -N \
       -R "select[port8543=1]" \
       -R "span[ptile=${PTILE}]" \
       -R "rusage[mem=${mem}]" \
       -a "docker(${docker})" \
       "${dep_args[@]}" \
       -o "${log_file}" \
       bash -lc ". /opt/conda/bin/activate && /bin/bash && ulimit -s unlimited && ${cmd}"
}

###############################################################################
# Global job group to cap total concurrent jobs across all years
# This leaves capacity for other users
bgmod -L 500 "/yany1/multiyear" 2>/dev/null || bgadd -L 500 "/yany1/multiyear"

log "Submitting parallel pipeline for 2007-2022..."
log "  Global concurrency limit: 500 jobs"
log "  ptile=${PTILE} (leaves 16-24 cores free per node)"

TOTAL_SUBMITTED=0

for YEAR in $(seq 2007 2022); do
  LOGDIR="${LOGBASE}/${YEAR}"
  mkdir -p "$LOGDIR"

  # Queue assignment: alternate by year
  if (( YEAR % 2 == 1 )); then
    Q_RG="general";  Q_TS="rvmartin"
  else
    Q_RG="rvmartin"; Q_TS="general"
  fi

  log "  Year ${YEAR}: regrid→${Q_RG}, tess→${Q_TS}"

  GRP="/yany1/multiyear"  # shared global group

  # ==== REGRID (365 days) — skip if forTess already complete ====
  n_rg=0
  regrid_needed=false
  for month in $(seq 1 12); do
    mstr=$(printf "%02d" "$month")
    ndays=$(days_in_month "$YEAR" "$month")
    for day in $(seq 1 "$ndays"); do
      dstr=$(printf "%02d" "$day")
      label="${YEAR}${mstr}${dstr}"
      t_out="${BASE_FS}/gchp-v2/forTessellation/${YEAR}/daily/01x01.Hours.13-15.${label}.nc4"
      g_out="${BASE_FS}/gchp-v2/forObservation-Geophysical/${YEAR}/daily/1x1km.Hours.13-15.${label}.nc4"
      if [[ -f "$t_out" && -f "$g_out" ]]; then continue; fi

      regrid_needed=true
      submit_job "${Q_RG}" "Rg_${label}" "${GRP}" 150000 "" \
        "${LOGDIR}/rg_${label}.out" "1yuyan/netcdf-mpi:latest" \
        "cd ${REGRID_CD} && python3 -u main_nearest_neighbour.py --year ${YEAR} --mon ${month} --day ${day}"
      n_rg=$((n_rg + 1))
    done
  done
  if [[ "$regrid_needed" == false ]]; then
    log "  Stage 1: regrid already complete, skipping"
  fi

  # ==== TESSELLATION (365 days) ====
  # If regrid is done/running, tess starts immediately; otherwise depends on same-day regrid
  n_ts=0
  for month in $(seq 1 12); do
    mstr=$(printf "%02d" "$month")
    inst=$(get_instrument "$YEAR" "$month")
    if [[ "$inst" == "OMI" ]]; then
      tess_script="tess_OMI_KNMI_v3.py"
      prefix="OMI_KNMI_Regrid"; qc="$QC_OMI"
      sat_daily="${BASE_FS}/NO2col-v3/OMI_KNMI/${YEAR}/daily"
    else
      tess_script="tess_TROPOMI_v3.py"
      prefix="Tropomi_Regrid"; qc="$QC_TROP"
      sat_daily="${BASE_FS}/NO2col-v3/TROPOMI/${YEAR}/daily"
    fi

    ndays=$(days_in_month "$YEAR" "$month")
    for day in $(seq 1 "$ndays"); do
      dstr=$(printf "%02d" "$day")
      label="${YEAR}${mstr}${dstr}"
      outfile="${sat_daily}/${prefix}_${label}_${qc}.nc"
      if [[ -f "$outfile" ]]; then continue; fi

      # Only add regrid dependency if regrid was submitted for this year
      ts_dep=""
      if [[ "$regrid_needed" == true ]]; then
        ts_dep="done(Rg_${label})"
      fi
      submit_job "${Q_TS}" "Ts_${label}" "${GRP}" 150000 \
        "${ts_dep}" \
        "${LOGDIR}/ts_${label}.out" "1yuyan/intel-py:202508" \
        "cd ${TESS_DIR} && python3 -u ${tess_script} --year ${YEAR} --mon ${month} --day ${day}"
      n_ts=$((n_ts + 1))
    done
  done

  # ==== GCHP AGGREGATE monthly (depends on ALL regrid done) ====
  last_day="${YEAR}1231"
  rg_dep=""
  if [[ "$regrid_needed" == true ]]; then
    rg_dep="done(Rg_${last_day})"
  fi

  for month in $(seq 1 12); do
    mstr=$(printf "%02d" "$month")
    monthly_out="${BASE_FS}/gchp-v2/forObservation-Geophysical/${YEAR}/monthly/1x1km.Hours.13-15.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$monthly_out" ]]; then continue; fi
    submit_job "general" "Ag_${YEAR}${mstr}" "${GRP}" 150000 \
      "${rg_dep}" \
      "${LOGDIR}/ag_${YEAR}_${mstr}.out" "1yuyan/netcdf-mpi:latest" \
      "cd ${REGRID_CD} && python3 -u aggregate.py ${YEAR} --month ${month} --no-plot"
  done

  # GCHP yearly aggregate (depends on all monthly)
  submit_job "general" "AgY_${YEAR}" "${GRP}" 200000 \
    "done(Ag_${YEAR}12)" \
    "${LOGDIR}/ag_yearly_${YEAR}.out" "1yuyan/netcdf-mpi:latest" \
    "cd ${REGRID_CD} && python3 -u aggregate.py ${YEAR} --yearly-only --no-plot"

  # ==== NO2COL MONTHLY AVERAGE (depends on tess) ====
  last_tess="Ts_${last_day}"
  for month in $(seq 1 12); do
    mstr=$(printf "%02d" "$month")
    inst=$(get_instrument "$YEAR" "$month")
    if [[ "$inst" == "OMI" ]]; then av_script="omi_KNMI_average_v3.py"
    else av_script="tropomi_average_HP_v3.py"; fi

    submit_job "general" "Av_${YEAR}${mstr}" "${GRP}" 150000 \
      "done(${last_tess})" \
      "${LOGDIR}/av_${YEAR}_${mstr}.out" "1yuyan/netcdf-mpi:latest" \
      "cd ${AV_DIR} && python3 -u ${av_script} ${YEAR} --month ${month} --no-plot"
  done

  # NO2col yearly average (depends on monthly avg)
  for inst_tag in OMI TROPOMI; do
    inst_jan=$(get_instrument "$YEAR" 1)
    inst_dec=$(get_instrument "$YEAR" 12)
    if [[ "$inst_tag" != "$inst_jan" && "$inst_tag" != "$inst_dec" ]]; then continue; fi
    if [[ "$inst_tag" == "OMI" ]]; then av_script="omi_KNMI_average_v3.py"
    else av_script="tropomi_average_HP_v3.py"; fi

    submit_job "general" "AvY_${inst_tag:0:3}${YEAR}" "${GRP}" 300000 \
      "done(Av_${YEAR}12)" \
      "${LOGDIR}/av_yearly_${inst_tag:0:3}_${YEAR}.out" "1yuyan/netcdf-mpi:latest" \
      "cd ${AV_DIR} && python3 -u ${av_script} ${YEAR} --yearly-only --no-plot"
  done

  # ==== GEONO2 v5.13 --slim (depends on GCHP yearly + NO2col yearly) ====
  for month in $(seq 1 12); do
    mstr=$(printf "%02d" "$month")
    inst=$(get_instrument "$YEAR" "$month")
    outfile="${BASE_FS}/GeoNO2-v5.13/${YEAR}/1x1km.GeoNO2.${YEAR}${mstr}.MonMean.nc"
    if [[ -f "$outfile" ]]; then continue; fi

    submit_job "general" "Ge_${YEAR}${mstr}" "${GRP}" 200000 \
      "done(AgY_${YEAR}) && done(Av_${YEAR}12)" \
      "${LOGDIR}/ge_${YEAR}_${mstr}.out" "1yuyan/netcdf-mpi:latest" \
      "cd ${GEONO2_CD} && python3 -u geono2_v5.13.py ${YEAR} --month ${month} --instrument ${inst} --slim --no-plot"
  done

  # ==== SAVE NPY (depends on all GeoNO2) ====
  submit_job "general" "Np_${YEAR}" "${GRP}" 100000 \
    "done(Ge_${YEAR}12)" \
    "${LOGDIR}/npy_${YEAR}.out" "1yuyan/netcdf-mpi:latest" \
    "cd ${GEONO2_CD} && python3 -u save_npy_v5_clamp.py ${YEAR} --version v5.13"

  yr_total=$((n_rg + n_ts))
  TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + yr_total))
  log "  ${YEAR}: ${n_rg} regrid + ${n_ts} tess + agg/avg/geo/npy submitted"
done

# ==== DERIVE LABELS (depends on ALL years' npy) ====
all_npy_dep=""
for YEAR in $(seq 2007 2022); do
  if [[ -n "$all_npy_dep" ]]; then all_npy_dep="${all_npy_dep} && "; fi
  all_npy_dep="${all_npy_dep}done(Np_${YEAR})"
done

submit_job "general" "DeriveObs" "/yany1/derive" 200000 \
  "${all_npy_dep}" \
  "${LOGBASE}/derive_obs.out" "1yuyan/python-gfortran:latest" \
  "cd ${LABEL_CD} && python3 -u derive_observation_HP.py"

log "================================================================"
log "  Total submitted: ~${TOTAL_SUBMITTED} regrid+tess jobs + agg/avg/geo/npy per year"
log "  All years run in parallel with LSF dependencies"
log "  Queues: odd years regrid→general/tess→rvmartin, even years opposite"
log "================================================================"
