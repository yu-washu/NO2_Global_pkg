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

GROUP="/yany1/gchp-monthly"
# if it exists, modify; otherwise add
bgmod -L "${TOTAL_FREE}" "${GROUP}" 2>/dev/null || bgadd -L "${TOTAL_FREE}" "${GROUP}"
echo "Using job‐group ${GROUP} (limit=${TOTAL_FREE})"

# —————————————————————————————————————————————————————————————
# 3) Loop over each month and submit one job per month
# —————————————————————————————————————————————————————————————
START="2023-01"
END="2023-12"
MEM=120000          # in MB (≈150 GB)
SCRIPT="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Regrid_GCHP/aggregate.py"
LOGDIR="logs_av"; mkdir -p "$LOGDIR"

d="${START}-01"           # normalize to a full date
end="${END}-01"

while [[ "$d" < "$end" ]] || [[ "$d" == "$end" ]]; do
  DATE=$(date -d "$d" +%Y%m)        # e.g., 201801
  YEAR=${DATE:0:4}
  MON=${DATE:4:2}

  echo "DEBUG: submitting Regrid${DATE}"
  bsub -q rvmartin \
       -J "av${DATE}" \
       -g "$GROUP" \
       -n 1 \
       -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin \
       -R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" \
       -R "span[ptile=${PTILE}]" \
       -R "select[port8543=1]" \
       -R "rusage[mem=${MEM}]" \
       -a "docker(1yuyan/netcdf-mpi:latest)" \
       bash -lc $'\n'"\
. /opt/conda/bin/activate && \
/bin/bash && \
ulimit -s unlimited && \
exec >\"${LOGDIR}/av_${DATE}.out\" 2>&1
echo \"[\$(hostname)] Regridding ${DATE}\"
python3 -u ${SCRIPT} ${DATE:0:4} --month ${DATE:4:2}
"

  d=$(date -I -d "$d + 1 month")
done

echo "Submitted days $START → $END into group $GROUP (limit=$TOTAL_FREE)."