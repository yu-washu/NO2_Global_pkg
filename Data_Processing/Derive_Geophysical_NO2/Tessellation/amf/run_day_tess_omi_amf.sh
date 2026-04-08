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

GROUP="/yany1/tess-omi-amf"
# if it exists, modify; otherwise add
# bgmod -L "${TOTAL_FREE}" "${GROUP}" 2>/dev/null || bgadd -L "${TOTAL_FREE}" "${GROUP}"
# echo "Using job‐group ${GROUP} (limit=${TOTAL_FREE})"

# —————————————————————————————————————————————————————————————
# 3) Loop over each date and submit one job per day
# —————————————————————————————————————————————————————————————
START="2005-01-01"
END="2005-01-31"
MEM=150000          # in MB (≈50 GB)
SCRIPT="/my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/amf/tess_OMI_KNMI_AMF_v2.py"
LOGDIR="logs"; mkdir -p "$LOGDIR"

d="$START"
while [[ "$d" < "$END" || "$d" == "$END" ]]; do
  DATE=$(date -d "$d" +%Y%m%d)

  echo "DEBUG: submitting Tessellation_AMF${DATE}"
  bsub -q rvmartin \
       -J "Tess_AMF${DATE}" \
       -g "$GROUP" \
       -n 1 \
       -W 499:00 \
       -u yany1@wustl.edu -G compute-rvmartin \
       -R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" \
       -R "select[hname!='compute1-exec-54.ris.wustl.edu']" \
       -R "select[hname!='compute1-exec-72.ris.wustl.edu']" \
       -R "select[hname!='compute1-exec-164.ris.wustl.edu']" \
       -R "select[port8543=1]" \
       -R "span[ptile=${PTILE}]" \
       -R "rusage[mem=${MEM}]" \
       -a "docker(1yuyan/intel-py:202508)" \
       bash -lc $'\n'"\
. /opt/conda/bin/activate && \
/bin/bash && \
ulimit -s unlimited && \
exec >\"${LOGDIR}/Tess_AMF_${DATE}.out\" 2>&1
echo \"[\$(hostname)] Tessellating ${DATE}\"
python3 -u ${SCRIPT} --year ${DATE:0:4} --mon ${DATE:4:2} --day ${DATE:6:2}
"

  d=$(date -I -d "$d + 1 day")
done

echo "Submitted days $START → $END into group $GROUP (limit=$TOTAL_FREE)."
