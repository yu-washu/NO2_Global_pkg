#!/usr/bin/env python3
"""
monitor_and_plot.py — watch GeoNO2 log files and submit --plot-only bsub jobs
whenever a month completes successfully (NetCDF saved) but the plot is missing
or failed.

PID 930001

Usage:
    python3 monitor_and_plot.py            # run once
    python3 monitor_and_plot.py --loop 300 # poll every 300 s
    nohup python3 monitor_and_plot.py --loop 600 > logs/monitor.out 2>&1 &
"""
import os
import re
import subprocess
import argparse
import time
from glob import glob

# ── Paths ────────────────────────────────────────────────────────────────────
LOG_DIR    = 'logs'
SCRIPT     = ('/my-projects2/1.project/NO2_DL_global/NO2_global_pkg'
              '/Data_Processing/Derive_Geophysical_NO2/GeoNO2/geono2_v5.py')
SUBMITTED  = os.path.join(LOG_DIR, 'plot_only_submitted.txt')   # tracks already-submitted
PLOT_MEM   = 50000   # MB — plots need far less RAM than the full pipeline
QUEUE      = 'rvmartin'
GROUP      = '/yany1/GeoNO2'
PTILE      = 48

# ── Patterns to search in log ────────────────────────────────────────────────
RE_NC_SAVED   = re.compile(r'Saved: .+\.nc')
RE_PLOT_SAVED = re.compile(r'Saved plot:')
RE_PLOT_WARN  = re.compile(r'\[WARN\] Plot failed')
RE_DONE       = re.compile(r'(End time:|Completed:)')
RE_OOM        = re.compile(r'Killed\s+python3')
RE_INSTRUMENT = re.compile(r'instrument=(\S+)')


def read_log(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ''


def classify(log_text):
    """Return (done, nc_saved, plot_ok, oom)."""
    done     = bool(RE_DONE.search(log_text))
    nc_saved = bool(RE_NC_SAVED.search(log_text))
    plot_ok  = bool(RE_PLOT_SAVED.search(log_text))
    oom      = bool(RE_OOM.search(log_text))
    return done, nc_saved, plot_ok, oom


def load_submitted():
    if not os.path.exists(SUBMITTED):
        return set()
    with open(SUBMITTED) as f:
        return {line.strip() for line in f if line.strip()}


def mark_submitted(key):
    with open(SUBMITTED, 'a') as f:
        f.write(key + '\n')


def submit_plot_only(year, month, instrument):
    month_str = f'{month:02d}'
    job_name  = f'Plot_{year}{month_str}'
    log_path  = os.path.join(LOG_DIR, f'Plot_{year}_{month_str}.out')

    cmd = (
        f'bsub -q {QUEUE} '
        f'-J {job_name} '
        f'-g {GROUP} '
        f'-n 1 '
        f'-W 60:00 '
        f'-u yany1@wustl.edu -G compute-rvmartin '
        f'-R "select[model==Intel_Xeon_Gold6154CPU300GHz||model==Intel_Xeon_Gold6242CPU280GHz]" '
        f'-R "select[port8543=1]" '
        f'-R "span[ptile={PTILE}]" '
        f'-R "rusage[mem={PLOT_MEM}]" '
        f'-a "docker(1yuyan/netcdf-mpi:latest)" '
        f'-o {log_path} '
        f'bash -lc \'\n'
        f'. /opt/conda/bin/activate && /bin/bash && '
        f'cd /my-projects2/1.project/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/GeoNO2 && '
        f'python3 -u {SCRIPT} {year} --month {month} --instrument {instrument} --plot-only\''
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f'  [SUBMITTED] plot-only job {job_name}: {result.stdout.strip()}')
    else:
        print(f'  [ERROR] bsub failed for {job_name}: {result.stderr.strip()}')
    return result.returncode == 0


def scan_once(dry_run=False):
    submitted = load_submitted()
    log_files = sorted(glob(os.path.join(LOG_DIR, 'GeoNO2_????_??.out')))

    if not log_files:
        print('No GeoNO2 log files found.')
        return

    for log_path in log_files:
        fname = os.path.basename(log_path)
        m = re.match(r'GeoNO2_(\d{4})_(\d{2})\.out', fname)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        key = f'{year}_{month:02d}'

        text = read_log(log_path)
        done, nc_saved, plot_ok, oom = classify(text)

        # detect instrument from log (default 'both')
        im = RE_INSTRUMENT.search(text)
        instrument = im.group(1) if im else 'both'

        status = ('done' if done else 'running') + \
                 (' | nc=OK' if nc_saved else ' | nc=MISSING') + \
                 (' | plot=OK' if plot_ok else ' | plot=MISSING') + \
                 (' | *** OOM KILLED ***' if oom else '')
        print(f'{fname}: {status}')

        needs_plot = done and nc_saved and not plot_ok
        if not needs_plot:
            continue
        if key in submitted:
            print(f'  → plot-only already submitted, skipping')
            continue

        if dry_run:
            print(f'  → [DRY RUN] would submit plot-only for {year}-{month:02d}')
        else:
            if submit_plot_only(year, month, instrument):
                mark_submitted(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', type=int, default=0,
                        help='Poll interval in seconds (0 = run once)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be submitted without actually submitting')
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    if args.loop > 0:
        print(f'Polling every {args.loop}s. Ctrl-C to stop.')
        while True:
            print(f'\n─── {time.strftime("%Y-%m-%d %H:%M:%S")} ───')
            scan_once(dry_run=args.dry_run)
            time.sleep(args.loop)
    else:
        scan_once(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
