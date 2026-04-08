import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import matplotlib.dates as mdates
%matplotlib inline
# --- Robust extractors ------------------------------------------------------
def grab_after_annual_avg(line):
    m = re.search(r"Annual\s*Average\s*:\s*,\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
                  line, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def grab_after_allpoints(line):
    m = re.search(r"AllPoints\s*:\s*,\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
                  line, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def classify_metric(line):
    if re.search(r"\bPWA\s*Monitors\b", line, re.IGNORECASE):
        return "PWA Monitors"
    if re.search(r"\bPWA\s*Model\b", line, re.IGNORECASE):
        return "PWA Model"
    return None

# ── Config ─────────────────────────────────────────────────────────────────
version = 'LightGBM_1003'
InDir = f'/storage1/fs1/rvmartin2/Active/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/Results/results-SpatialCV/statistical_indicators/Normaized-NO2_NO2_{version}_27Channel_5x5_BenchMark/'
FigDir = f'/storage1/fs1/rvmartin2/Active/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/Figures/figures-PWM_TimeSerires/'
os.makedirs(FigDir, exist_ok=True)

years = [2019, 2020, 2021, 2022, 2023]
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Utils ───────────────────────────────────────────────────────────────────
def detect_period(line):
    s = line.strip()
    if "Annual" in s:
        return "Annual"
    for m in months:
        if re.search(rf'\b{m}\b', s):
            return m
    return None

def parse_metrics_from_lines(lines):
    """Parse PWA Monitors and PWA Model from monthly sections"""
    data = {}
    regions = []
    current_region = None
    current_period = None

    for raw in lines:
        line = raw.rstrip("\n")
        low = line.lower()

        # Region header
        if line.startswith('Area:') or low.startswith('area'):
            parts = line.split(';', 1)
            region = parts[0].replace('Area:', '').strip() if parts else None
            if region and ('site number: 0' not in low):
                current_region = region
                regions.append(region)
                data.setdefault(current_region, {
                    'Months': {m: {'PWA Monitors': np.nan, 'PWA Model': np.nan} for m in months}
                })
            else:
                current_region = None
            continue

        # Section header
        if '--------------------------' in line:
            sec = detect_period(line)
            if sec is not None:
                current_period = sec
            continue

        if current_region is None or current_period is None:
            continue

        # Monthly lines
        if current_period in months:
            key = classify_metric(line)
            if key:
                v = grab_after_allpoints(line)
                if not (isinstance(v, float) and np.isnan(v)):
                    data[current_region]['Months'][current_period][key] = v

    return regions, data

# ── Aggregate time series data ──────────────────────────────────────────────
# Initialize storage
global_monitors = []
global_model = []
dates = []

# Process each year and month sequentially
for year in years:
    csv_file = os.path.join(
        InDir,
        f'AVDSpatialCV_{year}-{year}_Normaized-NO2_NO2_{version}_27Channel_5x5_BenchMark.csv'
    )
    
    if os.path.isfile(csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        regions, region_data = parse_metrics_from_lines(lines)
        
        for month in months:
            # Add date
            date_str = f"{year}-{months.index(month)+1:02d}"
            dates.append(datetime.strptime(date_str, '%Y-%m'))
            
            # Aggregate across all regions for Global
            monitors_vals = []
            model_vals = []
            
            for r in regions:
                mon_val = region_data[r]['Months'][month]['PWA Monitors']
                mod_val = region_data[r]['Months'][month]['PWA Model']
                if not np.isnan(mon_val):
                    monitors_vals.append(mon_val)
                if not np.isnan(mod_val):
                    model_vals.append(mod_val)
            
            # Store global average for this month
            global_monitors.append(np.nanmean(monitors_vals) if monitors_vals else np.nan)
            global_model.append(np.nanmean(model_vals) if model_vals else np.nan)
    else:
        # If file doesn't exist, add NaN for all 12 months
        for month in months:
            date_str = f"{year}-{months.index(month)+1:02d}"
            dates.append(datetime.strptime(date_str, '%Y-%m'))
            global_monitors.append(np.nan)
            global_model.append(np.nan)

# ── Plotting ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

fig, ax = plt.subplots(figsize=(12, 6))

# Convert to numpy arrays
monitors = np.array(global_monitors)
model = np.array(global_model)

# Plot time series
ax.plot(dates, monitors, 'o-', color='blue', linewidth=2, markersize=4, 
        label='PWA Monitors', alpha=0.8)
ax.plot(dates, model, 's-', color='red', linewidth=2, markersize=4, 
        label='PWA Model', alpha=0.8)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45, ha='right')

# Labels and styling
ax.set_yticks([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('PWA Value (ppb)', fontsize=14)
ax.set_title('Time Series: PWA Monitors vs PWA Model (Global)', fontsize=16, fontweight='bold')
ax.legend(loc='best', frameon=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Save figure
out_path = os.path.join(FigDir, f'PWA_TimeSeries_{years[0]}-{years[-1]}_Global.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
# plt.close(fig)

print(f"Figure saved to: {out_path}")
print(f"Total months plotted: {len(dates)}")
print(f"Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")