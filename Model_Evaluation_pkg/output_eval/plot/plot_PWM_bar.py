import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

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
    """Parse PWA Model from monthly sections for each region"""
    data = {}
    regions = []
    current_region = None
    current_period = None

    for raw in lines:
        line = raw.rstrip("\n")
        low = line.lower()

        # Region header - extract region name
        if line.startswith('Area:') or low.startswith('area'):
            parts = line.split(';', 1)
            region = parts[0].replace('Area:', '').strip() if parts else None
            # Skip Global and regions with 0 sites
            if region and region != 'Global' and ('site number: 0' not in low):
                current_region = region
                if region not in regions:
                    regions.append(region)
                    data.setdefault(current_region, {
                        'Months': {m: {'PWA Model': np.nan} for m in months}
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

        # Monthly lines - only extract PWA Model
        if current_period in months:
            key = classify_metric(line)
            if key == 'PWA Model':
                v = grab_after_allpoints(line)
                if not (isinstance(v, float) and np.isnan(v)):
                    data[current_region]['Months'][current_period][key] = v

    return regions, data

# ── Aggregate regional data across years ───────────────────────────────────
regional_monthly_data = defaultdict(lambda: [])
dates = []

all_regions = set()

for year in years:
    csv_file = os.path.join(
        InDir,
        f'AVDSpatialCV_{year}-{year}_Normaized-NO2_NO2_{version}_27Channel_5x5_BenchMark.csv'
    )
    
    if os.path.isfile(csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        regions, region_data = parse_metrics_from_lines(lines)
        all_regions.update(regions)
        
        for month in months:
            # Add date for each year-month combination
            date_str = f"{year}-{months.index(month)+1:02d}"
            dates.append(datetime.strptime(date_str, '%Y-%m'))
            
            # Store each region's PWA Model for this month
            for r in all_regions:
                if r in regions:
                    pwa_val = region_data[r]['Months'][month]['PWA Model']
                    regional_monthly_data[r].append(pwa_val if not np.isnan(pwa_val) else 0)
                else:
                    regional_monthly_data[r].append(0)

# ── Calculate total contribution per region ────────────────────────────────
region_totals = {}
for region, values in regional_monthly_data.items():
    region_totals[region] = np.sum(values)

# Sort regions by total contribution (descending)
sorted_regions = sorted(region_totals.items(), key=lambda x: x[1], reverse=True)
region_order = [r[0] for r in sorted_regions]

# ── Assign colors to regions ───────────────────────────────────────────────
color_palette = plt.cm.tab10.colors  # Use tab10 colormap
region_colors = {region: color_palette[i % len(color_palette)] 
                 for i, region in enumerate(region_order)}

# ── Prepare data for stacked bar plot ──────────────────────────────────────
n_months = len(dates)
regional_data_matrix = np.zeros((len(region_order), n_months))

for i, region in enumerate(region_order):
    regional_data_matrix[i, :] = regional_monthly_data[region]

# ── Create stacked bar plot ────────────────────────────────────────────────
plt.rcParams.update({
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
})

fig, ax = plt.subplots(figsize=(14, 6))

# X positions for bars
x_pos = np.arange(n_months)
bar_width = 0.8

# Create stacked bars
bottom = np.zeros(n_months)
bars = []
for i, region in enumerate(region_order):
    bar = ax.bar(x_pos, regional_data_matrix[i, :], bar_width, 
                 bottom=bottom, label=region, color=region_colors[region],
                 edgecolor='white', linewidth=0.5)
    bars.append(bar)
    bottom += regional_data_matrix[i, :]

# Format x-axis with dates
date_labels = [d.strftime('%Y-%m') for d in dates]
ax.set_xticks(x_pos[::3])  # Show every 3rd label to avoid crowding
ax.set_xticklabels(date_labels[::3], rotation=45, ha='right')

# Labels and styling
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('PWA Model (ppb)', fontsize=13)
# ax.set_title('Regional Contribution to Global PWA Model (Sorted by Total Contribution)', 
#              fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), 
          frameon=True, ncol=1)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# Save figure
out_path = os.path.join(FigDir, f'Regional_Contribution_PWA_Model_{years[0]}-{years[-1]}.png')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f"Figure saved to: {out_path}")
print(f"\nRegional contributions (sorted by total):")
for region, total in sorted_regions:
    print(f"  {region}: {total:.2f} ppb")