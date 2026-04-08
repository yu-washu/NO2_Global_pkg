# --- COMPLETE SCRIPT (R2 on left y-axis, RMSE on right) -------------------
import os
import re
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from collections import defaultdict
import re
import matplotlib.colors as mcolors

# --- robust extractors ------------------------------------------------------
def grab_after_annual_avg(line):
    # number immediately after "Annual Average:"
    m = re.search(r"Annual\s*Average\s*:\s*,\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
                  line, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def grab_after_allpoints_r2(line):
    # "AllPoints Test R2 - AllPoints: ,<val>"
    m = re.search(r"AllPoints\s+Test\s+R\s*2\s*-\s*AllPoints\s*:\s*,\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
                  line, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def grab_after_allpoints_rmse(line):
    # "AllPoints RMSE - AllPoints: ,<val>"
    m = re.search(r"AllPoints\s+RMSE\s*-\s*AllPoints\s*:\s*,\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
                  line, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def grab_after_allpoints_geo_r2(line):
    # "AllPoints Geophysical R2 - AllPoints: ,<val>"
    m = re.search(r"AllPoints\s+Geophysical\s+R\s*2\s*-\s*AllPoints\s*:\s*,\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
                  line, re.IGNORECASE)
    return float(m.group(1)) if m else np.nan

def classify_metric(line):
    # figure out which metric this line refers to
    if re.search(r"Test\s*R\s*(?:\^?2|²|2)\b", line, re.IGNORECASE):
        return "TEST_R2"
    if re.search(r"Geophysical\s*R\s*(?:\^?2|²|2)\b", line, re.IGNORECASE):
        return "GEO_R2"
    if re.search(r"\bRMSE\b", line, re.IGNORECASE):
        return "RMSE"
    return None

# ── Config ─────────────────────────────────────────────────────────────────
version = 'v4.1.0'
special_version = '_Geov5131-abs'
year='2023'

InDir = f'/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/Results/results-SelfIsolated_BLCOCV/statistical_indicators/'
FigDir = f'/fsx/yany1/1.project/NO2_DL_global/Training_Evaluation_Estimation/NO2/{version}/Figures/figures-SelfIsolated_BLCO_plot/'
os.makedirs(FigDir, exist_ok=True)
Buffers = np.array([0, 10, 20, 30, 40, 50, 100, 150, 200])
# Buffers = Buffers[Buffers!=460]
months  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasons = ['MAM','JJA','SON','DJF']  # not used for stats, only detection

# ── Utils ───────────────────────────────────────────────────────────────────
def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def detect_period(line):
    s = line.strip()
    if "Annual" in s:
        return "Annual"
    for m in months:
        if re.search(rf'\b{m}\b', s):
            return m
    for ss in seasons:
        if re.search(rf'\b{ss}\b', s):
            return ss
    return None

def parse_metrics_from_lines(lines):
    """
    Robust Annual + Monthly parser:
      - Annual: classify metric by keywords; pull number after 'Annual Average:'
      - Monthly: classify metric by keywords; pull number after 'AllPoints:'
    """
    data = {}
    regions = []
    current_region = None
    current_period = None

    for raw in lines:
        line = raw.rstrip("\n")
        low  = line.lower()

        # Region header
        if line.startswith('Area:') or low.startswith('area'):
            parts  = line.split(';', 1)
            region = parts[0].replace('Area:', '').strip() if parts else None
            if region and ('site number: 0' not in low):
                current_region = region
                regions.append(region)
                data.setdefault(region, {
                    'Annual': {'TEST_R2': np.nan, 'GEO_R2': np.nan, 'RMSE': np.nan},
                    'Months': {m: {'TEST_R2': np.nan, 'GEO_R2': np.nan, 'RMSE': np.nan} for m in months}
                })
            else:
                current_region = None
            continue

        # Section header (Annual / months / seasons)
        if '--------------------------' in line:
            sec = detect_period(line)
            if sec is not None:
                current_period = sec
            continue

        if current_region is None or current_period is None:
            continue

        # Annual lines: pick the right extractor per metric
        if current_period == 'Annual':
            key = classify_metric(line)
            if key:
                if key == 'TEST_R2':
                    v = grab_after_allpoints_r2(line)       # "AllPoints Test R2 - AllPoints:"
                elif key == 'RMSE':
                    v = grab_after_allpoints_rmse(line)     # "AllPoints RMSE - AllPoints:"
                elif key == 'GEO_R2':
                    if current_region == 'Global':
                        v = grab_after_annual_avg(line)     # Global: "… - Annual Average:"
                    else:
                        v = grab_after_allpoints_geo_r2(line)  # Regions: "… - AllPoints:"
                if not (isinstance(v, float) and np.isnan(v)):
                    data[current_region]['Annual'][key] = v

        # Monthly lines: "… AllPoints <metric> - AllPoints: ,<val>, …"
        elif current_period in months:
            key = classify_metric(line)
            if key:
                if key == 'TEST_R2':
                    v = grab_after_allpoints_r2(line)
                elif key == 'RMSE':
                    v = grab_after_allpoints_rmse(line)
                elif key == 'GEO_R2':
                    v = grab_after_allpoints_geo_r2(line)
                if not (isinstance(v, float) and np.isnan(v)):
                    data[current_region]['Months'][current_period][key] = v

    return regions, data

def nanmean(arr): return np.nan if len(arr) == 0 else np.nanmean(arr)
def nanstd(arr):  return np.nan if len(arr) == 0 else np.nanstd(arr, ddof=1)

def lighten_color(color, amount=0.5):
    """Lighten a color by mixing it with white."""
    try:
        c = mcolors.to_rgb(color)
        return tuple(c_val + (1 - c_val) * amount for c_val in c)
    except:
        return color

# ── Aggregate per-region series across buffers ──────────────────────────────
# Now we need separate storage for w/Geo and w/oGeo results
region_series_wGeo = defaultdict(lambda: defaultdict(lambda: {'annual': [], 'std': []}))
region_series_woGeo = defaultdict(lambda: defaultdict(lambda: {'annual': [], 'std': []}))
all_regions_seen = set()

# Global results for both w/Geo and w/oGeo
global_annual_wGeo = {'TEST_R2': [], 'GEO_R2': [], 'RMSE': []}
global_month_std_wGeo = {'TEST_R2': [], 'GEO_R2': [], 'RMSE': []}
global_annual_woGeo = {'TEST_R2': [], 'GEO_R2': [], 'RMSE': []}
global_month_std_woGeo = {'TEST_R2': [], 'GEO_R2': [], 'RMSE': []}

for i in Buffers:
    # w/Geo file (26 channels)
    csv_file = os.path.join(
        InDir,
        f'{i}km-10fold-5ClusterSeeds-SpatialCV_Absolute-NO2_NO2_{version}_26Channel_5x5{special_version}',
        f'SelfIsolated_BLCO-{year}-{year}_{i}km-10fold-5ClusterSeeds-SpatialCV_Absolute-NO2_NO2_{version}_26Channel_5x5{special_version}.csv'
    )


    # w/oGeo file (24 channels)
    csv_file_woGeo = os.path.join(
        InDir,
        f'{i}km-10fold-5ClusterSeeds-SpatialCV_Absolute-NO2_NO2_{version}_24Channel_5x5{special_version}',
        f'SelfIsolated_BLCO-{year}-{year}_{i}km-10fold-5ClusterSeeds-SpatialCV_Absolute-NO2_NO2_{version}_24Channel_5x5{special_version}.csv'
    )

    # Process w/Geo file
    if os.path.isfile(csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        regions_wGeo, region_data_wGeo = parse_metrics_from_lines(lines)
        all_regions_seen.update(regions_wGeo)
        
        # Process regions for w/Geo
        monthly_stack_wGeo = {k: [[] for _ in range(12)] for k in ('TEST_R2','GEO_R2','RMSE')}
        for r in regions_wGeo:
            ann = region_data_wGeo[r]['Annual']
            mons_TEST = [region_data_wGeo[r]['Months'][m]['TEST_R2'] for m in months]
            mons_GEO = [region_data_wGeo[r]['Months'][m]['GEO_R2'] for m in months]
            mons_RMSE = [region_data_wGeo[r]['Months'][m]['RMSE'] for m in months]

            for j in range(12):
                monthly_stack_wGeo['TEST_R2'][j].append(mons_TEST[j])
                monthly_stack_wGeo['GEO_R2'][j].append(mons_GEO[j])
                monthly_stack_wGeo['RMSE'][j].append(mons_RMSE[j])

            region_series_wGeo[r]['TEST_R2']['annual'].append(ann['TEST_R2'])
            region_series_wGeo[r]['TEST_R2']['std'].append(nanstd(mons_TEST))
            region_series_wGeo[r]['GEO_R2']['annual'].append(ann['GEO_R2'])
            region_series_wGeo[r]['GEO_R2']['std'].append(nanstd(mons_GEO))
            region_series_wGeo[r]['RMSE']['annual'].append(ann['RMSE'])
            region_series_wGeo[r]['RMSE']['std'].append(nanstd(mons_RMSE))

        # Global aggregation for w/Geo
        for k in ('TEST_R2','GEO_R2','RMSE'):
            ann_vals = [region_series_wGeo[r][k]['annual'][-1] for r in regions_wGeo]
            global_annual_wGeo[k].append(nanmean(ann_vals))
            month_means = [nanmean(monthly_stack_wGeo[k][j]) for j in range(12)]
            global_month_std_wGeo[k].append(nanstd(month_means))
    else:
        # Pad with NaNs for missing w/Geo file
        for r in all_regions_seen:
            for k in ('TEST_R2','GEO_R2','RMSE'):
                region_series_wGeo[r][k]['annual'].append(np.nan)
                region_series_wGeo[r][k]['std'].append(np.nan)
        for k in ('TEST_R2','GEO_R2','RMSE'):
            global_annual_wGeo[k].append(np.nan)
            global_month_std_wGeo[k].append(np.nan)

    # Process w/oGeo file
    if os.path.isfile(csv_file_woGeo):
        with open(csv_file_woGeo, 'r') as f:
            lines = f.readlines()
        regions_woGeo, region_data_woGeo = parse_metrics_from_lines(lines)
        all_regions_seen.update(regions_woGeo)
        
        # Process regions for w/oGeo
        monthly_stack_woGeo = {k: [[] for _ in range(12)] for k in ('TEST_R2','GEO_R2','RMSE')}
        for r in regions_woGeo:
            ann = region_data_woGeo[r]['Annual']
            mons_TEST = [region_data_woGeo[r]['Months'][m]['TEST_R2'] for m in months]
            mons_GEO = [region_data_woGeo[r]['Months'][m]['GEO_R2'] for m in months]
            mons_RMSE = [region_data_woGeo[r]['Months'][m]['RMSE'] for m in months]

            for j in range(12):
                monthly_stack_woGeo['TEST_R2'][j].append(mons_TEST[j])
                monthly_stack_woGeo['GEO_R2'][j].append(mons_GEO[j])
                monthly_stack_woGeo['RMSE'][j].append(mons_RMSE[j])

            region_series_woGeo[r]['TEST_R2']['annual'].append(ann['TEST_R2'])
            region_series_woGeo[r]['TEST_R2']['std'].append(nanstd(mons_TEST))
            region_series_woGeo[r]['GEO_R2']['annual'].append(ann['GEO_R2'])
            region_series_woGeo[r]['GEO_R2']['std'].append(nanstd(mons_GEO))
            region_series_woGeo[r]['RMSE']['annual'].append(ann['RMSE'])
            region_series_woGeo[r]['RMSE']['std'].append(nanstd(mons_RMSE))

        # Global aggregation for w/oGeo
        for k in ('TEST_R2','GEO_R2','RMSE'):
            ann_vals = [region_series_woGeo[r][k]['annual'][-1] for r in regions_woGeo]
            global_annual_woGeo[k].append(nanmean(ann_vals))
            month_means = [nanmean(monthly_stack_woGeo[k][j]) for j in range(12)]
            global_month_std_woGeo[k].append(nanstd(month_means))
    else:
        # Pad with NaNs for missing w/oGeo file
        for r in all_regions_seen:
            for k in ('TEST_R2','GEO_R2','RMSE'):
                region_series_woGeo[r][k]['annual'].append(np.nan)
                region_series_woGeo[r][k]['std'].append(np.nan)
        for k in ('TEST_R2','GEO_R2','RMSE'):
            global_annual_woGeo[k].append(np.nan)
            global_month_std_woGeo[k].append(np.nan)

    # Handle missing regions for alignment
    missing_regions_wGeo = all_regions_seen.difference(regions_wGeo if os.path.isfile(csv_file) else set())
    for r in missing_regions_wGeo:
        for k in ('TEST_R2','GEO_R2','RMSE'):
            if len(region_series_wGeo[r][k]['annual']) < len(Buffers[:i+1]):
                region_series_wGeo[r][k]['annual'].append(np.nan)
                region_series_wGeo[r][k]['std'].append(np.nan)
    
    missing_regions_woGeo = all_regions_seen.difference(regions_woGeo if os.path.isfile(csv_file_woGeo) else set())
    for r in missing_regions_woGeo:
        for k in ('TEST_R2','GEO_R2','RMSE'):
            if len(region_series_woGeo[r][k]['annual']) < len(Buffers[:i+1]):
                region_series_woGeo[r][k]['annual'].append(np.nan)
                region_series_woGeo[r][k]['std'].append(np.nan)

# --- Plot Setup ---
REGION_ORDER = [
    'Global',
    'Asia',
    'Europe',
    'North_America',
    'Africa',
    'South_America',
    'Oceania_Australia'
]

colors_dict = {
    'Global': 'black',
    'Asia': '#e02b35',
    'Africa': '#f0c571',
    'Europe': '#59a89c',
    'North_America': '#082a54',
    'Oceania_Australia': '#a559aa',
    'South_America': '#666666'
}

plt.rcParams.update({
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 13,
    "legend.title_fontsize": 16,
})

fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=False)
fig.subplots_adjust(left=0.10, right=0.60, bottom=0.14, top=0.94)

idx = np.array([np.where(Buffers == buffer)[0][0] for buffer in Buffers])

# region_list = ['Asia','Global', 'Europe', 'North_America']
region_list = ['Asia','Global', 'North_America', 'Europe']
plot_regions = [r for r in region_list if r != 'Global']

fallback = plt.cm.tab20.colors
region_colors = {r: colors_dict.get(r, fallback[i % len(fallback)])
                 for i, r in enumerate(plot_regions)}

# ================== DUMBBELL (R2 top, RMSE bottom) + GLOBAL SEPARATE ==================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- config ----
key_distances = np.array([0, 10, 20, 30, 40, 50, 100, 150, 200])

regions_all = [
    'Global',
    'Asia',
    'Europe',
    'North_America',
    'Africa',
    'South_America',
    'Oceania_Australia'
]
regions_non_global = [r for r in regions_all if r != 'Global']

# Safety: keep only distances that exist in Buffers
buf_set = set(Buffers.tolist())
key_distances = [d for d in key_distances if d in buf_set]
if len(key_distances) < 2:
    raise ValueError("Not enough key_distances found in Buffers.")

dist_idx = np.array([np.where(Buffers == d)[0][0] for d in key_distances], dtype=int)

# Colors
fallback = plt.cm.tab20.colors
region_colors = {r: colors_dict.get(r, fallback[i % len(fallback)])
                 for i, r in enumerate(regions_all)}

# Dumbbell geometry
xpos = np.arange(len(key_distances), dtype=float)
dx = 0.16
band_halfwidth = 0.11
alpha_band = 0.12

def draw_pill_band(ax, xcenter, y, s, color, alpha=0.12, z=1):
    if np.isfinite(y) and np.isfinite(s) and (s > 0):
        ax.fill_betweenx([y - s, y + s],
                         xcenter - band_halfwidth, xcenter + band_halfwidth,
                         color=color, alpha=alpha, zorder=z)

def plot_metric_dumbbell(ax, region, metric, marker_wo, marker_w, color, z_base=10):
    """
    metric in {"TEST_R2","RMSE"} from your region_series_* dictionaries
    """
    y_wo = np.asarray(region_series_woGeo[region][metric]['annual'], float)[dist_idx]
    y_w  = np.asarray(region_series_wGeo[region][metric]['annual'], float)[dist_idx]
    s_wo = np.asarray(region_series_woGeo[region][metric]['std'], float)[dist_idx]
    s_w  = np.asarray(region_series_wGeo[region][metric]['std'], float)[dist_idx]

    # local shaded pills
    for i in range(len(key_distances)):
        draw_pill_band(ax, xpos[i] - dx, y_wo[i], s_wo[i], color, alpha=alpha_band, z=z_base-5)
        draw_pill_band(ax, xpos[i] + dx, y_w[i],  s_w[i],  color, alpha=alpha_band, z=z_base-5)

    # connectors
    for i in range(len(key_distances)):
        if np.isfinite(y_wo[i]) and np.isfinite(y_w[i]):
            ax.plot([xpos[i]-dx, xpos[i]+dx], [y_wo[i], y_w[i]],
                    lw=1.1, color=color, alpha=0.9, zorder=z_base)

    # points
    ax.plot(xpos - dx, y_wo, linestyle="None", marker=marker_wo, markersize=5.0,
            markerfacecolor=color, markeredgecolor=color, zorder=z_base+2)
    ax.plot(xpos + dx, y_w,  linestyle="None", marker=marker_w, markersize=8.0,
            markerfacecolor=color, markeredgecolor=color, zorder=z_base+3)

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(d) for d in key_distances])

def plot_geo_baseline(ax, region, color, z=12):
    """
    Plot geophysical baseline R² as dashed line (no markers, no band).
    Uses region_series_wGeo[region]['GEO_R2']['annual'] at the same dist_idx.
    """
    y_geo = np.asarray(region_series_wGeo[region]['GEO_R2']['annual'], float)[dist_idx]

    ax.plot(
        xpos, y_geo,
        ls="--", lw=1.4,
        color=color, alpha=0.9,
        zorder=z
    )

# ---------------- FIGURE 1: GLOBAL ONLY ----------------
plt.rcParams.update({
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})

fig1, (ax1_top, ax1_bot) = plt.subplots(
    2, 1, figsize=(8.5, 5.8), sharex=True
)

gc = region_colors['Global']

# Top: R2
plot_metric_dumbbell(ax1_top, 'Global', 'TEST_R2', marker_wo='o', marker_w='*', color=gc, z_base=30)
plot_geo_baseline(ax1_top, 'Global', color=gc, z=15)
ax1_top.set_title("Global", fontsize=14)
ax1_top.set_ylabel(r"$R^2$")
ax1_top.grid(True, ls="--", alpha=0.25)

# Bottom: RMSE
plot_metric_dumbbell(ax1_bot, 'Global', 'RMSE', marker_wo='s', marker_w='D', color=gc, z_base=30)
ax1_bot.set_ylabel("RMSE")
ax1_bot.set_xlabel("Buffer distance (km)")
ax1_bot.grid(True, ls="--", alpha=0.25)

# Legend (style-only)
legend_handles = [
    Line2D([0],[0], marker="*", color="k", linestyle="None", markersize=9, label="w/Geo"),
    Line2D([0],[0], marker="o", color="k", linestyle="None", markersize=6, label="w/oGeo"),
    Line2D([0],[0], marker="D", color="k", linestyle="None", markersize=7, label="w/Geo (RMSE marker)"),
    Line2D([0],[0], marker="s", color="k", linestyle="None", markersize=6, label="w/oGeo (RMSE marker)"),
    Line2D([0],[0], color="k", lw=1.2, label="Connector = Geo effect"),
]
legend_handles.append(
    Line2D([0],[0], color="k", lw=1.4, ls="--", label="Geophysical baseline (R²)")
)

fig1.tight_layout(rect=[0, 0.14, 1, 1])
fig1.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True)

out_path1 = os.path.join(FigDir, f"{year}_Global_Dumbbell_R2_top_RMSE_bottom_0-50_step5.png")
fig1.savefig(out_path1, dpi=500, bbox_inches="tight")
plt.close(fig1)


# ---------------- FIGURE 2: OTHER REGIONS (3x2 blocks, each block has 2 rows) ----------------
# Layout idea:
# total rows = 3 region-rows * 2 (R2/RMSE) = 6
# total cols = 2
fig2 = plt.figure(figsize=(13, 12))
gs = fig2.add_gridspec(nrows=6, ncols=2)

for i, region in enumerate(regions_non_global):
    col = i % 2
    block = i // 2          # 0,1,2
    r_top = block * 2
    r_bot = r_top + 1

    ax_top = fig2.add_subplot(gs[r_top, col])
    ax_bot = fig2.add_subplot(gs[r_bot, col], sharex=ax_top)

    c = region_colors[region]

    # Top: R2
    plot_metric_dumbbell(ax_top, region, 'TEST_R2', marker_wo='o', marker_w='*', color=c, z_base=30)
    plot_geo_baseline(ax_top, region, color=c, z=15)
    ax_top.set_title(region.replace("_"," "), fontsize=13)
    ax_top.set_ylabel(r"$R^2$")
    ax_top.grid(True, ls="--", alpha=0.25)

    # Bottom: RMSE
    plot_metric_dumbbell(ax_bot, region, 'RMSE', marker_wo='s', marker_w='D', color=c, z_base=30)
    ax_bot.set_ylabel("RMSE")
    ax_bot.grid(True, ls="--", alpha=0.25)

    # Only label x-axis on bottom row of each column (last block)
    if block == 2:
        ax_bot.set_xlabel("Buffer distance (km)")
    else:
        ax_bot.set_xlabel("")
    plt.setp(ax_top.get_xticklabels(), visible=False)

# One shared legend for the whole figure
fig2.tight_layout(rect=[0, 0.07, 1, 1])
fig2.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True)

out_path2 = os.path.join(FigDir, f"{year}_Regions6_Dumbbell_R2_top_RMSE_bottom_0-50_step5.png")
fig2.savefig(out_path2, dpi=500, bbox_inches="tight")
plt.close(fig2)

# ---------------- FIGURE 3: 4 REGIONS (2x2 blocks, each block has 2 rows) ----------------
# Asia, Europe, North_America, Africa — exclude South_America & Oceania_Australia
regions_4 = ['Asia', 'Europe', 'North_America', 'Africa']

fig3 = plt.figure(figsize=(13, 8.5))
gs3 = fig3.add_gridspec(nrows=4, ncols=2)

for i, region in enumerate(regions_4):
    col = i % 2
    block = i // 2          # 0,1
    r_top = block * 2
    r_bot = r_top + 1

    ax_top = fig3.add_subplot(gs3[r_top, col])
    ax_bot = fig3.add_subplot(gs3[r_bot, col], sharex=ax_top)

    c = region_colors[region]

    # Top: R2
    plot_metric_dumbbell(ax_top, region, 'TEST_R2', marker_wo='o', marker_w='*', color=c, z_base=30)
    plot_geo_baseline(ax_top, region, color=c, z=15)
    ax_top.set_title(region.replace("_"," "), fontsize=13)
    ax_top.set_ylabel(r"$R^2$")
    ax_top.grid(True, ls="--", alpha=0.25)

    # Bottom: RMSE
    plot_metric_dumbbell(ax_bot, region, 'RMSE', marker_wo='s', marker_w='D', color=c, z_base=30)
    ax_bot.set_ylabel("RMSE")
    ax_bot.grid(True, ls="--", alpha=0.25)

    # Only label x-axis on bottom row (last block)
    if block == 1:
        ax_bot.set_xlabel("Buffer distance (km)")
    else:
        ax_bot.set_xlabel("")
    plt.setp(ax_top.get_xticklabels(), visible=False)

# One shared legend for the whole figure
fig3.tight_layout(rect=[0, 0.07, 1, 1])
fig3.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True)

out_path3 = os.path.join(FigDir, f"{year}_Regions4_Dumbbell_R2_top_RMSE_bottom_0-50_step5.png")
fig3.savefig(out_path3, dpi=500, bbox_inches="tight")
plt.close(fig3)