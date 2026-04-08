import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
df = pd.read_csv('/storage1/fs1/rvmartin2/Active/yany1/1.project/Evaluation/global/NO2_monthly_2019_processed.csv')

# Filter for 2019 only
# Assuming there's a 'year' column, or extract from date column
if 'year' in df.columns:
    df = df[df['year'] == 2019]
elif 'date' in df.columns:
    df['year'] = pd.to_datetime(df['date']).dt.year
    df = df[df['year'] == 2019]

print(f"Data filtered for 2019. Total rows: {len(df)}")

# Define seasons mapping
def get_season(month):
    """Map month to season"""
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    elif month in [12, 1, 2]:
        return 'Winter'
    return None

# Assuming there's a month column - adjust the column name if different
# If month is not present, you'll need to extract it from your date column
if 'month' in df.columns:
    df['season'] = df['month'].apply(get_season)
elif 'date' in df.columns:
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['season'] = df['month'].apply(get_season)
else:
    print("Warning: No month/date column found. Please check your data structure.")

# Define the NO2 columns to analyze
no2_columns = ['obs_no2', 'gchp_no2', 'geophysical_no2']
variable_labels = ['Observed NO₂', 'GCHP NO₂', 'Geophysical NO₂']

# Define regions
regions = ['northamerica', 'europe', 'asia']
region_colors = {
    'asia': 'red',
    'europe': 'green',
    'northamerica': 'orange',
}

# Season order
season_order = ['Winter', 'Spring', 'Summer', 'Fall']

# Set up the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figure with 3 rows (variables) x 4 columns (seasons)
# A4 size in inches: 8.27 x 11.69 (portrait) or 11.69 x 8.27 (landscape)
fig, axes = plt.subplots(3, 4, figsize=(16.54, 11.69))  # Double A4 width, A4 landscape height for 3 rows

# Iterate through each variable (rows)
for row_idx, (col_name, var_label) in enumerate(zip(no2_columns, variable_labels)):
    # Iterate through each season (columns)
    for col_idx, season in enumerate(season_order):
        ax = axes[row_idx, col_idx]
        
        # Filter data for this season
        season_df = df[df['season'] == season]
        
        # Filter non-null values for this variable
        var_season_data = season_df[season_df[col_name].notna()]
        
        if len(var_season_data) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {season}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{var_label}\n{season}', fontsize=12, fontweight='bold')
            continue
        
        column_data = var_season_data[col_name]
        
        # Plot overall distribution for this season
        ax.hist(column_data, bins=40, alpha=0.5, color='black', 
               label='Global', density=True)
        
        # Plot distribution for each region
        if 'region' in var_season_data.columns:
            for region in regions:
                region_data = var_season_data[var_season_data['region'] == region][col_name]
                if len(region_data) > 0:
                    ax.hist(region_data, bins=30, alpha=0.6, 
                           color=region_colors[region], label=region.capitalize(),
                           density=True, histtype='step', linewidth=2)
        
        # Title (season name on top row only)
        if row_idx == 0:
            title = f'{season}'
        else:
            title = ''
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        
        # Labels
        if row_idx == 2:  # Bottom row
            ax.set_xlabel('NO₂ Concentration (ppb)', fontsize=14)
        if col_idx == 0:  # Left column
            ax.set_ylabel('Density', fontsize=14)
            # Add variable name on the left side (vertical text)
            ax.text(-0.3, 0.5, var_label, transform=ax.transAxes,
                   fontsize=16, fontweight='bold', rotation=90,
                   verticalalignment='center', horizontalalignment='center')
        
        ax.set_xlim(0, 60)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add statistics
        mean_val = column_data.mean()
        median_val = column_data.median()
        # Count unique lat/lon pairs
        if 'lat' in var_season_data.columns and 'lon' in var_season_data.columns:
            n_samples = len(var_season_data[['lat', 'lon']].drop_duplicates())
        else:
            n_samples = len(column_data)
        
        stats_text = f'n={n_samples}\nmean={mean_val:.2f}\nmedian={median_val:.2f}'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), 
               fontsize=11)
        
        # Add legend only to top-right plot
        # if row_idx == 0 and col_idx == 3:
        #     ax.legend(loc='upper left', fontsize=12, frameon=True)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()


# Function to print seasonal statistics
def print_seasonal_stats():
    """Print summary statistics for each variable, season, and region"""
    
    print("\nSeasonal Summary Statistics")
    print("=" * 100)
    
    for col_name, var_label in zip(no2_columns, variable_labels):
        print(f"\n{var_label}:")
        print("-" * 100)
        
        for season in season_order:
            season_df = df[(df['season'] == season) & (df[col_name].notna())]
            
            if len(season_df) == 0:
                print(f"  {season}: No data available")
                continue
            
            column_data = season_df[col_name]
            
            # Overall seasonal stats
            print(f"  {season}:")
            print(f"    Overall: n={len(season_df):>5}, mean={column_data.mean():>7.3f}, "
                  f"median={column_data.median():>7.3f}, std={column_data.std():>7.3f}")
            
            # Regional stats
            if 'region' in season_df.columns:
                for region in regions:
                    region_data = season_df[season_df['region'] == region][col_name]
                    if len(region_data) > 0:
                        print(f"      {region.capitalize():>12}: n={len(region_data):>5}, "
                              f"mean={region_data.mean():>7.3f}, median={region_data.median():>7.3f}, "
                              f"std={region_data.std():>7.3f}")

# Call the function to print stats
print_seasonal_stats()


# Alternative: Box plots by season
def create_seasonal_box_plots():
    """Create seasonal box plots for each variable"""
    
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    
    for row_idx, season in enumerate(season_order):
        season_df = df[df['season'] == season]
        
        for col_idx, (col_name, var_label) in enumerate(zip(no2_columns, variable_labels)):
            ax = axes[row_idx, col_idx]
            
            var_season_data = season_df[season_df[col_name].notna()]
            
            if len(var_season_data) == 0 or 'region' not in var_season_data.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                continue
            
            # Prepare box plot data
            box_data = []
            box_labels = []
            box_colors = []
            
            for region in regions:
                region_data = var_season_data[var_season_data['region'] == region][col_name].dropna()
                if len(region_data) > 0:
                    box_data.append(region_data)
                    box_labels.append(region.capitalize())
                    box_colors.append(region_colors[region])
            
            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            # Titles and labels
            if row_idx == 0:
                title = f'{var_label}\n{season}'
            else:
                title = f'{season}'
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            
            if row_idx == 3:
                ax.set_xlabel('Region', fontsize=11)
            if col_idx == 0:
                ax.set_ylabel('NO₂ (ppb)', fontsize=11)
            
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Seasonal NO₂ Box Plots by Region', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

# Uncomment to create box plots
# create_seasonal_box_plots()