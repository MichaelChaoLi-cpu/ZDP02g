# %% [markdown]
# # Visualization for Materials

# %%

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap, Normalize

# %%
import Modelling
import SettingForFeatures
import TestingTools

# %%
import importlib
importlib.reload(SettingForFeatures)


# %% [markdown]
# ## Functions

# %%
def convert_count_data_into_map(
    count_df : pd.DataFrame
) -> np.ndarray:
    grid_array = np.full((180, 360), np.nan)
    count_df['lat_idx'] = (count_df['LATITUDE'] + 90).astype(int)
    count_df['lon_idx'] = (count_df['LONGITUDE'] + 180).astype(int)

    gridded_df = count_df.groupby(['lat_idx', 'lon_idx'])['Count'].sum().reset_index()

    lat_indices = gridded_df['lat_idx'].to_numpy()
    lon_indices = gridded_df['lon_idx'].to_numpy()
    values = gridded_df['Count'].to_numpy()

    grid_array[lat_indices, lon_indices] = values

    return grid_array[::-1,:]


# %%
def convert_data_into_map_by_mean(
    df : pd.DataFrame,
    var : str
) -> np.ndarray:
    grid_array = np.full((180, 360), np.nan)
    df['lat_idx'] = (df['LATITUDE'] + 90).astype(int)
    df['lon_idx'] = (df['LONGITUDE'] + 180).astype(int)

    gridded_df = df.groupby(['lat_idx', 'lon_idx'])[var].mean().reset_index()

    lat_indices = gridded_df['lat_idx'].to_numpy()
    lon_indices = gridded_df['lon_idx'].to_numpy()
    values = gridded_df[var].to_numpy()

    grid_array[lat_indices, lon_indices] = values

    return grid_array[::-1,:]


# %%

# %% [markdown]
# ## Runs

# %%
if __name__ == '__main__':
    pass

# %%
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.chdir(os.getenv("PROJECT_ROOT"))

# %%
os.makedirs(FIGURES := "./figures", exist_ok = True)

# %%
all_data = SettingForFeatures.data_load_combine_dataset(
    var_name = 'tas'
)

# %%
always_inputs = SettingForFeatures.return_always_input_variable_list() + SettingForFeatures.return_aim_mediate()

# %%
aim_variable = SettingForFeatures.return_aim_variable_ph()

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

# %%
# Define custom colormap
colors = ['blue', 'green', 'yellow', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# %%
# Define custom colormap
colors = ['blue', 'green', 'white', 'yellow', 'red']
custom_cmap_white = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# %%
VARIABLE_MAP_RENAMED = SettingForFeatures.return_readable_variable_name()

# %%

# %% [markdown]
# ### Plot Map of Observation

# %%
count_X = X[['LATITUDE', 'LONGITUDE', "WAVE"]]
count_X['Count'] = 1

# %%
count_df = count_X.groupby(['LATITUDE', 'LONGITUDE', "WAVE"]).sum().reset_index()


# %%
# --- Function to prepare data ---
def get_plot_data(wave_num, count_df):
    # Filter data for the specific wave and remove the WAVE column
    temp_df = count_df[count_df['WAVE'] == wave_num].drop(columns=['WAVE'])
    # Convert dataframe into a 2D grid array
    grid_array = convert_count_data_into_map(temp_df)
    # Reverse rows to align with latitude (South to North)
    data_array = grid_array[::-1].copy()
    # Calculate the sum of respondents across longitudes for each latitude
    latitude_sum = np.nansum(data_array, axis=1)
    return data_array, latitude_sum


# %%
idx = 'abcdefg'

# 1. Initialize Figure
# Adjust figsize for 3x2 layout: 18 width, 18 height (approx 6 per row)
fig, axes = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(18, 18), 
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

lats = np.linspace(-90, 89, 180)
lons = np.linspace(-180, 179, 360)
extent = [lons.min(), lons.max() + 1, lats.min(), lats.max() + 1]

for i, wave in enumerate([1, 2]):
    data_array, latitude_sum = get_plot_data(wave, count_df)
    
    ax_map = axes[i]
    # Use log10 scale for visualization, set aspect to 'auto' to align with the histogram
    im = ax_map.imshow(
        np.log10(data_array + 1), # Added +1 to avoid log(0) issues
        origin='lower',
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap,
        aspect='auto', 
        vmin=0,
        vmax=np.log10(5000)
    )
    
    # Add geographic features
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax_map.set_title(f'({idx[i]}) Wave {wave} Valid Respondent Count', loc='left', fontsize=14, fontweight='bold')
    
    # Configure gridlines
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

# --- Add Global Colorbar with Custom Log-Ticks ---
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 
cbar = fig.colorbar(
    im, 
    cax=cbar_ax, 
    orientation='horizontal'
)

# Setting custom ticks and labels:
# Tick positions: 0, 1, 2, 3 (log scale)
# Labels: 1, 10, 100, 1000+ (original values)
# Note: Since we used log10(data+1), 0 represents log10(1)
cbar.set_ticks([0, 1, 2, 3, np.log10(5000)])
cbar.set_ticklabels(['1', '10', '100', '1000', '5000 +'])
cbar.set_label('Total Respondents per Grid Cell', fontsize=12)


# Save and show the final result
output_path = os.path.join(FIGURES, 'fig01_repsondent_distribution.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# ### Variable of Interests

# %%
all_variable = X
all_variable[aim_variable] = y

# %%
all_variable.columns

# %%
all_variable_of_interest = all_variable[[aim_variable, 'LATITUDE', 'LONGITUDE', 'WAVE']].groupby(['LATITUDE', 'LONGITUDE', 'WAVE']).mean().reset_index()

# %%
# --- Configuration ---
variables = aim_variable
variable_label = VARIABLE_MAP_RENAMED.get(variables)
waves = [1, 2]
panel_idx = 'abcdef' # Labels for (a), (b), (c)...

# 1. Initialize Figure
# Adjust figsize for 3x2 layout: 18 width, 18 height (approx 6 per row)
fig, axes = plt.subplots(
    nrows=2, ncols=1, 
    figsize=(18, 18), 
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

# Shared extent for all maps
lats = np.linspace(-90, 89, 180)
lons = np.linspace(-180, 179, 360)
extent = [lons.min(), lons.max() + 1, lats.min(), lats.max() + 1]

# 2. Loop through Variables (Rows) and Waves (Columns)
im = None # Placeholder for colorbar
p_count = 0

var = variables
for r, wave in enumerate(waves):
    ax = axes[r]
    
    # Prepare Data
    temp_df = all_variable_of_interest[
        (all_variable_of_interest['WAVE'] == wave)
    ]
    
    # Use your custom function to convert to 2D grid
    grid_array = convert_data_into_map_by_mean(df=temp_df, var=var)
    data_array = grid_array[::-1].copy() # Ensure correct orientation
    
    # Plot Raster
    im = ax.imshow(
        data_array,
        origin='lower',
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap,
        aspect='auto',
        vmin=0, # Fixed scale as requested
        vmax=10
    )
    
    # Add Geographic Features
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.set_global()
    
    # Set Title: e.g., "(a) Life Satisfaction - Wave 1"
    ax.set_title(f'({panel_idx[p_count]}) {variable_label} - Wave {wave}', 
                 loc='left', fontsize=12, fontweight='bold')
    
    # Gridlines (Optional: only show labels on left and bottom to save space)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.8, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    p_count += 1

# 3. Add a single Shared Colorbar at the bottom
# [left, bottom, width, height] in normalized figure coordinates
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Average Score', fontsize=14, fontweight='bold')

# 4. Save and Show
output_path = os.path.join(FIGURES, 'fig02_physical_health_metrics.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# ### Mediate

# %%
all_variable = X

# %%
all_variable.columns

# %%
all_variable_of_interest = all_variable[SettingForFeatures.return_aim_mediate() + ['LATITUDE', 'LONGITUDE', 'WAVE']].groupby(['LATITUDE', 'LONGITUDE', 'WAVE']).mean().reset_index()

# %%
# --- Configuration ---
variables = SettingForFeatures.return_aim_mediate()
waves = [1, 2]
panel_idx = 'abcdefgh' # Labels for (a), (b), (c)...

# 1. Initialize Figure
# Adjust figsize for 3x2 layout: 18 width, 18 height (approx 6 per row)
fig, axes = plt.subplots(
    nrows=3, ncols=2, 
    figsize=(24, 18), 
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

# Shared extent for all maps
lats = np.linspace(-90, 89, 180)
lons = np.linspace(-180, 179, 360)
extent = [lons.min(), lons.max() + 1, lats.min(), lats.max() + 1]

# 2. Loop through Variables (Rows) and Waves (Columns)
im = None # Placeholder for colorbar
p_count = 0

for r, var in enumerate(variables):
    for c, wave in enumerate(waves):
        ax = axes[r, c]
        
        # Prepare Data
        temp_df = all_variable_of_interest[
            (all_variable_of_interest['WAVE'] == wave)
        ]
        
        # Use your custom function to convert to 2D grid
        grid_array = convert_data_into_map_by_mean(df=temp_df, var=var)
        data_array = grid_array[::-1].copy() # Ensure correct orientation
        
        # Plot Raster
        im = ax.imshow(
            data_array,
            origin='lower',
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=custom_cmap,
            aspect='auto',
            vmin=0, # Fixed scale as requested
            vmax=1
        )
        
        # Add Geographic Features
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.set_global()
        
        # Set Title: e.g., "(a) Life Satisfaction - Wave 1"
        ax.set_title(f'({panel_idx[p_count]}) {VARIABLE_MAP_RENAMED.get(var)} - Wave {wave}', 
                     loc='left', fontsize=12, fontweight='bold')
        
        # Gridlines (Optional: only show labels on left and bottom to save space)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.8, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        if c > 0: gl.left_labels = False # Only show latitude on the first column
        if r < 3: gl.bottom_labels = False # Only show longitude on the last row
        
        p_count += 1

# 3. Add a single Shared Colorbar at the bottom
# [left, bottom, width, height] in normalized figure coordinates
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.015]) 
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Probability', fontsize=14, fontweight='bold')

# 4. Save and Show
output_path = os.path.join(FIGURES, 'fig03_mediate_metrics.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# %%

# %% [markdown]
# ### TAS

# %%
all_variable_of_interest = all_variable[['TAS_MEAN', 'TAS_STD', 'LATITUDE', 'LONGITUDE', 'WAVE']].groupby(['LATITUDE', 'LONGITUDE', 'WAVE']).mean().reset_index()

# %%
# --- Configuration ---
variables = ['TAS_MEAN', 'TAS_STD']
variable_labels = ['Annual Mean Temperature', 'Annual Standard Deviation of Temperature']
waves = [1, 2]
panel_idx = 'abcdef'  # Labels for (a), (b), (c)...

# 1. Initialize Figure
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(24, 12),
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.05}
)

# Shared extent for all maps
lats = np.linspace(-90, 89, 180)
lons = np.linspace(-180, 179, 360)
extent = [lons.min(), lons.max() + 1, lats.min(), lats.max() + 1]

# 2. Loop through Variables (Rows) and Waves (Columns)
p_count = 0
im_row0 = None  # for (a)(b)
im_row1 = None  # for (c)(d)

for r, var in enumerate(variables):
    for c, wave in enumerate(waves):
        ax = axes[r, c]

        # Prepare Data
        temp_df = all_variable_of_interest[all_variable_of_interest['WAVE'] == wave]
        grid_array = convert_data_into_map_by_mean(df=temp_df, var=var)
        data_array = grid_array[::-1].copy()

        # row-specific color scale
        if r == 0:
            vmin, vmax = 0, 30 
            cmap = custom_cmap
        else:
            vmin, vmax = 0, 10
            cmap = custom_cmap

        # Plot Raster
        im = ax.imshow(
            data_array,
            origin='lower',
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )

        # store mappable for shared colorbars
        if r == 0:
            im_row0 = im
        else:
            im_row1 = im

        # Add Geographic Features
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.set_global()

        # Title
        ax.set_title(
            f'({panel_idx[p_count]}) {variable_labels[r]} - Wave {wave}',
            loc='left', fontsize=12, fontweight='bold'
        )

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.8, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        if c > 0:
            gl.left_labels = False                 # only left column shows lat labels
        if r < (len(variables) - 1):
            gl.bottom_labels = False               # only last row shows lon labels

        p_count += 1

# 3. Add two shared Colorbars (one per row)
cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.30])
cbar1 = fig.colorbar(im_row0, cax=cbar_ax1, orientation='vertical')
cbar1.set_label('Average Temperature', fontsize=12, fontweight='bold')

cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.30])
cbar2 = fig.colorbar(im_row1, cax=cbar_ax2, orientation='vertical')
cbar2.set_label('Average STD', fontsize=12, fontweight='bold')

# 4. Save and Show
output_path = os.path.join(FIGURES, 'fig04_tas.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()


# %%

# %%

# %% [markdown]
# ### Global Relationship

# %%
def get_result_and_plot(
    output_path : str,
    varname='lai'
):

    idx = 'abcdefghijkl'
    stats = ['MEAN', 'STD']

    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(14, 20),
        sharex='col',
        sharey=False
    )

    indicator = [SettingForFeatures.return_aim_variable_ph()] + SettingForFeatures.return_aim_mediate()

    for i, outcome in enumerate(indicator):
        for j, stat in enumerate(stats):

            ax = axes[i, j]

            # ---- load data ----
            potential_values = np.load(
                os.path.join(
                    'results',
                    f'{indicator[i]}_potenital_values_{varname.upper()}_{stat}.npy'
                )
            )

            pdp_array = np.load(
                os.path.join(
                    'results',
                    f'{indicator[i]}_pdp_array_{varname.upper()}_{stat}.npy'
                )
            )

            # ---- statistics ----
            pdp_mean = np.mean(pdp_array, axis=0)
            pdp_std  = np.std(pdp_array, axis=0)

            # ---- plot mean line ----
            ax.plot(
                potential_values,
                pdp_mean,
                linewidth=2,
                label='Mean Prediction'
            )

            # ---- 95% confidence interval ----
            ax.fill_between(
                potential_values,
                pdp_mean - 1.96 * pdp_std,
                pdp_mean + 1.96 * pdp_std,
                alpha=0.3,
                label=r'$\pm 1.96\sigma$'
            )

            # ---- quadratic fitted line ----
            coef = np.polyfit(potential_values, pdp_mean, deg=2)
            fit_line = np.polyval(coef, potential_values)
            
            ax.plot(
                potential_values,
                fit_line,
                linestyle='--',
                linewidth=2,
                label='U-shaped Fit',
                color = 'red'
            )

            # ---- labels ----
            if i == 4:
                if stat == 'MEAN':
                    ax.set_xlabel(f'Mean of {varname.upper()}')
                else:
                    ax.set_xlabel(f'Standard Deviation of {varname.upper()}')

            if j == 0:
                ax.set_ylabel(f'Predicted {VARIABLE_MAP_RENAMED.get(outcome)}')

            ax.set_title(f'{idx[2*i+j]}:{VARIABLE_MAP_RENAMED.get(outcome)}', loc = 'left')
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


# %%
get_result_and_plot(
    output_path = os.path.join(FIGURES, 'fig05_global_variation.jpg'),
    varname='tas'
)


# %%

# %% [markdown]
# ### Important

# %%
def return_importance_df(
    addr = os.path.join('results', 'importance.csv')
):
    VARIABLE_MAP_RENAMED = SettingForFeatures.return_readable_variable_name()
    
    feature_importance_full = pd.read_csv(addr, index_col = 0)
    feature_importance_full.index = feature_importance_full.index.map(VARIABLE_MAP_RENAMED)
    feature_importance_full = feature_importance_full.iloc[:,:10]
    feature_importance_full_sum = feature_importance_full.sum(axis = 0)
    feature_importance_full = feature_importance_full / feature_importance_full_sum * 100

    feature_importance_full['mean'] = feature_importance_full.mean(axis = 1)
    feature_importance_full['std'] = feature_importance_full.std(axis = 1)
    
    return feature_importance_full.sort_values('mean', ascending = False)


# %%
mediate_variables = SettingForFeatures.return_aim_mediate()

# %%
feature_importance = return_importance_df(os.path.join('results', 'importance.csv'))
feature_importance_c = return_importance_df(os.path.join('results', f'{mediate_variables[0]}_importance.csv'))
feature_importance_d = return_importance_df(os.path.join('results', f'{mediate_variables[1]}_importance.csv'))
feature_importance_f = return_importance_df(os.path.join('results', f'{mediate_variables[2]}_importance.csv'))

# %%
feature_importance.index = feature_importance.index.fillna('Household Income (USD)')
feature_importance_c.index = feature_importance_c.index.fillna('Household Income (USD)')
feature_importance_d.index = feature_importance_d.index.fillna('Household Income (USD)')
feature_importance_f.index = feature_importance_f.index.fillna('Household Income (USD)')

# %%
fig, axes = plt.subplots(
    nrows=4, ncols=1,
    figsize=(14, 24),
    sharex=True
)

datasets = [
    (feature_importance,    'a: Mental Health', 'tab:blue'),
    (feature_importance_c,  f"b: {VARIABLE_MAP_RENAMED.get(mediate_variables[0])}", 'tab:green'),
    (feature_importance_d,  f"c: {VARIABLE_MAP_RENAMED.get(mediate_variables[1])}", 'tab:red'),
    (feature_importance_f,  f"d: {VARIABLE_MAP_RENAMED.get(mediate_variables[2])}", 'tab:purple'),
]

for ax, (df, title, color) in zip(axes, datasets):
    df_plot = df.copy()

    df_plot.index = df_plot.index.astype(str)

    ax.barh(
        df_plot.index,
        df_plot["mean"].astype(float),
        xerr=(df_plot["std"].astype(float) * 1.96),
        color=color,
        alpha=0.9,
        capsize=4
    )

    ax.invert_yaxis()
    ax.set_title(title, loc='left')
    ax.grid(linestyle='--', alpha=0.6)

for ax in axes:
    for label in ax.get_yticklabels():
        if 'LAI' in label.get_text():
            label.set_fontweight('bold')

axes[-1].set_xlabel("Importance (%)")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "fig07_feature_importance_3panel.png"), dpi=300, bbox_inches='tight')
plt.show()

# %%

# %%

# %% [markdown]
# ### Direct Effect

# %%
# =========================
# Configuration
# =========================
effect_variables = ["TAS_MEAN", "TAS_STD"]
panel_titles = [
    "(a) Grid-level Average Direct Impacts of Change in Mean of Temperature",
    "(b) Grid-level Average Direct Impacts of Change in STD of Temperature"
]

lats = np.linspace(-90, 89, 180)
lons = np.linspace(-180, 179, 360)
extent = [lons.min(), lons.max() + 1, lats.min(), lats.max() + 1]

# =========================
# Create figure (2 rows, 1 column)
# =========================
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(18, 18),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"hspace": 0.08}
)

im = None  # placeholder for colorbar

# =========================
# Loop over panels
# =========================
for i, (ax, effect_variable, title) in enumerate(
    zip(axes, effect_variables, panel_titles)
):
    # Load data
    direct_effect = pd.read_parquet(
        os.path.join("results", f"direct_effect_of_{effect_variable}.parquet")
    )
    print(direct_effect.mean())

    grid_array = convert_data_into_map_by_mean(
        df=direct_effect,
        var="effect"
    )
    data_array = grid_array[::-1].copy()

    # Plot raster
    im = ax.imshow(
        data_array,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap_white,
        aspect="auto",
        vmin=-0.1,
        vmax=0.1
    )

    # Borders and coastlines
    ax.add_feature(
        cfeature.BORDERS,
        linestyle="-",
        edgecolor="black",
        linewidth=0.4
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

    # Global extent
    ax.set_global()

    # Title (left aligned)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

# =========================
# Shared colorbar (right side)
# =========================
cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
cbar.set_label(
    "Grid-level Average Direct Effect",
    fontsize=12,
    fontweight="bold"
)

# =========================
# Save and show
# =========================
plt.savefig(
    os.path.join(FIGURES, "fig08_09_direct_effect_2panel.jpg"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# %%

# %% [markdown]
# ### Total Effect

# %%
# =========================
# Configuration
# =========================
effect_variables = ["TAS_MEAN", "TAS_STD"]
panel_titles = [
    "(a) Grid-level Average Total Impacts of Change in Mean of Temperature",
    "(b) Grid-level Average Total Impacts of Change in STD of Temperature"
]

lats = np.linspace(-90, 89, 180)
lons = np.linspace(-180, 179, 360)
extent = [lons.min(), lons.max() + 1, lats.min(), lats.max() + 1]

# =========================
# Create figure (2 rows, 1 column)
# =========================
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(18, 18),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"hspace": 0.08}
)

im = None  # placeholder for colorbar

# =========================
# Loop over panels
# =========================
for i, (ax, effect_variable, title) in enumerate(
    zip(axes, effect_variables, panel_titles)
):
    # Load data
    direct_effect = pd.read_parquet(
        os.path.join("results", f"total_effect_of_{effect_variable}.parquet")
    )

    grid_array = convert_data_into_map_by_mean(
        df=direct_effect,
        var="effect"
    )
    data_array = grid_array[::-1].copy()
    print(np.nanmean(data_array))

    # Plot raster
    im = ax.imshow(
        data_array,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=custom_cmap_white,
        aspect="auto",
        vmin=-2,
        vmax=2
    )

    # Borders and coastlines
    ax.add_feature(
        cfeature.BORDERS,
        linestyle="-",
        edgecolor="black",
        linewidth=0.4
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

    # Global extent
    ax.set_global()

    # Title (left aligned)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

# =========================
# Shared colorbar (right side)
# =========================
cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
cbar.set_label(
    "Grid-level Average Total Effect",
    fontsize=12,
    fontweight="bold"
)

# =========================
# Save and show
# =========================
plt.savefig(
    os.path.join(FIGURES, "fig10_total_effect_2panel.jpg"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# %%

# %%

# %%
