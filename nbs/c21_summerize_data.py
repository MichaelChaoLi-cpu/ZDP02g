# %% [markdown]
# # Summerize Data

# %%

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import json
import numpy as np
import pandas as pd

# %%
import SettingForFeatures

# %%
import importlib
importlib.reload(SettingForFeatures)


# %% [markdown]
# ## Functions

# %%
def calculate_bootstrap_se(data_series, n_bootstraps=1000, statistic_func=np.mean):
    """
    Calculates the Bootstrap Standard Error (SE) for a statistic (default is mean).
    
    Parameters:
    data_series (pd.Series): The data series to calculate the statistic from.
    n_bootstraps (int): The number of resamples (bootstraps).
    statistic_func (function): The statistic function to apply (e.g., np.mean, np.median).
    
    Returns:
    float: The Bootstrap Standard Error of the statistic.
    """
    n_samples = len(data_series)
    bootstrap_statistics = []
    
    for _ in range(n_bootstraps):
        # Resample with replacement, size equal to original sample
        resampled_data = np.random.choice(data_series, size=n_samples, replace=True)
        # Calculate the statistic (e.g., mean) on the resampled data
        stat = statistic_func(resampled_data)
        bootstrap_statistics.append(stat)
        
    # The Bootstrap SE is the standard deviation of the bootstrap distribution
    return np.std(bootstrap_statistics)


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
os.makedirs(TABLES := './tables', exist_ok = True)

# %%
COUNTRY_NAMES = {1: 'Argentina', 2:'Australia', 3:'Brazil', 
                 4:'Egypt', 5:'Germany', 6:'India', 
                 7:'Indonesia', 8:'Israel', 9:'Japan', 
                 10:'Kenya', 11:'Mexico', 12:'Nigeria', 
                 13:'the Philippines', 14:'Poland', 16:'South Africa', 
                 17:'Spain', 18:'Tanzania', 19:'Turkey', 
                 20:'the United Kingdom', 22:'the United States', 23:'Sweden', 
                 24:'Hong Kong',
                 15:'Others', 21: 'Others', 25: 'Others'}

# %%
df_raw = pd.read_csv(
    'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv',
    na_values=['', ' ', 'NA', 'N/A'],
    low_memory=False
)

# %%
all_data = SettingForFeatures.data_load_combine_dataset(
    var_name = 'tas'
)

# %%
wave_1_raw = df_raw['COUNTRY'].value_counts(dropna=False).sort_index()

# %%
wave_2_raw = df_raw[df_raw['WAVE_Y2'] == 2.0]['COUNTRY'].value_counts(dropna=False).sort_index()

# %%
merged_data = wave_1_raw.to_frame().reset_index().merge(wave_2_raw.to_frame().reset_index(), on = 'COUNTRY', how = 'left')

# %%
wave_1_valid = all_data[all_data['WAVE'] == 1]['COUNTRY'].value_counts(dropna=False).sort_index()

# %%
wave_2_valid = all_data[all_data['WAVE'] == 2]['COUNTRY'].value_counts(dropna=False).sort_index()

# %%
wave_2_valid.name = 'x'

# %%
merged_data = merged_data.merge(wave_1_valid.to_frame().reset_index(), on = 'COUNTRY', how = 'left')

# %%
merged_data = merged_data.merge(wave_2_valid.to_frame().reset_index(), on = 'COUNTRY', how = 'left')

# %%
merged_data.columns = ['Country', 'Respondents in Wave 1', 'Respondents in Wave 2', 'Valid Data in Wave 1', 'Valid Data in Wave 2']

# %%
merged_data['Country'] = merged_data['Country'].map(COUNTRY_NAMES)

# %%
merged_data

# %%
merged_data.to_excel(os.path.join(TABLES, 'TableS1_respondentCount.xlsx'))

# %%

# %% [markdown]
# ### Data Summary

# %%
always_inputs = SettingForFeatures.return_always_input_variable_list()

# %%
mediate_variables = SettingForFeatures.return_aim_mediate()

# %%
aim_variable = SettingForFeatures.return_aim_variable_ph()

# %%
data_summary = all_data[[aim_variable] + mediate_variables + always_inputs].describe().T.reset_index()

# %%
VARIABLE_MAP_RENAMED = SettingForFeatures.return_readable_variable_name()

# %%
data_summary['index'] = data_summary['index'].map(VARIABLE_MAP_RENAMED)

# %%
data_summary.to_excel(os.path.join(TABLES, 'Table1_DataSummary.xlsx'))

# %%
data_summary

# %%

# %% [markdown]
# ### Hyperparameter

# %%
with open(save_path := os.path.join('results', 'accuracy_comparison.json'), 'r', encoding='utf-8') as f:
    accuracy_comparison = json.load(f)

# %%
df = pd.DataFrame({
    'Physical Health': accuracy_comparison
})
df.index = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'tree_method', 'device', 'random_state', 
            'XGBoost Mean Performance', 'XGBoost SD Performance', 'XGBoost Min Performance', 'XGBoost Max Performance', 
            'Benchmark Mean Performance', 'Benchmark SD Performance', 'Benchmark Min Performance', 'Benchmark Max Performance'] 
df = df.reset_index().rename(columns={'index': 'model'})

# %%
for mediate in mediate_variables:
    with open(save_path := os.path.join('results', f'{mediate}_accuracy_comparison.json'), 'r', encoding='utf-8') as f:
        accuracy_comparison = json.load(f)

    df_emotion = pd.DataFrame({
        VARIABLE_MAP_RENAMED.get(mediate): accuracy_comparison
    }).iloc[2:,:]

    df_emotion.index = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'tree_method', 'device', 'random_state', 
                        'XGBoost Mean Performance', 'XGBoost SD Performance', 'XGBoost Min Performance', 'XGBoost Max Performance', 
                        'Benchmark Mean Performance', 'Benchmark SD Performance', 'Benchmark Min Performance', 'Benchmark Max Performance'] 
    df_emotion = df_emotion.reset_index().rename(columns={'index': 'model'})
    
    df = df.merge(df_emotion, on = 'model')

# %%
df

# %%
df.to_excel(os.path.join(TABLES, 'TableS2_HyperTable.xlsx'))

# %%

# %% [markdown]
# ### Effect Country-level Summary

# %%
all_data = SettingForFeatures.data_load_combine_dataset(
    var_name = 'tas'
)

# %%
mean_lai_direct_effect = pd.read_parquet(os.path.join('results', f'direct_effect_of_TAS_MEAN.parquet')).drop(columns = ['LATITUDE', 'LONGITUDE'])

# %%
print(mean_lai_direct_effect['effect'].mean())
print(calculate_bootstrap_se(mean_lai_direct_effect['effect'], n_bootstraps=1000, statistic_func=np.mean))

# %%
mean_lai_direct_effect.columns = ['Direct Effect of Mean of Temperature']

# %%
mean_lai_direct_effect['COUNTRY'] = all_data['COUNTRY'].copy()

# %%
std_lai_direct_effect = pd.read_parquet(os.path.join('results', f'direct_effect_of_TAS_STD.parquet')).drop(columns = ['LATITUDE', 'LONGITUDE'])

# %%
print(std_lai_direct_effect['effect'].mean())
print(calculate_bootstrap_se(std_lai_direct_effect['effect'], n_bootstraps=1000, statistic_func=np.mean))

# %%
std_lai_direct_effect.columns = ['Direct Effect of STD of Temperature']

# %%
mean_lai_total_effect = pd.read_parquet(os.path.join('results', f'total_effect_of_TAS_MEAN.parquet')).drop(columns = ['LATITUDE', 'LONGITUDE'])

# %%
print(mean_lai_total_effect['effect'].mean())
print(calculate_bootstrap_se(mean_lai_total_effect['effect'], n_bootstraps=1000, statistic_func=np.mean))

# %%
mean_lai_total_effect.columns = ['Total Effect of Mean of Temperature']

# %%
std_lai_total_effect = pd.read_parquet(os.path.join('results', f'total_effect_of_TAS_STD.parquet')).drop(columns = ['LATITUDE', 'LONGITUDE'])

# %%
print(std_lai_total_effect['effect'].mean())
print(calculate_bootstrap_se(std_lai_total_effect['effect'], n_bootstraps=1000, statistic_func=np.mean))

# %%
std_lai_total_effect.columns = ['Total Effect of STD of Temperature']

# %%
merged_df = mean_lai_direct_effect.merge(
    std_lai_direct_effect, left_index = True, right_index = True
).merge(
    mean_lai_total_effect, left_index = True, right_index = True
).merge(
    std_lai_total_effect, left_index = True, right_index = True
)

# %%
df_mean = merged_df.groupby(['COUNTRY']).mean().reset_index()

# %%
bootstrap_se_results = {}

print(f"Starting Bootstrap calculation ({1000} resamples per country/feature)...")

# 1. Iterate through groups defined by COUNTRY
grouped = merged_df.groupby('COUNTRY')

for country, group_df in grouped:
    country_se = {}
    
    # 2. Calculate Bootstrap SE for each numerical feature
    # We exclude the grouping column itself
    numerical_features = group_df.select_dtypes(include=np.number).columns.drop('COUNTRY', errors='ignore')
    
    for feature in numerical_features:
        se = calculate_bootstrap_se(group_df[feature], n_bootstraps=1000)
        country_se[feature] = se
    
    bootstrap_se_results[country] = country_se

print("Calculation complete.")

# %%
df_mean['COUNTRY'] = df_mean['COUNTRY'].map(COUNTRY_NAMES)

# %%
formatted_array = np.vectorize(lambda v: f"{v:.4f}")(df_mean.iloc[:, 1:].values)
df_mean.iloc[:,1:] = formatted_array

# %%
df_mean['Statistic'] = 'Mean'

# %%
df_bootstrap_se = pd.DataFrame(bootstrap_se_results).T.reset_index()
df_bootstrap_se.rename(columns={'index': 'COUNTRY'}, inplace=True)
df_bootstrap_se['COUNTRY'] = df_bootstrap_se['COUNTRY'].map(COUNTRY_NAMES)

# %%
formatted_array = np.vectorize(lambda v: f"({v:.4f})")(df_bootstrap_se.iloc[:, 1:].values)
df_bootstrap_se.iloc[:,1:] = formatted_array

# %%
df_bootstrap_se['Statistic'] = 'SE'

# %%
stacked_data = []
countries = df_mean['COUNTRY']

for country in countries:
    stacked_data.append(df_mean[df_mean['COUNTRY'] == country])
    stacked_data.append(df_bootstrap_se[df_bootstrap_se['COUNTRY'] == country])

# %%
df_stacked = pd.concat(stacked_data)

# %%
df_stacked = df_stacked.set_index(['COUNTRY', 'Statistic'])

# %%
df_stacked.to_excel(os.path.join('tables', 'country_level_effect_summary.xlsx'))

# %%
