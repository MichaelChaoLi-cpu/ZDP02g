# %% [markdown]
# # Physical Health Basic Value Prediction

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
import ExplainResult
import Modelling
import SettingForFeatures
import TestingTools

# %%
import importlib
importlib.reload(ExplainResult)


# %%

# %% [markdown]
# ## Functions

# %%
def json_serializable(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    raise TypeError(f"Type not serializable: {type(obj)}")


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
all_data = SettingForFeatures.data_load_combine_dataset(
    var_name = 'tas'
)

# %%
always_inputs = SettingForFeatures.return_always_input_variable_list() + SettingForFeatures.return_aim_mediate()

# %%
aim_variable = SettingForFeatures.return_aim_variable_ph()

# %%
n_splits = 10

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

# %%

# %% [markdown]
# ### Direct Effect MEAN

# %%
X.columns

# %%
effect_variable = 'TAS_MEAN'

# %%
directly_increase_one = pd.read_parquet(os.path.join('results', f'prediction_{effect_variable}_increase1degree_{aim_variable}.parquet'))

# %%
directly_increase_one['PHYSICAL_HLTH']

# %%
no_change = pd.read_parquet(os.path.join('results', f'prediction_{aim_variable}.parquet'))

# %%
no_change['PHYSICAL_HLTH']

# %%
direct_effect = directly_increase_one[aim_variable] - no_change[aim_variable]

# %%
TestingTools.plot_histogram(direct_effect.to_numpy())

# %%
direct_effect.describe()

# %%
loc = X[['LATITUDE', 'LONGITUDE']]

# %%
loc['effect'] = direct_effect

# %%
loc.to_parquet(os.path.join('results', f'direct_effect_of_{effect_variable}.parquet'))

# %%

# %%

# %% [markdown]
# ### Direct Effect LAI_STD

# %%
X.columns

# %%
effect_variable = 'TAS_STD'

# %%
directly_increase_one = pd.read_parquet(os.path.join('results', f'prediction_{effect_variable}_increase0x1std_{aim_variable}.parquet'))

# %%
no_change = pd.read_parquet(os.path.join('results', f'prediction_{aim_variable}.parquet'))

# %%
direct_effect = directly_increase_one[aim_variable] - no_change[aim_variable]

# %%
TestingTools.plot_histogram(direct_effect.to_numpy())

# %%
direct_effect.describe()

# %%
loc = X[['LATITUDE', 'LONGITUDE']]

# %%
loc['effect'] = direct_effect

# %%
loc.to_parquet(os.path.join('results', f'direct_effect_of_{effect_variable}.parquet'))

# %%

# %%

# %%

# %%
