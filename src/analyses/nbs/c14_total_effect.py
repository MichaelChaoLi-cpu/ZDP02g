# Python source converted from src/analyses/notebooks/c14_total_effect.ipynb
# Code-cell order and Markdown explanations are preserved.

# Resolve project paths before importing the analysis helper modules.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_PROJECT_ROOT / "src/analyses/notebooks"))
_os.chdir(_PROJECT_ROOT)


# %% [markdown]
# # Physical Health Total Effect

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
os.chdir(_PROJECT_ROOT)

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
# ### Total Effect LAI_MEAN

# %%
X.columns

# %%
effect_variable = 'TAS_MEAN'

# %%
directly_increase_one = pd.read_parquet(os.path.join('data/exp/model_outputs', f'prediction_{effect_variable}_TotalEffect_increase1degree_{aim_variable}.parquet'))

# %%
no_change = pd.read_parquet(os.path.join('data/exp/model_outputs', f'prediction_{aim_variable}.parquet'))

# %%
direct_effect = directly_increase_one[aim_variable].to_numpy() - no_change[aim_variable].to_numpy()

# %%
TestingTools.plot_histogram(direct_effect)

# %%
loc = X[['LATITUDE', 'LONGITUDE']]

# %%
loc['effect'] = direct_effect

# %%
loc.to_parquet(os.path.join('data/exp/model_outputs', f'total_effect_of_{effect_variable}.parquet'))

# %%


# %%


# %% [markdown]
# ### Total Effect LAI_STD

# %%
X.columns

# %%
effect_variable = 'TAS_STD'

# %%
directly_increase_one = pd.read_parquet(os.path.join('data/exp/model_outputs', f'prediction_{effect_variable}_TotalEffect_increase0x1std_{aim_variable}.parquet'))

# %%
no_change = pd.read_parquet(os.path.join('data/exp/model_outputs', f'prediction_{aim_variable}.parquet'))

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
loc.to_parquet(os.path.join('data/exp/model_outputs', f'total_effect_of_{effect_variable}.parquet'))

# %%


# %%


# %%


# %%

