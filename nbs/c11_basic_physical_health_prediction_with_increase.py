# %% [markdown]
# # Physical Health Basic Value Prediction with increase

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

# %% [markdown]
# ### Test

# %%
start_status = 42
prediction_times = 10

# %%
effect_variable = 'TAS_MEAN'

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

y_df = pd.DataFrame(np.full(
    shape = (X.shape[0], prediction_times), 
    fill_value = np.nan))
y_df.index = X.index

with open(f"./params.yaml", "r") as f:
    params = yaml.safe_load(f)
params['device'] = 'cuda:0'
print(params)

for col_index, random_state in enumerate(range(start_status, start_status + prediction_times)):


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
    
        # Predict
        X_tide = X_test
        X_tide[effect_variable] = X_test[effect_variable] + 1
        y_pred = model.predict(X_tide)

        y_df.loc[X_test.index, col_index] = y_pred

y_df[aim_variable] = y_df.mean(axis = 1)
y_df.to_parquet(os.path.join('results', f'prediction_{effect_variable}_increase1degree_{aim_variable}.parquet'))

# %%
effect_variable = 'TAS_STD'

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

y_df = pd.DataFrame(np.full(
    shape = (X.shape[0], prediction_times), 
    fill_value = np.nan))
y_df.index = X.index

with open(f"./params.yaml", "r") as f:
    params = yaml.safe_load(f)
params['device'] = 'cuda:0'
print(params)

for col_index, random_state in enumerate(range(start_status, start_status + prediction_times)):


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
    
        # Predict
        X_tide = X_test
        X_tide[effect_variable] = X_test[effect_variable] + 0.1
        y_pred = model.predict(X_tide)

        y_df.loc[X_test.index, col_index] = y_pred

y_df[aim_variable] = y_df.mean(axis = 1)
y_df.to_parquet(os.path.join('results', f'prediction_{effect_variable}_increase0x1std_{aim_variable}.parquet'))

# %%

# %%
