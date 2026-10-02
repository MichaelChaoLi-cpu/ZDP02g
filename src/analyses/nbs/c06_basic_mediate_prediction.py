# Python source converted from src/analyses/notebooks/c06_basic_mediate_prediction.ipynb
# Code-cell order and Markdown explanations are preserved.

# Resolve project paths before importing the analysis helper modules.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_PROJECT_ROOT / "src/analyses/notebooks"))
_os.chdir(_PROJECT_ROOT)


# %% [markdown]
# # Emotion Basic Value Prediction

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
always_inputs = SettingForFeatures.return_always_input_variable_list()

# %%
aim_variables = SettingForFeatures.return_aim_mediate()

# %%
n_splits = 10

# %%


# %% [markdown]
# ### Test

# %%
start_status = 42
prediction_times = 10

# %%
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )

    y_df = pd.DataFrame(np.full(
        shape = (X.shape[0], prediction_times), 
        fill_value = np.nan))
    y_df.index = X.index

    with open(f"./src/analyses/config/{aim_variable}_params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    print(params)

    for col_index, random_state in enumerate(range(start_status, start_status + prediction_times)):

    
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, test_idx in kf.split(X):
            # Split
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
        
            # Predict
            y_pred = model.predict_proba(X_test)
    
            y_df.iloc[test_idx, col_index] = y_pred[:,1]

    y_df[aim_variable] = y_df.mean(axis = 1)
    y_df.to_parquet(os.path.join('data/exp/model_outputs', f'prediction_{aim_variable}.parquet'))

# %%

