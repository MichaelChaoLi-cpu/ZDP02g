# Python source converted from src/analyses/notebooks/c07_fine_tuning_hyperparameter_ph.ipynb
# Code-cell order and Markdown explanations are preserved.

# Resolve project paths before importing the analysis helper modules.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_PROJECT_ROOT / "src/analyses/notebooks"))
_os.chdir(_PROJECT_ROOT)


# %% [markdown]
# # Fine-tuning Hyoerparameter

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import json
import numpy as np
import pandas as pd
import random
import yaml

# %%
import Modelling
import SettingForFeatures

# %%
import importlib
importlib.reload(Modelling)

# %%


# %% [markdown]
# ## Functions

# %%
def generate_clean_reg_params(n_samples=500):
    params_list = []

    # Pre-defined clean grids
    n_estimators_grid = list(range(200, 1501, 100))     # 200,300,...1500
    learning_rate_grid = [0.001, 0.003, 0.005, 0.01, 
                          0.02, 0.03, 0.05, 0.07, 0.1, 
                          0.15, 0.2, 0.3]               # clean LR values
    max_depth_grid = list(range(3, 13))                # 3–12
    subsample_grid = [round(x, 1) for x in np.linspace(0.5, 1.0, 6)]
    colsample_grid = [round(x, 1) for x in np.linspace(0.5, 1.0, 6)]

    for _ in range(n_samples):
        params = {
            "n_estimators": random.choice(n_estimators_grid),
            "learning_rate": random.choice(learning_rate_grid),
            "max_depth": random.choice(max_depth_grid),
            "subsample": random.choice(subsample_grid),
            "colsample_bytree": random.choice(colsample_grid),
            "tree_method": "hist",
            "device": "cuda:1"
        }
        params_list.append(params)

    return params_list

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
mediate_inputs = SettingForFeatures.return_aim_mediate()

# %%
for mediate_var in mediate_inputs:
    pred_emotion = pd.read_parquet(os.path.join('data/exp/model_outputs', f'prediction_{mediate_var}.parquet'))
    all_data[mediate_var] = pred_emotion[mediate_var]

# %%
always_inputs = SettingForFeatures.return_always_input_variable_list() + mediate_inputs

# %%
always_inputs

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
# ### fine-tuning

# %%
random_param_list = generate_clean_reg_params(500)

# %%
for p in random_param_list:
    _ = Modelling.xgb_reg_kfold_cv(
        X, y,
        n_splits = n_splits,
        params = p,
        log_dir = "data/exp/logs",
        log_file = "xgb_reg_ph_finetuning_clean.csv"
    )

# %%


# %% [markdown]
# ### check best model

# %%
log = pd.read_csv(os.path.join("data/exp/logs", "xgb_reg_ph_finetuning_clean.csv"))

# %%
log[['r2', 'xgb_params']].groupby('xgb_params').mean().sort_values('r2', ascending=False)

# %%
param_str = log[['r2', 'xgb_params']].groupby('xgb_params').mean().sort_values('r2', ascending=False).reset_index().iloc[0,0]

# %%
params = json.loads(param_str)

# %%
params['random_state'] = 42

# %%
with open("./src/analyses/config/params.yaml", "w") as f:
    yaml.dump(params, f, sort_keys=False)

# %%
params

# %%

