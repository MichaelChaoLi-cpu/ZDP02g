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
importlib.reload(SettingForFeatures)


# %%

# %% [markdown]
# ## Functions

# %%
def generate_clean_cls_params(n_samples=500):
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
            "objective": "binary:logistic",  
            "eval_metric": "logloss",          
            "n_estimators": random.choice(n_estimators_grid),
            "learning_rate": random.choice(learning_rate_grid),
            "max_depth": random.choice(max_depth_grid),
            "subsample": random.choice(subsample_grid),
            "colsample_bytree": random.choice(colsample_grid),
            "tree_method": "hist",
            "device": "cuda"
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
os.chdir(os.getenv("PROJECT_ROOT"))

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
# ### fine-tuning

# %%
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )

    y = y.astype(int)
    random_param_list = generate_clean_cls_params(500)
    
    for p in random_param_list:
        _ = Modelling.xgb_cls_kfold_cv(
            X, y,
            n_splits = n_splits,
            params = p,
            log_dir = "logs",
            log_file = f"{aim_variable}_xgb_cls_finetuning_clean.csv"
        )

# %%

# %% [markdown]
# ### check best model

# %%
for aim_variable in aim_variables:
    log = pd.read_csv(os.path.join("logs", f"{aim_variable}_xgb_cls_finetuning_clean.csv"))
    log[['accuracy', 'xgb_params']].groupby('xgb_params').mean().sort_values('accuracy', ascending=False)
    param_str = log[['accuracy', 'xgb_params']].groupby('xgb_params').mean().sort_values('accuracy', ascending=False).reset_index().iloc[0,0]
    params = json.loads(param_str)
    params['random_state'] = 42
    
    with open(f"./{aim_variable}_params.yaml", "w") as f:
        yaml.dump(params, f, sort_keys=False)

# %%
