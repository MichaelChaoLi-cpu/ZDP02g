# Python source converted from src/analyses/notebooks/c08_check_explanation_physical_health_tas_relationship.ipynb
# Code-cell order and Markdown explanations are preserved.

# Resolve project paths before importing the analysis helper modules.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_PROJECT_ROOT / "src/analyses/notebooks"))
_os.chdir(_PROJECT_ROOT)


# %% [markdown]
# # Check Explanation of the relationship between Mental HEalth and LAI

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
def compute_single_pdp_income(
    var: str,
    X: pd.DataFrame,
    model_list: list,
    real_boundary=(0, 13),
    stripe: float = 1
) -> tuple:
    # Determine the grid boundaries based on the specified quantiles
    low_b = real_boundary[0]
    up_b = real_boundary[1]
    
    # Generate the discrete grid of feature values
    potenital_values = np.arange(low_b, up_b, stripe, dtype = np.float64)
    X_adjust = X.copy()
    
    pdp_list = []
    
    # Iterate through each model in the ensemble/list
    for model in model_list:
        pdp = np.full_like(potenital_values, fill_value=np.nan)
        
        # Iterate through each grid point (potential value)
        for idx, potenital_value in enumerate(potenital_values):
            # 1. Substitute the feature column with the current fixed value
            X_adjust[var] = np.exp(potenital_value) - 1
            
            # 2. Predict the outcome for the entire adjusted dataset
            y_pred = model.predict(X_adjust)
            
            # 3. Calculate the partial dependence (average prediction)
            pdp[idx] = np.mean(y_pred)
        
        pdp_list.append(pdp)

    pdp_array = np.array(pdp_list)

    return potenital_values, pdp_array

# %%
def compute_two_feature_pdp_self_defined_real_value(
    var_1: str,
    var_2: str,
    X: pd.DataFrame,
    model_list: list,
    real_boundary_1=(0.05, 0.95),
    real_boundary_2=(0.05, 0.95),
    stripe_1: float = 0.2,
    stripe_2: float = 0.2
) -> tuple:

    # 1. Determine Grid Boundaries and Create Grid for var_1
    low_b_1 = real_boundary_1[0]
    up_b_1 = real_boundary_1[1]
    potenital_values_1 = np.arange(low_b_1, up_b_1, stripe_1, dtype = np.float64)

    # 2. Determine Grid Boundaries and Create Grid for var_2
    low_b_2 = real_boundary_2[0]
    up_b_2 = real_boundary_2[1]
    potenital_values_2 = np.arange(low_b_2, up_b_2, stripe_2, dtype = np.float64)

    # Calculate the shape of the 2D PDP result
    N_1 = len(potenital_values_1)
    N_2 = len(potenital_values_2)
    
    X_adjust = X.copy()
    pdp_list = []
    
    # Iterate through each model
    for model in model_list:
        # Initialize the 2D PDP array for the current model
        pdp = np.full((N_1, N_2), fill_value=np.nan)
        
        # 3. Outer Loop: Iterate through the grid points of the first feature (rows)
        for idx_1, potential_value_1 in enumerate(potenital_values_1):
            # Temporarily fix var_1 for this entire row/iteration
            X_adjust[var_1] = potential_value_1
            
            # 4. Inner Loop: Iterate through the grid points of the second feature (columns)
            for idx_2, potential_value_2 in enumerate(potenital_values_2):
                
                # Temporarily fix var_2 for this specific grid cell
                X_adjust[var_2] = np.exp(potential_value_2) - 1
                
                # Predict and store the average prediction for this grid cell (var_1, var_2)
                y_pred = model.predict(X_adjust)
                pdp[idx_1, idx_2] = np.mean(y_pred)
        
        pdp_list.append(pdp)

    # Combine results into a 3D array (N_models, N_grid_1, N_grid_2)
    pdp_array = np.array(pdp_list)

    return potenital_values_1, potenital_values_2, pdp_array

# %%


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
aim_variable = SettingForFeatures.return_aim_variable_ph()

# %%
n_splits = 10

# %%


# %% [markdown]
# ### Test

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

with open(f"./src/analyses/config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

print(params)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

r2_list = []
for train_idx, test_idx in kf.split(X):
    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(rmse, mae, r2)
    r2_list.append(r2)

r2_ols_list = []
for train_idx, test_idx in kf.split(X):
    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(rmse, mae, r2)
    r2_ols_list.append(r2)

accuracy_comparison = params
accuracy_comparison['XGBoost Mean R2'] = np.mean(r2_list).astype(float)
accuracy_comparison['XGBoost SD R2'] = np.std(r2_list).astype(float)
accuracy_comparison['XGBoost Min R2'] = np.min(r2_list).astype(float)
accuracy_comparison['XGBoost Max R2'] = np.max(r2_list).astype(float)

accuracy_comparison['OLS Mean R2'] = np.mean(r2_ols_list).astype(float)
accuracy_comparison['OLS SD R2'] = np.std(r2_ols_list).astype(float)
accuracy_comparison['OLS Min R2'] = np.min(r2_ols_list).astype(float)
accuracy_comparison['OLS Max R2'] = np.max(r2_ols_list).astype(float)

save_path = os.path.join('data/exp/model_outputs', f'accuracy_comparison.json')
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(
        accuracy_comparison,
        f,
        default=json_serializable,
        ensure_ascii=False,
        indent=4
    )

# %%


# %% [markdown]
# ### Check Importance

# %%
X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

with open(f"./src/analyses/config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

print(params)

feature_importance_df_list = []
fold = 1

for train_idx, test_idx in kf.split(X):

    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # Importance
    booster = model.get_booster()
    score = booster.get_score(importance_type='gain')

    # Convert to aligned DF: feature as index, importance as column
    df_imp = pd.DataFrame(score, index=[0]).T
    df_imp.columns = [f"fold_{fold}"]

    feature_importance_df_list.append(df_imp)
    
    fold += 1

# 🔥 Merge all folds by feature name
feature_importance_full = pd.concat(feature_importance_df_list, axis=1).fillna(0)

#print(feature_importance_full)

feature_importance_full['mean_importance'] = feature_importance_full.mean(axis = 1)
feature_importance_full = feature_importance_full.sort_values('mean_importance')

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
plt.gca().invert_yaxis()
plt.xlabel("Gain Importance")
plt.title("XGBoost Feature Importance")
plt.show()

feature_importance_full.to_csv(os.path.join('data/exp/model_outputs', f'importance.csv'))

# %%


# %% [markdown]
# ### Check PDP

# %%
def plot_MEAN(aim_variable, varname):
    potenital_values = np.load(os.path.join('data/exp/model_outputs', f'{aim_variable}_potenital_values_{varname.upper()}_MEAN.npy'))
    pdp_array = np.load(os.path.join('data/exp/model_outputs', f'{aim_variable}_pdp_array_{varname.upper()}_MEAN.npy'))

    pdp_mean = np.mean(pdp_array, axis = 0)
    pdp_std = np.std(pdp_array, axis = 0)

    # 1. Plot the mean PDP line (same as before)
    plt.plot(potenital_values, pdp_mean, linewidth=2, label="Mean Prediction")
    
    # 2. Add the Error/Confidence Band using fill_between
    # The band represents [mean - std] to [mean + std]
    plt.fill_between(
        potenital_values, 
        pdp_mean - pdp_std * 1.96,  # Lower bound
        pdp_mean + pdp_std * 1.96,  # Upper bound
        color='gray',        # Color of the shaded area
        alpha=0.3,           # Transparency
        label="$\pm 1.96 \sigma$" # Label for the legend
    )
    
    # Optional: Add labels and grid
    plt.xlabel("Annual Mean LAI")
    plt.ylabel(f"Predicted {aim_variable}")
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(FIGURES, f'fig05_{aim_variable}_PDP_{varname.upper()}_MEAN.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
def plot_STD(aim_variable, varname):
    potenital_values = np.load(os.path.join('data/exp/model_outputs',f'{aim_variable}_potenital_values_{varname.upper()}_STD.npy'))
    pdp_array = np.load(os.path.join('data/exp/model_outputs',f'{aim_variable}_pdp_array_{varname.upper()}_STD.npy'))

    pdp_mean = np.mean(pdp_array, axis = 0)
    pdp_std = np.std(pdp_array, axis = 0)

    # 1. Plot the mean PDP line (same as before)
    plt.plot(potenital_values, pdp_mean, linewidth=2, label="Mean Prediction")
    
    # 2. Add the Error/Confidence Band using fill_between
    # The band represents [mean - std] to [mean + std]
    plt.fill_between(
        potenital_values, 
        pdp_mean - pdp_std * 1.96,  # Lower bound
        pdp_mean + pdp_std * 1.96,  # Upper bound
        color='gray',        # Color of the shaded area
        alpha=0.3,           # Transparency
        label="$\pm 1.96 \sigma$" # Label for the legend
    )
    
    # Optional: Add labels and grid
    plt.xlabel("Annual Standard Deviation of LAI")
    plt.ylabel(f"Predicted {aim_variable}")
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(FIGURES, f'fig06_{aim_variable}_PDP_{varname.upper()}_STD.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
varname = 'tas'

X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = aim_variable,
)

with open(f"./src/analyses/config/params.yaml", "r") as f:
    params = yaml.safe_load(f)

print(params)

model_list = Modelling.get_regmodel_list(
    X, y,
    n_splits = n_splits,
    params = params,
)

potenital_values, pdp_array = ExplainResult.compute_single_pdp_self_defined(
    var = f'{varname.upper()}_MEAN',
    X = X,
    model_list = model_list,
    range_boundary = (0.05, 0.95),
    stripe = 0.1
)

np.save(os.path.join('data/exp/model_outputs',f'{aim_variable}_potenital_values_{varname.upper()}_MEAN.npy'), potenital_values)
np.save(os.path.join('data/exp/model_outputs',f'{aim_variable}_pdp_array_{varname.upper()}_MEAN.npy'), pdp_array)

os.makedirs(FIGURES := "./data/results/figures", exist_ok = True)

plot_MEAN(aim_variable, varname)

potenital_values, pdp_array = ExplainResult.compute_single_pdp_self_defined(
    var = f'{varname.upper()}_STD',
    X = X,
    model_list = model_list,
    range_boundary = (0.05, 0.95),
    stripe = 0.1
)

np.save(os.path.join('data/exp/model_outputs',f'{aim_variable}_potenital_values_{varname.upper()}_STD.npy'), potenital_values)
np.save(os.path.join('data/exp/model_outputs',f'{aim_variable}_pdp_array_{varname.upper()}_STD.npy'), pdp_array)    

plot_STD(aim_variable, varname)

# %%
X['TAS_MEAN'].describe()

# %%


# %%


# %%


# %%


# %%

