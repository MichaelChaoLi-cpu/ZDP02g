# %% [markdown]
# # Check Explanation of the relationship between Emotion and LAI

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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
always_inputs = SettingForFeatures.return_always_input_variable_list()

# %%
aim_variables = SettingForFeatures.return_aim_mediate()

# %%
n_splits = 10

# %%

# %% [markdown]
# ### Test

# %%
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )

    with open(f"./{aim_variable}_params.yaml", "r") as f:
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
        y_pred = np.where(y_pred>0.5, 1, 0)
    
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
    
        print(accuracy, precision, recall, f1)
        r2_list.append(accuracy)

    r2_ols_list = []
    for train_idx, test_idx in kf.split(X):
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Train
        model = LogisticRegression(penalty = None, max_iter=1)
        model.fit(X_train, y_train)
    
        # Predict
        y_pred = model.predict(X_test)
    
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
    
        print(accuracy, precision, recall, f1)
        r2_ols_list.append(accuracy)

    accuracy_comparison = params
    accuracy_comparison['XGBoost Mean Accuracy'] = np.mean(r2_list).astype(float)
    accuracy_comparison['XGBoost SD Accuracy'] = np.std(r2_list).astype(float)
    accuracy_comparison['XGBoost Min Accuracy'] = np.min(r2_list).astype(float)
    accuracy_comparison['XGBoost Max Accuracy'] = np.max(r2_list).astype(float)

    accuracy_comparison['OLS Mean Accuracy'] = np.mean(r2_ols_list).astype(float)
    accuracy_comparison['OLS SD Accuracy'] = np.std(r2_ols_list).astype(float)
    accuracy_comparison['OLS Min Accuracy'] = np.min(r2_ols_list).astype(float)
    accuracy_comparison['OLS Max Accuracy'] = np.max(r2_ols_list).astype(float)

    save_path = os.path.join('results', f'{aim_variable}_accuracy_comparison.json')
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
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )

    with open(f"./{aim_variable}_params.yaml", "r") as f:
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

        # Predict
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred>0.5, 1, 0)
    
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        print(f"Fold {fold}: accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    
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

    feature_importance_full.to_csv(os.path.join('results', f'{aim_variable}_importance.csv'))


# %%

# %% [markdown]
# ### Check PDP

# %%
def plot_MEAN(aim_variable, varname):
    potenital_values = np.load(os.path.join('results', f'{aim_variable}_potenital_values_{varname.upper()}_MEAN.npy'))
    pdp_array = np.load(os.path.join('results', f'{aim_variable}_pdp_array_{varname.upper()}_MEAN.npy'))

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
    plt.xlabel("Annual Mean")
    plt.ylabel(f"Predicted {aim_variable}")
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(FIGURES, f'fig05_{aim_variable}_PDP_{varname.upper()}_MEAN.jpg'), dpi=300, bbox_inches='tight')
    plt.show()


# %%
def plot_STD(aim_variable, varname):
    potenital_values = np.load(os.path.join('results',f'{aim_variable}_potenital_values_{varname.upper()}_STD.npy'))
    pdp_array = np.load(os.path.join('results',f'{aim_variable}_pdp_array_{varname.upper()}_STD.npy'))

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
    plt.xlabel("Annual Standard Deviation")
    plt.ylabel(f"Predicted {aim_variable}")
    plt.grid(True)
    plt.legend()

    plt.savefig(os.path.join(FIGURES, f'fig06_{aim_variable}_PDP_{varname.upper()}_STD.jpg'), dpi=300, bbox_inches='tight')
    plt.show()



# %%
varname = 'tas'
for aim_variable in aim_variables:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs,
        aim_variable = aim_variable,
    )

    with open(f"./{aim_variable}_params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    print(params)

    model_list = Modelling.get_clsmodel_list(
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

    np.save(os.path.join('results',f'{aim_variable}_potenital_values_{varname.upper()}_MEAN.npy'), potenital_values)
    np.save(os.path.join('results',f'{aim_variable}_pdp_array_{varname.upper()}_MEAN.npy'), pdp_array)

    os.makedirs(FIGURES := "./figures", exist_ok = True)

    plot_MEAN(aim_variable, varname)

    potenital_values, pdp_array = ExplainResult.compute_single_pdp_self_defined(
        var = f'{varname.upper()}_STD',
        X = X,
        model_list = model_list,
        range_boundary = (0.05, 0.95),
        stripe = 0.1
    )

    np.save(os.path.join('results',f'{aim_variable}_potenital_values_{varname.upper()}_STD.npy'), potenital_values)
    np.save(os.path.join('results',f'{aim_variable}_pdp_array_{varname.upper()}_STD.npy'), pdp_array)    

    plot_STD(aim_variable, varname)

# %%
for aim_variable in aim_variables:
    plot_MEAN(aim_variable, varname)

# %%

# %%

# %%

# %%
