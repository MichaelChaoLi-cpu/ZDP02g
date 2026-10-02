# Python source converted from src/analyses/notebooks/c03_investigate_on_tas.ipynb
# Code-cell order and Markdown explanations are preserved.

# Resolve project paths before importing the analysis helper modules.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_PROJECT_ROOT / "src/analyses/notebooks"))
_os.chdir(_PROJECT_ROOT)


# %% [markdown]
# # Investigation of Laef

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from itertools import combinations, chain
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# %%
import Modelling

# %%
import importlib
importlib.reload(Modelling)

# %%


# %% [markdown]
# ## Functions

# %%
def random_search_aim_cls(
    all_data : pd.DataFrame,
    aim_variable: str,
    always_inputs: list,
    reg_params : dict,
    n_splits : int = 10
): 
    all_data_use =  all_data[
        always_inputs + [aim_variable]
    ]
    X = all_data_use.drop(columns = aim_variable)
    y = all_data_use[aim_variable]

    _ = Modelling.xgb_cls_kfold_cv(
        X, y,
        n_splits=n_splits,
        params=reg_params,
        log_dir="data/exp/logs",
        log_file="xgb_cls_booming.csv"
    )

# %%
def random_search_aim(
    all_data : pd.DataFrame,
    aim_variable: str,
    always_inputs: list,
    reg_params : dict,
    n_splits : int = 10
): 
    all_data_use =  all_data[
        always_inputs + [aim_variable]
    ]
    X = all_data_use.drop(columns = aim_variable)
    y = all_data_use[aim_variable]

    _ = Modelling.xgb_reg_kfold_cv(
        X, y,
        n_splits=n_splits,
        params=reg_params,
        log_dir="data/exp/logs",
        log_file="xgb_reg_booming.csv"
    )

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
gfs_dataset = pd.read_parquet(os.path.join('data', 'processed', 'GlobalFlourishingDataWithLonLit.parquet'))
df_out_2023 = pd.read_parquet('data/processed/df_out_2023.parquet')
df_out_2024 = pd.read_parquet('data/processed/df_out_2024.parquet')
df_out_2050 = pd.read_parquet('data/processed/df_out_2050.parquet')
df_out_2070 = pd.read_parquet('data/processed/df_out_2070.parquet')
df_out_2100 = pd.read_parquet('data/processed/df_out_2100.parquet')

# %%
var_name = 'tas'

# %%
df_out_2023['WAVE'] = 1.0
df_out_2024['WAVE'] = 2.0
df_out_2023 = df_out_2023[['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name}_mean_2023', f'{var_name}_std_2023']]
df_out_2023.columns = ['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name.upper()}_MEAN', f'{var_name.upper()}_STD']
df_out_2024 = df_out_2024[['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name}_mean_2024', f'{var_name}_std_2024']]
df_out_2024.columns = ['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name.upper()}_MEAN', f'{var_name.upper()}_STD']
df_temp_now = pd.concat([df_out_2023, df_out_2024], axis = 0).reset_index(drop=True)

# %%
gfs_temp_dataset = gfs_dataset.merge(df_temp_now, on = ['LATITUDE', 'LONGITUDE', 'WAVE'])

# %%
gfs_temp_dataset.shape

# %%
gfs_temp_dataset.columns

# %%
all_data = gfs_temp_dataset.drop(columns = ['ID']).dropna()

# %%
all_data.columns

# %%
always_inputs = [
    'INCOME_REAL', 'COUNTRY', 'WAVE', 'LATITUDE', 'LONGITUDE', 'AGE',
    'EMPLOYMENT', 'MARITAL_STATUS', 'HAVE_CHILD', 'NUM_HOUSEHOLD_Y1',
    'OWN_RENT_HOME_Y1', 'URBAN_RURAL', 'EXPENSES', 'GENDER', 'EDUCATION_3',
    'CLOSE_TO', 
    f'{var_name.upper()}_MEAN', f'{var_name.upper()}_STD'
]

# %%
mediate = ['BODILY_PAIN',  'HEALTH_PROB', 'DAYS_EXERCISE',]

# %%
potential_output = ['PHYSICAL_HLTH']

# %%
reg_params = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda"
}

# %%
cls_params = {
    "objective": "binary:logistic",   
    "eval_metric": "logloss",          
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda"
}

# %%


# %% [markdown]
# ### Basic Importance check

# %% [markdown]
# #### output

# %%
n_splits = 10

for aim_variable in potential_output:
    X, y = Modelling.prepare_data(
        all_data = all_data,
        always_inputs = always_inputs + mediate,
        aim_variable = aim_variable,
    )
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    feature_importance_df_list = []
    fold = 1
    
    for train_idx, test_idx in kf.split(X):
    
        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
        # Train
        model = xgb.XGBRegressor(**reg_params)
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
    
    feature_importance_full['mean_importance'] = feature_importance_full.mean(axis = 1)
    feature_importance_full = feature_importance_full.sort_values('mean_importance')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain Importance")
    plt.title("XGBoost Feature Importance")
    plt.show()

# %% [markdown]
# #### BODILY_PAIN

# %%
mediate[0]

# %%
n_splits = 10

X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = mediate[0],
)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

feature_importance_df_list = []
fold = 1

for train_idx, test_idx in kf.split(X):

    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Ensure binary labels {0,1}
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ---- imbalance handling: scale_pos_weight per fold ----
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    
    if n_pos == 0 or n_neg == 0:
        print(f"Fold {fold}: skipped (n_pos={n_pos}, n_neg={n_neg})")
        fold += 1
        continue

    spw = n_neg / n_pos

    params = copy.deepcopy(cls_params)
    params["scale_pos_weight"] = spw
    
    # Train classifier
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1
    y_pred = (y_prob >= 0.5).astype(int)         # default threshold = 0.5
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)         # recall for positive class (1)
    precision = precision_score(y_test, y_pred)   # precision for positive class (1)
    f1 = f1_score(y_test, y_pred)
    
    print(
        f"Fold {fold}: "
        f"ACC={acc:.4f}, "
        f"Recall={recall:.4f}, "
        f"Precision={precision:.4f}, "
        f"F1={f1:.4f}"
    )
    
    # Feature importance (unchanged)
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    
    df_imp = pd.DataFrame(score, index=[0]).T
    df_imp.columns = [f"fold_{fold}"]

    feature_importance_df_list.append(df_imp)
    
    fold += 1

# 🔥 Merge all folds by feature name
feature_importance_full = pd.concat(feature_importance_df_list, axis=1).fillna(0)

feature_importance_full['mean_importance'] = feature_importance_full.mean(axis = 1)
feature_importance_full = feature_importance_full.sort_values('mean_importance')

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
plt.gca().invert_yaxis()
plt.xlabel("Gain Importance")
plt.title("XGBoost Feature Importance")
plt.show()

# %% [markdown]
# #### HEALTH_PROB

# %%
mediate[1]

# %%
n_splits = 10

X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = mediate[1],
)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

feature_importance_df_list = []
fold = 1

for train_idx, test_idx in kf.split(X):

    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Ensure binary labels {0,1}
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ---- imbalance handling: scale_pos_weight per fold ----
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    
    if n_pos == 0 or n_neg == 0:
        print(f"Fold {fold}: skipped (n_pos={n_pos}, n_neg={n_neg})")
        fold += 1
        continue

    spw = n_neg / n_pos

    params = copy.deepcopy(cls_params)
    params["scale_pos_weight"] = spw
    
    # Train classifier
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1
    y_pred = (y_prob >= 0.5).astype(int)         # default threshold = 0.5
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)         # recall for positive class (1)
    precision = precision_score(y_test, y_pred)   # precision for positive class (1)
    f1 = f1_score(y_test, y_pred)
    
    print(
        f"Fold {fold}: "
        f"ACC={acc:.4f}, "
        f"Recall={recall:.4f}, "
        f"Precision={precision:.4f}, "
        f"F1={f1:.4f}"
    )
    
    # Feature importance (unchanged)
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    
    df_imp = pd.DataFrame(score, index=[0]).T
    df_imp.columns = [f"fold_{fold}"]

    feature_importance_df_list.append(df_imp)
    
    fold += 1

# 🔥 Merge all folds by feature name
feature_importance_full = pd.concat(feature_importance_df_list, axis=1).fillna(0)

feature_importance_full['mean_importance'] = feature_importance_full.mean(axis = 1)
feature_importance_full = feature_importance_full.sort_values('mean_importance')

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
plt.gca().invert_yaxis()
plt.xlabel("Gain Importance")
plt.title("XGBoost Feature Importance")
plt.show()

# %%


# %% [markdown]
# #### DAYS_EXERCISE

# %%
mediate[2]

# %%
n_splits = 10

X, y = Modelling.prepare_data(
    all_data = all_data,
    always_inputs = always_inputs,
    aim_variable = mediate[2],
)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

feature_importance_df_list = []
fold = 1

for train_idx, test_idx in kf.split(X):

    # Split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Ensure binary labels {0,1}
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # ---- imbalance handling: scale_pos_weight per fold ----
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    
    if n_pos == 0 or n_neg == 0:
        print(f"Fold {fold}: skipped (n_pos={n_pos}, n_neg={n_neg})")
        fold += 1
        continue

    spw = n_neg / n_pos

    params = copy.deepcopy(cls_params)
    params["scale_pos_weight"] = spw
    
    # Train classifier
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1
    y_pred = (y_prob >= 0.5).astype(int)         # default threshold = 0.5
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)         # recall for positive class (1)
    precision = precision_score(y_test, y_pred)   # precision for positive class (1)
    f1 = f1_score(y_test, y_pred)
    
    print(
        f"Fold {fold}: "
        f"ACC={acc:.4f}, "
        f"Recall={recall:.4f}, "
        f"Precision={precision:.4f}, "
        f"F1={f1:.4f}"
    )
    
    # Feature importance (unchanged)
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    
    df_imp = pd.DataFrame(score, index=[0]).T
    df_imp.columns = [f"fold_{fold}"]

    feature_importance_df_list.append(df_imp)
    
    fold += 1

# 🔥 Merge all folds by feature name
feature_importance_full = pd.concat(feature_importance_df_list, axis=1).fillna(0)

feature_importance_full['mean_importance'] = feature_importance_full.mean(axis = 1)
feature_importance_full = feature_importance_full.sort_values('mean_importance')

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_full.index, feature_importance_full["mean_importance"])
plt.gca().invert_yaxis()
plt.xlabel("Gain Importance")
plt.title("XGBoost Feature Importance")
plt.show()

# %%


# %%


# %%


# %% [markdown]
# ### Conclusion

# %%
for this in potential_output:
    value_counts = all_data[this].value_counts().sort_index()
    
    plt.figure(figsize=(10,5))
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f"Frequency of Each Unique Value in {this}")
    plt.xlabel("y value")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.show()

# %%
for this in mediate:
    value_counts = all_data[this].value_counts().sort_index()
    
    plt.figure(figsize=(10,5))
    plt.bar(value_counts.index, value_counts.values)
    plt.title(f"Frequency of Each Unique Value in {this}")
    plt.xlabel("y value")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.show()

# %%

