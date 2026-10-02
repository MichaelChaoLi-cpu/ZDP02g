import numpy as np
import os, time, json
import pandas as pd
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def xgb_reg_kfold_cv(
    X, y,
    n_splits=10,
    params=None,
    log_dir="data/exp/logs",
    log_file="xgb_cv_results.csv"
):
    """
    K-fold cross-validation with full logging:
    - Timestamps
    - Fold-level training time
    - XGBoost parameters
    - GPU usage
    - Feature names (X.columns)
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    logs = []
    fold = 1

    # Record feature names once
    feature_names = list(X.columns)
    feature_names_json = json.dumps(feature_names)
    output_name = y.name

    print("\n🚀 Starting XGBoost K-Fold CV...\n")

    for train_idx, test_idx in kf.split(X):

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        elapsed = time.time() - start_time

        logs.append({
            "timestamp": timestamp,
            "fold": fold,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "training_time_sec": elapsed,
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(feature_names),
            "feature_names": feature_names_json,
            "output_name" : output_name,
            "xgb_params": json.dumps(params)
        })

        fold += 1

    print(f"Fold {fold} → RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Time={elapsed:.2f}s")
    
    # Save CSV (append mode)
    df_log = pd.DataFrame(logs)

    df_log.to_csv(
        log_path,
        index=False,
        mode='a',
        header=not os.path.exists(log_path)
    )

    print(f"\n✅ CV completed. Log saved to: {log_path}")

    return df_log


def xgb_cls_kfold_cv(
    X, y,
    n_splits=10,
    params=None,
    log_dir="data/exp/logs",
    log_file="xgb_cls_cv_results.csv"
):
    """
    K-fold cross-validation for XGBoost classification with full logging:
    - timestamps
    - fold-level training time
    - XGBoost parameters
    - GPU usage
    - feature names
    - classification metrics (accuracy, F1, precision, recall, AUC)
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    logs = []
    fold = 1

    feature_names = list(X.columns)
    feature_names_json = json.dumps(feature_names)
    output_name = y.name

    print("\n🚀 Starting XGBoost Classification K-Fold CV...\n")

    for train_idx, test_idx in kf.split(X):

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize classifier
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict class labels
        y_pred = model.predict(X_test)


        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        elapsed = time.time() - start_time

        logs.append({
            "timestamp": timestamp,
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "training_time_sec": elapsed,
            "n_samples_train": len(X_train),
            "n_samples_test": len(X_test),
            "n_features": len(feature_names),
            "feature_names": feature_names_json,
            "output_name": output_name,
            "xgb_params": json.dumps(params)
        })

        fold += 1
    print(f"ACC={accuracy:.4f}, F1={f1:.4f}, PREC={precision:.4f}, RECALL={recall:.4f}, Time={elapsed:.2f}s")
    
    df_log = pd.DataFrame(logs)

    df_log.to_csv(
        log_path,
        index=False,
        mode='a',
        header=not os.path.exists(log_path)
    )

    print(f"\n✅ Classification CV completed. Log saved to: {log_path}")

    return df_log

def prepare_data(
    all_data: pd.DataFrame,
    always_inputs: list,
    aim_variable: str
) -> tuple:
    """
    Prepare feature matrix X and target vector y for supervised learning.

    This function:
        1. Selects the required predictors (always_inputs) and the target variable.
        2. Validates that all required columns exist in the dataset.
        3. Removes rows containing missing values in selected variables.
        4. Splits the cleaned dataset into:
               X : feature matrix
               y : target variable

    Parameters
    ----------
    all_data : pd.DataFrame
        Full dataset containing predictors and the target variable.

    always_inputs : list
        List of predictor variable names that must always be included.

    aim_variable : str
        Name of the target variable (dependent variable).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing only the selected predictors.

    y : pd.Series
        Target variable.

    Raises
    ------
    ValueError
        If required variables are missing in the dataset.

    """

    # Ensure required columns exist
    required_columns = set(always_inputs + [aim_variable])
    missing_cols = required_columns - set(all_data.columns)

    if missing_cols:
        raise ValueError(f"The following required columns are missing: {missing_cols}")

    # Subset and drop NA
    all_data_use = all_data[always_inputs + [aim_variable]].dropna().copy()

    # Split into X and y
    X = all_data_use.drop(columns=aim_variable)
    y = all_data_use[aim_variable]

    return X, y

def get_regmodel_list(
    X, y,
    n_splits=10,
    params=None,
) -> list:
    """
    Train an XGBoost classifier across K folds and return the list of fitted models.

    This function performs a standard K-fold split of the dataset, trains one
    XGBoost model on each training fold, and stores all fitted models in order.
    It can be used for model ensembling, uncertainty estimation, or stability
    analysis of classifier behavior across different data partitions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used for training the classifier.

    y : pd.Series
        Target vector for supervised learning.

    n_splits : int, default 10
        Number of folds to use in K-Fold cross-validation.

    params : dict, optional
        Dictionary of hyperparameters for initializing XGBClassifier.
        Example fields may include:
            - n_estimators
            - learning_rate
            - max_depth
            - subsample
            - colsample_bytree
            - objective
            - device / tree_method
        If None, default XGBoost parameters are used.

    Returns
    -------
    list
        A list of trained XGBClassifier models, one for each fold.
        The i-th element corresponds to the model trained on the i-th split.

    Notes
    -----
    - The dataset is shuffled before splitting to ensure random fold assignment.
    - No predictions or evaluation metrics are computed inside this function.
    - The returned model list is suitable for:
        * Majority-vote / averaging ensembling
        * Feature importance aggregation across folds
        * Stability and robustness checks
        * Out-of-fold inference pipelines
    """
    model_list = [] 
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize classifier
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        model_list.append(model)

    return model_list

def get_clsmodel_list(
    X, y,
    n_splits=10,
    params=None,
) -> list:
    """
    Train an XGBoost classifier across K folds and return the list of fitted models.

    This function performs a standard K-fold split of the dataset, trains one
    XGBoost model on each training fold, and stores all fitted models in order.
    It can be used for model ensembling, uncertainty estimation, or stability
    analysis of classifier behavior across different data partitions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used for training the classifier.

    y : pd.Series
        Target vector for supervised learning.

    n_splits : int, default 10
        Number of folds to use in K-Fold cross-validation.

    params : dict, optional
        Dictionary of hyperparameters for initializing XGBClassifier.
        Example fields may include:
            - n_estimators
            - learning_rate
            - max_depth
            - subsample
            - colsample_bytree
            - objective
            - device / tree_method
        If None, default XGBoost parameters are used.

    Returns
    -------
    list
        A list of trained XGBClassifier models, one for each fold.
        The i-th element corresponds to the model trained on the i-th split.

    Notes
    -----
    - The dataset is shuffled before splitting to ensure random fold assignment.
    - No predictions or evaluation metrics are computed inside this function.
    - The returned model list is suitable for:
        * Majority-vote / averaging ensembling
        * Feature importance aggregation across folds
        * Stability and robustness checks
        * Out-of-fold inference pipelines
    """
    model_list = [] 
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize classifier
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        model_list.append(model)

    return model_list