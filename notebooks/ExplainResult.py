import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from sklearn.inspection import partial_dependence

def compute_pdp_single(
    feature : str,
    model : object,
    X : pd.DataFrame, 
    grid_resolution=50 
) -> dict:
    """
    Computes and plots the Partial Dependence Plot (PDP) for a single feature.

    The Partial Dependence Plot shows the marginal effect of a specific feature 
    on the predicted outcome of a trained machine learning model.

    

    Parameters
    ----------
    feature : str
        The name of the feature (column in X) for which to compute the PDP.
    model : object
        The fitted machine learning estimator (e.g., scikit-learn model) 
        that implements 'predict' or 'predict_proba'.
    X : pd.DataFrame
        The data used to train the model, containing all features. 
        It is used to determine the feature values (grid) over which the 
        partial dependence is computed.
    grid_resolution : int, optional
        The number of equidistant points (grid points) between the minimum and 
        maximum values of the feature used to compute the partial dependence. 
        (default is 50).

    Returns
    -------
    dict
        A dictionary containing the results of the partial dependence computation,
        specifically the feature grid values ('grid_values') and the 
        corresponding averaged predictions ('average').

    Notes
    -----
    This function internally uses `sklearn.inspection.partial_dependence`. 
    It also generates and displays a matplotlib plot of the resulting PDP.
    """
    feature_index = list(X.columns).index(feature)

    pdp_result = partial_dependence(
        estimator=model,
        X=X,
        features=[feature_index],
        grid_resolution=grid_resolution 
    )
    
    xs = pdp_result["grid_values"][0]
    ys = pdp_result["average"][0]
    
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel(feature)
    plt.ylabel("Predicted Response")
    plt.title(f"PDP for {feature}")
    plt.grid(True)
    plt.show()

    return pdp_result

def compute_pdp_interaction(
    feature1: str,
    feature2: str,
    model: object,
    X: pd.DataFrame,
    grid_resolution: int = 30,
    plot_3d: bool = False
) -> dict:
    """
    Computes and plots the 2D Partial Dependence Plot (PDP) for two features 
    to visualize feature interaction.

    The 2D PDP (or interaction PDP) shows the joint marginal effect of two 
    features on the predicted outcome of a trained machine learning model, 
    revealing potential interaction effects. 

    Parameters
    ----------
    feature1 : str
        The name of the first feature (column in X).
    feature2 : str
        The name of the second feature (column in X).
    model : object
        The fitted machine learning estimator (e.g., scikit-learn model) 
        that implements 'predict' or 'predict_proba'.
    X : pd.DataFrame
        The data used to train the model, containing all features. 
        It is used to determine the feature values (grid) over which the 
        partial dependence is computed.
    grid_resolution : int, optional
        The number of equidistant points (grid points) between the minimum and 
        maximum values for *each* feature used to compute the partial dependence. 
        The total number of grid points will be grid_resolution * grid_resolution.
        (default is 30).
    plot_3d : bool, optional
        If True, an additional 3D surface plot is generated to visualize the 
        interaction surface. (default is False). 

    Returns
    -------
    dict
        A dictionary containing the results of the partial dependence computation,
        specifically the feature grid values ('grid_values') and the 
        corresponding averaged predictions ('average').

    Notes
    -----
    This function internally uses `sklearn.inspection.partial_dependence` with 
    a tuple of features to compute the interaction. 
    It generates a matplotlib contour plot by default and an optional 3D surface plot.
    """

    f1 = list(X.columns).index(feature1)
    f2 = list(X.columns).index(feature2)

    pdp = partial_dependence(
        estimator=model,
        X=X,
        features=[(f1, f2)],
        grid_resolution=grid_resolution
    )

    xs = pdp["grid_values"][0]   # grid for feature1
    ys = pdp["grid_values"][1]   # grid for feature2
    zz = pdp["average"][0].reshape(len(xs), len(ys))

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(xs, ys, zz.T, cmap="viridis", levels=20)
    plt.colorbar(cp)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"2D PDP Interaction: {feature1} × {feature2}")
    plt.show()

    # --- Optional 3D Surface Plot ---
    if plot_3d:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        Xs, Ys = np.meshgrid(xs, ys)

        ax.plot_surface(Xs, Ys, zz.T, cmap="viridis", edgecolor="none")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlabel("Predicted Response")
        ax.set_title(f"3D Surface PDP: {feature1} × {feature2}")
        plt.show()
    
    return pdp

def compute_single_pdp_self_defined(
    var: str,
    X: pd.DataFrame,
    model_list: list,
    range_boundary=(0.05, 0.95),
    stripe: float = 0.2
) -> tuple:
    """
    Manually computes the Partial Dependence Plot (PDP) for a single feature 
    across multiple models by iterating through a custom grid of feature values.

    This function calculates the average predicted response by fixing a single 
    feature to specific 'potential values' while marginalizing (averaging) 
    over the distribution of all other features in the dataset X.

    Parameters
    ----------
    var : str
        The name of the target feature (column in X) for which to compute the PDP.
    X : pd.DataFrame
        The reference dataset containing all features used for model training. 
        It is used to define the grid range and for prediction.
    model_list : list
        A list of fitted machine learning estimators (e.g., scikit-learn models) 
        that implement a 'predict' method.
    range_boundary : tuple, optional
        The lower and upper quantiles (0.0 to 1.0) of the target feature's 
        distribution in X used to define the grid start and stop points.
        (default is (0.05, 0.95)).
    stripe : float, optional
        The step size (interval) used to generate the discrete grid of 
        'potential values' between the lower and upper bounds. (default is 0.2).

    Returns
    -------
    potenital_values : np.ndarray
        The array of discrete grid points for the target feature.
    pdp_array : np.ndarray
        A 2D array of shape (N_models, N_grid_points) containing the mean 
        predicted response for each model at each grid point.

    Notes
    -----
    The grid boundaries are determined by rounding up the lower quantile and 
    rounding down the upper quantile of the feature 'var' in X.
    
    The output `pdp_array` can be used to plot the mean PDP and to calculate 
    the variance or confidence interval across the ensemble of models.
    """
    # Determine the grid boundaries based on the specified quantiles
    low_b = np.ceil(X[var].quantile(range_boundary[0]))
    up_b = np.floor(X[var].quantile(range_boundary[1]))
    
    # Generate the discrete grid of feature values
    potenital_values = np.arange(low_b, up_b, stripe)
    X_adjust = X.copy()
    
    pdp_list = []
    
    # Iterate through each model in the ensemble/list
    for model in model_list:
        pdp = np.full_like(potenital_values, fill_value=np.nan)
        
        # Iterate through each grid point (potential value)
        for idx, potenital_value in enumerate(potenital_values):
            # 1. Substitute the feature column with the current fixed value
            X_adjust[var] = potenital_value
            
            # 2. Predict the outcome for the entire adjusted dataset
            y_pred = model.predict(X_adjust)
            
            # 3. Calculate the partial dependence (average prediction)
            pdp[idx] = np.mean(y_pred)
        
        pdp_list.append(pdp)

    pdp_array = np.array(pdp_list)

    return potenital_values, pdp_array

def compute_two_feature_pdp_self_defined(
    var_1: str,
    var_2: str,
    X: pd.DataFrame,
    model_list: list,
    range_boundary=(0.05, 0.95),
    stripe: float = 0.2
) -> tuple:
    """
    Manually computes the 2D Partial Dependence Plot (Interaction PDP) for 
    a pair of features across multiple models.

    This function calculates the average predicted response by fixing two 
    features to specific grid values while marginalizing (averaging) over the 
    distribution of all other features in the dataset X.

    Parameters
    ----------
    var_1 : str
        The name of the first feature (column in X).
    var_2 : str
        The name of the second feature (column in X).
    X : pd.DataFrame
        The reference dataset containing all features.
    model_list : list
        A list of fitted machine learning estimators.
    range_boundary : tuple, optional
        The lower and upper quantiles (0.0 to 1.0) used to define the grid 
        start and stop points for BOTH features. (default is (0.05, 0.95)).
    stripe : float, optional
        The step size (interval) used to generate the discrete grid for 
        BOTH features. (default is 0.2).

    Returns
    -------
    potenital_values_1 : np.ndarray
        The array of discrete grid points for the first feature (var_1).
    potenital_values_2 : np.ndarray
        The array of discrete grid points for the second feature (var_2).
    pdp_array : np.ndarray
        A 3D array of shape (N_models, N_grid_1, N_grid_2) containing the 
        mean predicted response for each model at each grid point combination.
    """
    
    # 1. Determine Grid Boundaries and Create Grid for var_1
    low_b_1 = np.ceil(X[var_1].quantile(range_boundary[0]))
    up_b_1 = np.floor(X[var_1].quantile(range_boundary[1]))
    potenital_values_1 = np.arange(low_b_1, up_b_1, stripe)

    # 2. Determine Grid Boundaries and Create Grid for var_2
    low_b_2 = np.ceil(X[var_2].quantile(range_boundary[0]))
    up_b_2 = np.floor(X[var_2].quantile(range_boundary[1]))
    potenital_values_2 = np.arange(low_b_2, up_b_2, stripe)

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
                X_adjust[var_2] = potential_value_2
                
                # Predict and store the average prediction for this grid cell (var_1, var_2)
                y_pred = model.predict(X_adjust)
                pdp[idx_1, idx_2] = np.mean(y_pred)
        
        pdp_list.append(pdp)

    # Combine results into a 3D array (N_models, N_grid_1, N_grid_2)
    pdp_array = np.array(pdp_list)

    return potenital_values_1, potenital_values_2, pdp_array

def compute_single_pdp_self_defined(
    var: str,
    X: pd.DataFrame,
    model_list: list,
    range_boundary=(0.05, 0.95),
    stripe: float = 0.2
) -> tuple:
    # Determine the grid boundaries based on the specified quantiles
    low_b = np.ceil(X[var].quantile(range_boundary[0]))
    up_b = np.floor(X[var].quantile(range_boundary[1]))
    
    # Generate the discrete grid of feature values
    potenital_values = np.arange(low_b, up_b, stripe)
    X_adjust = X.copy()
    
    pdp_list = []
    
    # Iterate through each model in the ensemble/list
    for model in model_list:
        pdp = np.full_like(potenital_values, fill_value=np.nan)
        
        # Iterate through each grid point (potential value)
        for idx, potenital_value in enumerate(potenital_values):
            # 1. Substitute the feature column with the current fixed value
            X_adjust[var] = potenital_value
            
            # 2. Predict the outcome for the entire adjusted dataset
            y_pred = model.predict(X_adjust)
            
            # 3. Calculate the partial dependence (average prediction)
            pdp[idx] = np.mean(y_pred)
        
        pdp_list.append(pdp)

    pdp_array = np.array(pdp_list)

    return potenital_values, pdp_array