import matplotlib.pyplot as plt
import numpy as np

from typing import Optional

def visualize_map_array(
    data_to_visual: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "bwr",
    title: str = "Spatial Map"
) -> None:
    """
    Visualize a 2D spatial array using matplotlib with an optional symmetric color scale.

    This function displays a 2D map (e.g., temperature anomaly, ratio, residual)
    using a diverging colormap centered around zero when vmin/vmax are not specified.

    Parameters
    ----------
    data_to_visual : np.ndarray
        2D array to visualize (e.g., 3600×7200 MODIS grid or 256×512 CMIP6 grid).
    vmin : float, optional
        Minimum value for color normalization. If None, automatically determined.
    vmax : float, optional
        Maximum value for color normalization. If None, automatically determined.
    cmap : str, optional
        Matplotlib colormap name (default: "bwr").
    title : str, optional
        Title of the plot.

    Notes
    -----
    - NaN values are automatically ignored during color scaling.
    - Uses a high DPI (300) and large figure size for detailed inspection.
    - The color scale defaults to symmetric bounds around 0 if both vmin and vmax are None.
    """

    data_to_visual = np.asarray(data_to_visual)
    assert data_to_visual.ndim == 2, "Input must be a 2D array."

    # Determine symmetric color range if none is provided
    if (vmin is None) and (vmax is None):
        vmax = np.nanmax(np.abs(data_to_visual))
        vmin = -vmax

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    # Display the data
    im = ax.imshow(
        data_to_visual,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value", rotation=270, labelpad=10)

    # Configure plot aesthetics
    ax.set_title(title, fontsize=14)
    ax.grid(False)

    plt.tight_layout()
    plt.show()

def plot_histogram(
    data_to_visual: np.ndarray,
    bins: int = 50,
    title: str = "Histogram of Values",
    xlabel: str = "Temperature",
    ylabel: str = "Frequency"
) -> None:
    """
    Plot a histogram of 2D or 3D array values, ignoring NaNs.

    This function flattens the input array (after removing NaN values),
    then plots a histogram with summary statistics annotated on the figure.

    Parameters
    ----------
    data_to_visual : np.ndarray
        Input array containing numeric values (can be 1D, 2D, or 3D).
    bins : int, optional
        Number of histogram bins (default: 50).
    title : str, optional
        Title of the plot (default: "Histogram of Values").
    xlabel : str, optional
        Label for the x-axis (default: "Temperature").
    ylabel : str, optional
        Label for the y-axis (default: "Frequency").

    Notes
    -----
    - NaN values are removed before computing the histogram.
    - The mean and standard deviation are displayed in the upper-right corner.
    """

    # Remove NaN values and flatten to 1D
    data_flat = data_to_visual[~np.isnan(data_to_visual)].ravel()

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    ax.hist(data_flat, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)

    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Compute and display summary statistics
    mean_val = np.mean(data_flat)
    std_val = np.std(data_flat)
    ax.text(
        0.98, 0.95,
        f"Mean = {mean_val:.3f}\nStd = {std_val:.3f}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()

