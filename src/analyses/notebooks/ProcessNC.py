import numpy as np
import pandas as pd

from netCDF4 import Dataset

def nc_to_numpy_nc4(
    nc_path: str, 
    varname: str = "tas"
) -> tuple:
    """
    Load a variable from a NetCDF file (CMIP-style) and return:
    - The data array with corrected latitude and longitude orientation
    - The latitude array (ascending: South → North)
    - The longitude array (in the range -180 → 180)

    The function automatically:
        1. Loads the specified variable into a NumPy ndarray
        2. Flips latitude if it is descending (e.g., 90 → -90)
        3. Converts longitude from 0–360 to -180–180 if needed
        4. Sorts the array by the corrected longitude

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF (.nc) file.

    varname : str, default "tas"
        Name of the variable to extract from the NetCDF file.

    Returns
    -------
    arr : np.ndarray
        Cleaned climate variable array with shape (time, lat, lon).

    lat : np.ndarray
        Corrected latitude array (ascending order).

    lon : np.ndarray
        Corrected longitude array (in -180 → 180 range).
    """


    ds = Dataset(nc_path)
    arr = np.array(ds.variables[varname][:])   # (time, lat, lon)
    lat = ds.variables["lat"][:]
    lon = ds.variables["lon"][:]
    ds.close()

    if lon.max() > 180:
        lon_new = ((lon + 180) % 360) - 180
    else:
        lon_new = lon.copy()

    sort_idx = np.argsort(lon_new)
    lon_new = lon_new[sort_idx]
    arr = arr[:, ::-1, sort_idx]

    return arr

def covert_K_to_C(
    arr: np.ndarray
) -> np.ndarray:
    arr_new = arr - 273.15
    return arr_new

def obtian_unique_location(
    df_addr: str
) -> pd.DataFrame:
    """
    Load a Parquet file and extract the unique geographic locations
    based on LATITUDE and LONGITUDE.

    This function reads a dataset from the specified Parquet file path,
    selects the LATITUDE and LONGITUDE columns, removes duplicate coordinate
    pairs, and returns a DataFrame containing only the distinct locations.

    Parameters
    ----------
    df_addr : str
        The file path to the Parquet data file. The file must contain
        columns named 'LATITUDE' and 'LONGITUDE'.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns, 'LATITUDE' and 'LONGITUDE',
        containing only unique coordinate pairs from the original dataset.

    Notes
    -----
    - Useful for identifying unique spatial points in geospatial datasets.
    - Raises an error if LATITUDE or LONGITUDE columns are missing.
    """

    df = pd.read_parquet(df_addr)
    df_location = df[['LATITUDE', 'LONGITUDE']].drop_duplicates()
    return df_location

def generate_grid_index_from_array(
    one_month_array: np.ndarray
) -> tuple:
    """
    Generate latitude and longitude coordinate grids for a 2D climate
    array representing one monthly field (e.g., tas for a single month).

    The function assumes the array follows a standard CMIP-style spatial
    layout:

        - The first axis (height, H) corresponds to latitude.
        - The second axis (width, W) corresponds to longitude.
        - Latitude decreases from +90° (north pole) to -90° (south pole).
        - Longitude increases from -180° to +180°.

    Using the array shape, the function generates:
        - A latitude grid of length H using linspace(90, -90, H)
        - A longitude grid of length W using linspace(-180, 180, W)

    Parameters
    ----------
    one_month_array : np.ndarray
        A 2D NumPy array with shape (H, W), representing the spatial
        distribution of a climate variable for one month.

    Returns
    -------
    lat_grid : np.ndarray
        A 1D array of length H containing latitude values from
        +90° to -90°.

    lon_grid : np.ndarray
        A 1D array of length W containing longitude values from
        -180° to +180°.

    Notes
    -----
    - This function does not read metadata from a NetCDF file; it
      infers spatial coordinates purely from array dimensions.
    - The output grid matches the required format for 256×512
      global climate arrays used in your CMIP6 workflow.
    """
    
    assert len(one_month_array.shape) == 2, "need a map in 2d array" 

    H, W = one_month_array.shape
    lat_grid = np.linspace(90, -90, H)
    lon_grid = np.linspace(-180, 180, W)

    return lat_grid, lon_grid

def convert_df_to_grid_indices(
    df_location : pd.DataFrame, 
    lat_grid : np.ndarray, 
    lon_grid : np.ndarray
) -> pd.DataFrame:
    """
    Convert geographic coordinates (latitude, longitude) in a DataFrame into
    corresponding grid indices of a CMIP-style climate data array.

    This function takes a table of point locations and maps each location to
    the nearest grid cell in the provided latitude and longitude grids.
    The resulting row and column indices can be directly used to index a
    NumPy climate array with shape (time, lat, lon).

    Parameters
    ----------
    df_location : pd.DataFrame
        A DataFrame containing at least the columns:
        - 'LATITUDE'  : latitude in degrees (-90 to 90)
        - 'LONGITUDE' : longitude in degrees (-180 to 180)

    lat_grid : np.ndarray
        One-dimensional array of latitude values defining the climate grid.
        Must be sorted (ascending or descending is acceptable).

    lon_grid : np.ndarray
        One-dimensional array of longitude values defining the climate grid.
        Must match the ordering of the corresponding climate data array.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with two additional columns:
        - 'lat_idx' : index of the closest latitude in `lat_grid`
        - 'lon_idx' : index of the closest longitude in `lon_grid`

        These indices can be used to extract values from a NumPy array:

            value = arr[:, lat_idx, lon_idx]

    Notes
    -----
    - The matching is done using nearest-neighbor search (`argmin`).
    - This function assumes rectangular latitude-longitude grids.
    - If your climate grid has fixed resolution (e.g., 256x512), you may
      prefer direct analytical index computation for speed.
    """

    lat_points = df_location["LATITUDE"].to_numpy()
    lon_points = df_location["LONGITUDE"].to_numpy()

    lat_idx = np.array([np.abs(lat_grid - la).argmin() for la in lat_points])
    lon_idx = np.array([np.abs(lon_grid - lo).argmin() for lo in lon_points])

    df_out = df_location.copy()
    df_out["lat_idx"] = lat_idx
    df_out["lon_idx"] = lon_idx
    return df_out

def extract_values_single_scenario(
    df : pd.DataFrame, 
    mean_arr : np.ndarray, 
    std_arr : np.ndarray, 
    info : str,
    var_name : str = 'tas'
) -> pd.DataFrame:
    """
    Extract climate statistics (mean and standard deviation) for a single
    scenario or single dataset at each geographic location defined in the
    input DataFrame.

    For each location (lat_idx, lon_idx) in `df`, the function retrieves the
    corresponding values from `mean_arr` and `std_arr`. The two input arrays
    are assumed to be 3-dimensional climate grids with shape:

        (time, lat, lon)

    where `time` typically represents the 12 monthly values for a given year.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that must contain the columns:
            - 'lat_idx' : row index into the latitude dimension
            - 'lon_idx' : column index into the longitude dimension

    mean_arr : np.ndarray
        A NumPy array of shape (time, lat, lon) containing mean values
        for the given climate scenario.

    std_arr : np.ndarray
        A NumPy array of the same shape as `mean_arr`, containing
        standard deviation values for the same scenario.

    info : str
        A label appended to the output column names. For example,
        using `info="2023"` will create columns:
            'tas_mean_2023' and 'tas_std_2023'.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with two additional columns:
            - 'tas_mean_{info}' : list of time-series mean values
            - 'tas_std_{info}'  : list of time-series standard deviations

        Each row contains an array of time-length values (e.g., 12 months).

    Notes
    -----
    - This function uses nearest-grid point extraction, not interpolation.
    - The output columns contain array-like objects (lists or NumPy arrays)
      representing the full time series for each location.
    """

    mean_list = []
    std_list = []

    for lat_i, lon_i in zip(df["lat_idx"], df["lon_idx"]):
        mean_list.append(mean_arr[lat_i, lon_i])  # 12 numbers
        std_list.append(std_arr[lat_i, lon_i])    # 12 numbers

    df_out = df.copy()
    df_out[f"{var_name}_mean_{info}"] = mean_list
    df_out[f"{var_name}_std_{info}"] = std_list

    return df_out