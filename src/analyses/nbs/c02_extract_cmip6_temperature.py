# Python source converted from src/analyses/notebooks/c02_extract_cmip6_temperature.ipynb
# Code-cell order and Markdown explanations are preserved.

# Resolve project paths before importing the analysis helper modules.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
_sys.path.insert(0, str(_PROJECT_ROOT / "src/analyses/notebooks"))
_os.chdir(_PROJECT_ROOT)


# %% [markdown]
# # Link to CMIP6 Simulation

# %%


# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import numpy as np
import pandas as pd

from netCDF4 import Dataset

# %%
import TestingTools

# %%


# %% [markdown]
# ## Functions

# %%
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

# %%
def obtain_addresses_nc(
    years: list = [2023, 2024, 2050, 2070, 2100],
    ssps: list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585'],
    base_root: str = 'data/raw/cmip6/'
) -> list:
    """
    Generate a structured list of CMIP6 NetCDF file paths for
    EC-Earth3-Veg ScenarioMIP 'tas' monthly temperature outputs.

    This function produces a nested list where the outer dimension corresponds
    to SSP scenarios, and the inner dimension corresponds to years. For each
    SSP–year combination, the function constructs the expected NetCDF filename:

        tas_Amon_EC-Earth3-Veg_{ssp}_r1i1p1f1_gr_{year}01-{year}12.nc

    The final output has the structure:

        [
            [file_ssp119_2023, file_ssp119_2024, ...],
            [file_ssp126_2023, file_ssp126_2024, ...],
            ...
        ]

    Parameters
    ----------
    years : list of int, default [2023, 2024, 2050, 2070, 2100]
        The list of years for which file paths will be generated.

    ssps : list of str, default ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        ScenarioMIP SSP experiment identifiers. One list of files is created for
        each SSP scenario.

    base_root : str, default 'data/raw/cmip6/'
        Root directory where NetCDF files are stored.

    Returns
    -------
    list of list of str
        A nested list where each sub-list contains file paths for one SSP scenario
        across all specified years.

    Notes
    -----
    - No file existence check is performed; only file paths are constructed.
    - Assumes EC-Earth3-Veg, realization r1i1p1f1, and grid label 'gr'.
    """

    ssp_year_list = []
    for ssp in ssps:
        year_list = []
        for year in years:
            addr = f'{base_root}tas_Amon_EC-Earth3-Veg_{ssp}_r1i1p1f1_gr_{year}01-{year}12.nc'
            year_list.append(addr)
        ssp_year_list.append(year_list)

    return ssp_year_list

# %%
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

# %%
def make_array_based_addr_list(
    ssp_year_list: list,
    varname: str = "tas"
) -> np.ndarray:
    """
    Convert a nested list of NetCDF file paths (grouped by SSP and year)
    into a nested NumPy array structure containing climate data arrays.

    This function expects input in the form:
        [
            [addr_ssp1_year1, addr_ssp1_year2, ...],
            [addr_ssp2_year1, addr_ssp2_year2, ...],
            ...
        ]
    For each NetCDF file path, the function loads the 'tas' variable using
    `nc_to_numpy_nc4` and stores the resulting NumPy array. The output is a
    nested NumPy array (dtype=object) with the same SSP–year structure.

    Parameters
    ----------
    ssp_year_list : list of list of str
        Nested list where each sub-list contains NetCDF file paths for one
        SSP scenario across multiple years.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of dtype object, where:
        - axis 0 indexes SSP scenarios
        - axis 1 indexes years within each SSP scenario
        - each entry is a NumPy ndarray loaded from a NetCDF file

    Notes
    -----
    - Assumes all NetCDF files contain variable 'tas'.
    - Uses `nc_to_numpy_nc4` to load each dataset.
    - Output uses dtype=object because climate arrays typically differ in
      shape (e.g., varying time dimension between years).
    """

    ssp_year_arrs = []
    for year_list in ssp_year_list:
        year_arrs = []
        for year_addr in year_list:
            arr = nc_to_numpy_nc4(
                nc_path=year_addr,
                varname=varname
            )
            year_arrs.append(arr)
        ssp_year_arrs.append(year_arrs)

    # Convert nested list → nested NumPy array
    ssp_year_arrs = np.array(ssp_year_arrs)

    return ssp_year_arrs

# %%
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

# %%
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

# %%
def covert_K_to_C(
    arr: np.ndarray
) -> np.ndarray:
    arr_new = arr - 273.15
    return arr_new

# %%
def extract_values_single_scenario(
    df : pd.DataFrame, 
    mean_arr : np.ndarray, 
    std_arr : np.ndarray, 
    info : str
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
    df_out[f"tas_mean_{info}"] = mean_list
    df_out[f"tas_std_{info}"] = std_list

    return df_out

# %%
def extract_values_five_scenario(
    df : pd.DataFrame, 
    mean_arr : np.ndarray, 
    std_arr : np.ndarray, 
    info : str
) -> pd.DataFrame:
    """
    Extract climate variable statistics (mean and standard deviation) for all
    five SSP scenarios at each geographic location defined in the input DataFrame.

    For each row in `df`, the function uses the corresponding grid indices
    (lat_idx, lon_idx) to retrieve values from the provided climate data arrays.
    The arrays `mean_arr` and `std_arr` are assumed to contain values for all
    five SSP scenarios, stacked along the first dimension.

    Expected array shapes
    ---------------------
    mean_arr : np.ndarray
        Shape (5, T, lat, lon), where:
        - 5 corresponds to the five SSP scenarios:
          ssp119, ssp126, ssp245, ssp370, ssp585
        - T is the number of time steps (e.g., 12 months)
        - lat, lon correspond to the spatial grid

    std_arr : np.ndarray
        Same shape as mean_arr, representing standard deviations.

    df : pd.DataFrame
        Must contain the columns:
            - 'lat_idx' : row index into the spatial grid
            - 'lon_idx' : column index into the spatial grid

    Behavior
    --------
    For each location (lat_idx, lon_idx), the function extracts:

        mean_arr[:, lat_idx, lon_idx]  → 5 × T values  
        std_arr[:, lat_idx, lon_idx]   → 5 × T values  

    The resulting lists are assigned to new scenario-specific columns whose names
    incorporate the string provided in `info`.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with five additional mean-value columns and
        five additional std-value columns, each corresponding to one SSP scenario.

    Notes
    -----
    - Column assignment in this implementation uses tuple column names, which
      may not behave as intended in standard Pandas usage.
    - A corrected version with explicit individual columns is recommended.
    """
    mean_list = []
    std_list = []

    for lat_i, lon_i in zip(df["lat_idx"], df["lon_idx"]):
        mean_list.append(mean_arr[:, lat_i, lon_i])  # 12 numbers
        std_list.append(std_arr[:, lat_i, lon_i])    # 12 numbers

    df_out = df.copy()
    df_out[[f"tas_mean_ssp119_{info}", f"tas_mean_ssp126_{info}", 
        f"tas_mean_ssp245_{info}", f"tas_mean_ssp370_{info}", 
        f"tas_mean_ssp585_{info}"]] = mean_list
    df_out[[f"tas_std_ssp119_{info}", f"tas_std_ssp126_{info}", 
        f"tas_std_ssp245_{info}", f"tas_std_ssp370_{info}", 
        f"tas_std_ssp585_{info}"]] = std_list

    return df_out

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
df_location = obtian_unique_location(
    df_addr = os.path.join('data', 'processed', 'GlobalFlourishingDataWithLonLit.parquet')
)

# %%
ssp_year_list = obtain_addresses_nc(
    years = [2023, 2024, 2050, 2070, 2100],
    ssps = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585'],
    base_root = 'data/raw/cmip6/'
)

# %%
ssp_year_arrs = make_array_based_addr_list(
    ssp_year_list = ssp_year_list
)

# %%
ssp_year_arrs = covert_K_to_C(
    arr = ssp_year_arrs
)

# %%
tas_2023_arr = np.mean(ssp_year_arrs[:,0,:,:,:], axis = 0) 
tas_2023_monmean_arr = np.mean(tas_2023_arr, axis = 0) 
tas_2023_monstd_arr = np.std(tas_2023_arr, axis = 0)

# %%
tas_2024_arr = np.mean(ssp_year_arrs[:,1,:,:,:], axis = 0) 
tas_2024_monmean_arr = np.mean(tas_2024_arr, axis = 0) 
tas_2024_monstd_arr = np.std(tas_2024_arr, axis = 0)

# %%
tas_ssp_2050_arr = ssp_year_arrs[:,2,:,:,:] 
tas_ssp_2050_mean_arr = np.mean(tas_ssp_2050_arr, axis = 1) 
tas_ssp_2050_std_arr = np.std(tas_ssp_2050_arr, axis = 1)

# %%
tas_ssp_2070_arr = ssp_year_arrs[:,3,:,:,:]
tas_ssp_2070_mean_arr = np.mean(tas_ssp_2070_arr, axis = 1) 
tas_ssp_2070_std_arr = np.std(tas_ssp_2070_arr, axis = 1)

# %%
tas_ssp_2100_arr = ssp_year_arrs[:,4,:,:,:]
tas_ssp_2100_mean_arr = np.mean(tas_ssp_2100_arr, axis = 1) 
tas_ssp_2100_std_arr = np.std(tas_ssp_2100_arr, axis = 1)

# %%
lat_grid, lon_grid = generate_grid_index_from_array(
    one_month_array = ssp_year_arrs[0,0,0,:,:]
)

# %%
df_out = convert_df_to_grid_indices(
    df_location, 
    lat_grid, 
    lon_grid
)

# %%
df_out_2023 = extract_values_single_scenario(
    df = df_out, 
    mean_arr = tas_2023_monmean_arr, 
    std_arr = tas_2023_monstd_arr, 
    info = 2023
)

# %%
df_out_2024 = extract_values_single_scenario(
    df = df_out, 
    mean_arr = tas_2024_monmean_arr, 
    std_arr = tas_2024_monstd_arr, 
    info = 2024
)

# %%
df_out_2050 = extract_values_five_scenario(
    df = df_out, 
    mean_arr = tas_ssp_2050_mean_arr, 
    std_arr = tas_ssp_2050_std_arr, 
    info = 2050
)

# %%
df_out_2070 = extract_values_five_scenario(
    df = df_out, 
    mean_arr = tas_ssp_2070_mean_arr, 
    std_arr = tas_ssp_2070_std_arr, 
    info = 2070
)

# %%
df_out_2100 = extract_values_five_scenario(
    df = df_out, 
    mean_arr = tas_ssp_2100_mean_arr, 
    std_arr = tas_ssp_2100_std_arr, 
    info = 2100
)

# %%
df_out_2023.to_parquet('data/processed/df_out_2023.parquet')
df_out_2024.to_parquet('data/processed/df_out_2024.parquet')
df_out_2050.to_parquet('data/processed/df_out_2050.parquet')
df_out_2070.to_parquet('data/processed/df_out_2070.parquet')
df_out_2100.to_parquet('data/processed/df_out_2100.parquet')

# %%


# %%

