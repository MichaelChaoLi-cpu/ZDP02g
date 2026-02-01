# %% [markdown]
# # Data Preprocess

# %%

# %% [markdown]
# ## Import

# %%
import os, sys
sys.path.append(os.path.abspath("."))

# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# %%

# %% [markdown]
# ## Functions

# %%
def pivot_raw_data_longer(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transform a wide raw dataset into a longer format by stacking Y1 and Y2 versions
    of variables that appear in both groups.

    Steps:
    1. Identify all columns ending with "_Y1" and "_Y2".
    2. Extract base variable names (remove the suffix).
    3. Find common variables that exist in both Y1 and Y2 groups (preserving order).
    4. Keep only:
        - Variables without Y1/Y2 suffix
        - Matched Y1 variables
        - Matched Y2 variables
    5. Create two DataFrames:
        - One containing Y1 variables (renamed to base names)
        - One containing Y2 variables (renamed to base names)
    6. Stack (concat) the Y1 and Y2 rows to create a long-format dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame in wide format.

    Returns
    -------
    pd.DataFrame
        A longer-format DataFrame where Y1 and Y2 observations are stacked vertically
        with harmonized column names.
    """

    # Identify columns ending with _Y1 and remove suffix
    Y1_list = [item for item in df.columns.to_list() if "_Y1" in item]
    Y1_removey1_list = [item[:-3] for item in Y1_list]

    # Identify columns ending with _Y2 and remove suffix
    Y2_list = [item for item in df.columns.to_list() if "_Y2" in item]
    Y2_removey2_list = [item[:-3] for item in Y2_list]

    # Find variables that exist in both Y1 and Y2 groups (order preserved)
    common_elements = [x for x in Y1_removey1_list if x in Y2_removey2_list]

    # Columns that are not Y1 or Y2 variables
    NoY_list = [item for item in df.columns.to_list()
                if "_Y2" not in item and "_Y1" not in item]
    Keep_Y1_list = ['NUM_HOUSEHOLD_Y1', 'OWN_RENT_HOME_Y1', 'HEALTH_GROWUP_Y1']

    # Build lists of columns to keep for Y1 and Y2 subsets
    keep_y1_variable = NoY_list + [item + "_Y1" for item in common_elements] + Keep_Y1_list
    keep_y2_variable = NoY_list + [item + "_Y2" for item in common_elements] + Keep_Y1_list

    # Create Y1 dataset and rename columns to base names
    df_y1 = df[keep_y1_variable]
    df_y1.columns = NoY_list + common_elements + Keep_Y1_list

    # Create Y2 dataset and rename columns to base names
    df_y2 = df[keep_y2_variable]
    df_y2.columns = NoY_list + common_elements + Keep_Y1_list

    # Stack Y1 and Y2 observations
    df_long = pd.concat([df_y1, df_y2], axis=0).reset_index(drop=True)

    return df_long


# %% [markdown]
# ### Standard

# %%
income_mapping = {
    # Argentina (ARS -> USD, avg rate: 605.4765)
    101: 19.83,
    102: 59.48,
    103: 148.71,
    104: 297.29,
    105: 495.48,
    106: 693.67,
    107: 891.87,
    108: 1090.06,
    109: 1288.25,
    110: 1486.45,
    111: 1684.64,
    112: 1882.83,
    113: 2180.11,
    114: 2675.59,
    115: 3468.32,
    116: 3963.8,

    # Australia (AUD -> USD, avg rate: 1.5103)
    201: 13242.56,
    202: 16553.6,
    203: 23174.92,
    204: 29796.24,
    205: 36417.56,
    206: 44694.28,
    207: 57936.87,
    208: 82766.79,
    209: 132426.56,
    210: 165532.76,

    # Brazil (BRL -> USD, avg rate: 5.1917)
    301: 231.17,
    302: 694.52,
    303: 1445.72,
    304: 2023.69,
    305: 2890.41,
    306: 4046.21,
    307: 5201.81,
    308: 6357.42,
    309: 8091.02,
    310: 10402.47,
    311: 12713.91,
    312: 16180.93,
    313: 20803.78,
    314: 28893.7,
    315: 34671.04,

    # Egypt (EGP -> USD, avg rate: 37.9628)
    401: 63.22,
    402: 110.79,
    403: 237.25,
    404: 632.38,
    405: 1264.51,
    406: 1896.64,
    407: 2687.09,
    408: 4741.73,
    409: 6321.96,

    # Germany (EUR -> USD, avg rate: 0.9244)
    501: 6490.95,
    502: 12988.4,
    503: 22724.73,
    504: 29215.77,
    505: 35706.72,
    506: 42197.67,
    507: 48688.62,
    508: 58425.04,
    509: 71406.82,
    510: 87634.24,
    511: 113597.97,
    512: 129818.89,

    # India (INR -> USD, avg rate: 83.1343)
    601: 144.35,
    602: 216.56,
    603: 360.92,
    604: 505.29,
    605: 649.65,
    606: 793.98,
    607: 974.41,
    608: 1263.15,
    609: 1804.46,
    610: 2526.13,
    611: 3969.61,
    612: 6134.72,
    613: 7217.31,

    # Indonesia (IDR -> USD, avg rate: 15546.1665)
    701: 115.77,
    702: 173.65,
    703: 347.37,
    704: 617.53,
    705: 964.89,
    706: 1350.85,
    707: 1736.79,
    708: 2508.62,
    709: 4631.34,
    710: 8490.87,
    711: 10806.44,

    # Israel (ILS -> USD, avg rate: 3.6835)
    801: 3257.79,
    802: 6517.05,
    803: 13032.57,
    804: 17919.12,
    805: 21176.84,
    806: 24434.57,
    807: 27692.29,
    808: 30950.01,
    809: 34207.74,
    810: 37465.52,
    811: 43981.02,
    812: 50496.52,
    813: 55382.96,
    814: 61898.51,
    815: 97733.54,
    816: 130309.24,

    # Japan (JPY -> USD, avg rate: 145.9287)
    901: 4111.54,
    902: 6167.37,
    903: 10279.05,
    904: 14390.68,
    905: 18502.26,
    906: 22613.84,
    907: 28781.29,
    908: 34948.62,
    909: 39060.2,
    910: 45227.66,
    911: 53450.8,
    912: 61673.94,
    913: 69897.19,
    914: 78120.33,
    915: 82231.91,

    # Kenya (KES -> USD, avg rate: 137.3344)
    1001: 43.68,
    1002: 67.79,
    1003: 131.2,
    1004: 218.54,
    1005: 305.88,
    1006: 393.21,
    1007: 480.55,
    1008: 611.66,
    1009: 786.53,
    1010: 961.25,
    1011: 1310.73,
    1012: 1878.75,
    1013: 2839.84,
    1014: 5242.66,
    1015: 6990.22,

    # Mexico (MXN -> USD, avg rate: 18.0318)
    1101: 366.02,
    1102: 516.14,
    1103: 832.23,
    1104: 1164.99,
    1105: 1497.75,
    1106: 1830.51,
    1107: 2329.53,
    1108: 2995.07,
    1109: 3993.25,
    1110: 5657.0,
    1111: 8318.99,
    1112: 11646.5,
    1113: 16637.66,
    1114: 23292.53,
    1115: 29947.4,
    1116: 39929.79,
    1117: 46584.34,

    # Nigeria (NGN -> USD, avg rate: 1062.0796)
    1201: 41.81,
    1202: 54.82,
    1203: 84.76,
    1204: 124.32,
    1205: 175.14,
    1206: 237.29,
    1207: 305.04,
    1208: 378.53,
    1209: 474.66,
    1210: 587.55,
    1211: 700.54,
    1212: 813.52,
    1213: 999.93,
    1214: 1412.29,
    1215: 1694.71,

    # Philippines (PHP -> USD, avg rate: 56.4605)
    1301: 148.76,
    1302: 233.9,
    1303: 478.37,
    1304: 743.96,
    1305: 1062.88,
    1306: 1487.83,
    1307: 1912.87,
    1308: 2656.84,
    1309: 3719.47,
    1310: 4782.21,
    1311: 5844.85,
    1312: 7438.88,
    1313: 9564.38,
    1314: 14877.75,
    1315: 19128.32,

    # Poland (PLN -> USD, avg rate: 4.0922)
    1401: 293.32,
    1402: 881.18,
    1403: 2200.82,
    1404: 3666.99,
    1405: 5133.17,
    1406: 6599.34,
    1407: 8065.52,
    1408: 9531.69,
    1409: 10997.87,
    1410: 13197.19,
    1411: 16129.61,
    1412: 20528.21,
    1413: 26392.97,
    1414: 36656.36,
    1415: 51318.26,
    1416: 58647.86,

    # South Africa (ZAR -> USD, avg rate: 18.3895)
    1601: 130.51,
    1602: 179.78,
    1603: 440.81,
    1604: 979.15,
    1605: 1631.64,
    1606: 2284.14,
    1607: 2936.75,
    1608: 3589.35,
    1609: 4568.13,
    1610: 5873.32,
    1611: 8157.16,
    1612: 11419.95,
    1613: 16314.01,
    1614: 19576.4,

    # Spain (EUR -> USD, avg rate: 0.9244)
    1701: 3245.47,
    1702: 8120.16,
    1703: 14611.12,
    1704: 17856.56,
    1705: 21102.09,
    1706: 24347.53,
    1707: 27593.06,
    1708: 30838.5,
    1709: 34083.94,
    1710: 37329.47,
    1711: 42197.67,
    1712: 48688.62,
    1713: 58425.04,
    1714: 71406.82,
    1715: 77891.36,

    # Tanzania (TZS -> USD, avg rate: 2490.4715)
    1801: 96.35,
    1802: 168.62,
    1803: 361.34,
    1804: 602.28,
    1805: 843.19,
    1806: 1084.11,
    1807: 1325.02,
    1808: 1686.44,
    1809: 2168.33,
    1810: 2891.06,
    1811: 3372.81,

    # Turkey (TRY -> USD, avg rate: 28.2722)
    1901: 42.45,
    1902: 127.59,
    1903: 318.57,
    1904: 530.72,
    1905: 743.0,
    1906: 955.26,
    1907: 1167.43,
    1908: 1379.68,
    1909: 1591.85,
    1910: 1804.11,
    1911: 2016.31,
    1912: 2334.64,
    1913: 2971.3,
    1914: 3820.22,
    1915: 4669.04,
    1916: 5093.45,

    # United Kingdom (GBP -> USD, avg rate: 0.7935)
    2001: 7561.64,
    2002: 11350.12,
    2003: 18911.72,
    2004: 26473.33,
    2005: 34034.94,
    2006: 41596.64,
    2007: 52939.22,
    2008: 68062.5,
    2009: 83185.77,
    2010: 105870.76,
    2011: 120986.54,

    # United States (USD)
    2201: 12000.0,
    2202: 18000.0,
    2203: 30000.0,
    2204: 42000.0,
    2205: 54000.0,
    2206: 75000.0,
    2207: 105000.0,
    2208: 150000.0,
    2209: 210000.0,
    2210: 240000.0,

    # Sweden (SEK -> USD, avg rate: 10.5889)
    2301: 11332.72,
    2302: 14166.32,
    2303: 22665.81,
    2304: 31165.3,
    2305: 36831.55,
    2306: 42497.91,
    2307: 50997.36,
    2308: 62330.04,
    2309: 73662.61,
    2310: 84995.29,
    2311: 101994.16,
    2312: 113326.23,

    # Hong Kong (HKD -> USD, avg rate: 7.8167)
    2401: 7675.88,
    2402: 11514.56,
    2403: 19190.37,
    2404: 26866.29,
    2405: 38380.12,
    2406: 53731.83,
    2407: 65245.63,
    2408: 72921.44,
    2409: 84435.28,
    2410: 99787.01,
    2411: 115138.74,
    2412: 138166.33,
    2413: 153517.28,
}

# Exchange rates used (World Bank 2023-2024 average, LCU per USD):
exchange_rates = {
    'Argentina': 605.4765,
    'Australia': 1.5103,
    'Brazil': 5.1917,
    'Egypt': 37.9628,
    'Germany': 0.9244,
    'Hong Kong': 7.8167,
    'India': 83.1343,
    'Indonesia': 15546.1665,
    'Israel': 3.6835,
    'Japan': 145.9287,
    'Kenya': 137.3344,
    'Mexico': 18.0318,
    'Nigeria': 1062.0796,
    'Philippines': 56.4605,
    'Poland': 4.0922,
    'South Africa': 18.3895,
    'Spain': 0.9244,
    'Sweden': 10.5889,
    'Tanzania': 2490.4715,
    'Turkey': 28.2722,
    'United Kingdom': 0.7935,
    'United States': 1.0000,
}


# %%
def process_gfs_data(df):
    df['INCOME'] = df['INCOME'].replace({9900: pd.NA, 9996: pd.NA, 9997: pd.NA, 9998: pd.NA, 9999: pd.NA})

    df['INCOME'] = df['INCOME'].map(income_mapping)

    return df


# %%

# %%

# %% [markdown]
# ## Runs 

# %% [markdown]
# ### Basic Flow

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
df_raw = pd.read_csv(
    'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv',
    na_values=['', ' ', 'NA', 'N/A'],
    low_memory=False
)

# %%
df = df_raw.copy()

# %%
for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except:
        print(col)

# %%
columns_to_process = df.columns.difference(['AGE_Y1', 'AGE_Y2', 'LONGITUDE_Y1', 'LONGITUDE_Y2'])
mask = df[columns_to_process].isin([98., 99., -98., -99.])
df.loc[:, columns_to_process] = df[columns_to_process].mask(mask, np.nan)

# %%

# %%

# %% [markdown]
# ### Go

# %%
variable_list = []

# %%
df_long = pivot_raw_data_longer(
    df = df
)

# %%

# %% [markdown]
# ### Income

# %%
df_long['INCOME'] = df_long['INCOME'].replace({9900: pd.NA, 9996: pd.NA, 9997: pd.NA, 9998: pd.NA, 9999: pd.NA})

# %%
df_long = df_long.dropna(subset = ['INCOME'])

# %%
df_long['INCOME_REAL'] = df_long['INCOME'].astype(int).map(income_mapping)

# %%
variable_list.append('INCOME_REAL')

# %%

# %% [markdown]
# ### ID and Country

# %%
df_long.columns

# %%
variable_list.append('ID')

# %%
variable_list.append('COUNTRY')

# %%

# %% [markdown]
# ### Wave  

# %%
variable = 'WAVE'

# %%
df_long[variable].isna().any()

# %%
df_long[variable].value_counts(dropna=False).sort_index()

# %%
df_long.shape

# %%
df_long = df_long.dropna(subset = variable)

# %%
df_long.shape

# %%
variable_list.append(variable)

# %%

# %% [markdown]
# ### LONGITUDE  LATITUDE

# %%
df_long.shape

# %%
df_long = df_long.dropna(subset=['LONGITUDE', 'LATITUDE'])

# %%
df_long.shape

# %%
variable_list.append('LATITUDE')
variable_list.append('LONGITUDE')

# %%

# %% [markdown]
# ## Demographic Variables

# %% [markdown]
# ### Age

# %%
df_long['AGE'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long[df_long['AGE'] < 998]
df_long = df_long.dropna(subset = 'AGE')

# %%
df_long.shape

# %%
variable_list.append('AGE')

# %%

# %% [markdown]
# ### EMPLOYMENT

# %%
df_long['EMPLOYMENT'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long.dropna(subset = 'EMPLOYMENT')

# %%
df_long.shape

# %%
variable_list.append('EMPLOYMENT')

# %%

# %% [markdown]
# ### MARITAL_STATUS

# %%
df_long['MARITAL_STATUS'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long.dropna(subset='MARITAL_STATUS')

# %%
df_long.shape

# %%
variable_list.append('MARITAL_STATUS')

# %%

# %% [markdown]
# ###  NUM_CHILDREN

# %%
df_long = df_long.dropna(subset='NUM_CHILDREN')   
df_long['HAVE_CHILD'] = (df_long['NUM_CHILDREN'] > 0).astype(float)

# %%
df_long.shape

# %%
variable_list.append('HAVE_CHILD')

# %%

# %% [markdown]
# ### NUM_adult

# %%
df_long['NUM_HOUSEHOLD_Y1'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long[df_long['NUM_HOUSEHOLD_Y1'] < 15]
df_long = df_long.dropna(subset='NUM_HOUSEHOLD_Y1')   

# %%
df_long.shape

# %%
variable_list.append('NUM_HOUSEHOLD_Y1')

# %%

# %% [markdown]
# ### OWN_RENT_HOME

# %%
df_long['OWN_RENT_HOME_Y1'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long.dropna(subset = 'OWN_RENT_HOME_Y1')

# %%
df_long.shape

# %%
variable_list.append('OWN_RENT_HOME_Y1')

# %%

# %% [markdown]
# ### URBAN_RURAL

# %%
df_long['URBAN_RURAL'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long.dropna(subset = 'URBAN_RURAL')

# %%
df_long.shape

# %%
variable_list.append('URBAN_RURAL')

# %%

# %% [markdown]
# ### EXPENSES

# %%
df_long['EXPENSES'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long.dropna(subset = 'EXPENSES')

# %%
df_long.shape

# %%
variable_list.append('EXPENSES')

# %%

# %% [markdown]
# ### Gender

# %%
df_long['GENDER'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long[df_long['GENDER'] < 4]
df_long = df_long.dropna(subset = 'GENDER')

# %%
df_long.shape

# %%
variable_list.append('GENDER')

# %%

# %% [markdown]
# ### EDUCATION

# %%
df_long['EDUCATION_3'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long.dropna(subset = 'EDUCATION_3')

# %%
df_long.shape

# %%
variable_list.append('EDUCATION_3')

# %%

# %% [markdown]
# ### CLOSE_TO

# %%
variable = 'CLOSE_TO'

# %%
df_long[variable].value_counts(dropna=False).sort_index()

# %%
df_long[variable] = df_long[variable].replace({2: 0})
df_long = df_long.dropna(subset = variable)

# %%
df_long.shape

# %%
variable_list.append(variable)

# %%

# %% [markdown]
# ## Main Survey Questions

# %% [markdown]
# ### BODILY_PAIN

# %%
df_long['BODILY_PAIN'].value_counts(dropna=False).sort_index()

# %%
df_long = df_long[df_long['BODILY_PAIN'] < 5]
df_long = df_long.dropna(subset = 'BODILY_PAIN')

# %%
df_long['BODILY_PAIN'] = np.where(df_long['BODILY_PAIN'] > 2, 0, 1)

# %%
df_long.shape

# %%
variable_list.append('BODILY_PAIN')

# %%

# %% [markdown]
# ### HEALTH_PROB

# %%
variable = 'HEALTH_PROB'

# %%
df_long[variable].value_counts(dropna=False).sort_index()

# %%
df_long[variable] = df_long[variable].replace({2: 0})
df_long = df_long.dropna(subset = variable)

# %%
df_long.shape

# %%
variable_list.append(variable)

# %%

# %% [markdown]
# ### DAYS_EXERCISE

# %%
variable = 'DAYS_EXERCISE'

# %%
df_long[variable].value_counts(dropna=False).sort_index()

# %%
df_long = df_long[df_long[variable] < 8]
df_long = df_long.dropna(subset = variable)

# %%
df_long[variable] = np.where(df_long[variable] > 0, 1, 0)

# %%
variable_list.append(variable)

# %%

# %% [markdown]
# ## Output

# %%
variable = 'PHYSICAL_HLTH'

# %%
df_long[variable].value_counts(dropna=False).sort_index()

# %%
df_long = df_long[df_long[variable] < 11]
df_long = df_long.dropna(subset = variable)

# %%
variable_list.append(variable)

# %%

# %%

# %% [markdown]
# ## Save the Washed Dataset

# %%
variable_list

# %%
WahsedDataset = df_long[variable_list]

# %%
WahsedDataset.shape

# %%
WahsedDataset.to_parquet(os.path.join('data', 'processed', 'GlobalFlourishingDataWithLonLit.parquet'))

# %%
