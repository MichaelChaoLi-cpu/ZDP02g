import os
import pandas as pd

def data_load_combine_dataset(
    var_name : str = 'lai'
)-> pd.DataFrame:
    """
    Load, harmonize, and merge Global Flourishing Survey data with gridded
    environmental summary statistics for multiple survey waves.

    This function:
    1. Loads the Global Flourishing Survey dataset with geographic coordinates.
    2. Loads preprocessed environmental summary data (mean and standard deviation)
       for two survey years (2023 and 2024).
    3. Standardizes column names and assigns wave identifiers.
    4. Concatenates yearly environmental data into a single long-format table.
    5. Merges the environmental data with the survey dataset by latitude,
       longitude, and wave.
    6. Drops non-essential identifier columns and rows with missing values.

    Parameters
    ----------
    var_name : str, default='lai'
        Name of the environmental variable to merge (e.g., 'lai', 'tas').
        The function expects columns named:
        - '{var_name}_mean_2023', '{var_name}_std_2023'
        - '{var_name}_mean_2024', '{var_name}_std_2024'
        in the corresponding yearly parquet files.

    Returns
    -------
    pd.DataFrame
        A cleaned and merged DataFrame containing:
        - Geographic coordinates (LATITUDE, LONGITUDE)
        - Survey wave identifier (WAVE)
        - Environmental variable summary statistics
          ('{VAR_NAME}_MEAN', '{VAR_NAME}_STD')
        - Global Flourishing Survey outcome and covariate variables

        Rows with missing values are removed.

    Notes
    -----
    - Survey waves are encoded as:
        * 2023 → WAVE = 1
        * 2024 → WAVE = 2
    - Environmental variable names in the output are converted to uppercase
      for consistency.
    - This function assumes spatial alignment between survey locations and
      gridded environmental data.

    Examples
    --------
    >>> df = data_load_combine_dataset(var_name='lai')
    >>> df.columns
    Index([... 'LAI_MEAN', 'LAI_STD', ...], dtype='object')
    """
    gfs_dataset = pd.read_parquet(os.path.join('data', 'processed', 'GlobalFlourishingDataWithLonLit.parquet'))
    df_out_2023 = pd.read_parquet('data/processed/df_out_2023.parquet')
    df_out_2024 = pd.read_parquet('data/processed/df_out_2024.parquet')
    df_out_2023['WAVE'] = 1.0
    df_out_2024['WAVE'] = 2.0
    df_out_2023 = df_out_2023[['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name}_mean_2023', f'{var_name}_std_2023']]
    df_out_2023.columns = ['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name.upper()}_MEAN', f'{var_name.upper()}_STD']
    df_out_2024 = df_out_2024[['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name}_mean_2024', f'{var_name}_std_2024']]
    df_out_2024.columns = ['LATITUDE', 'LONGITUDE', 'WAVE', f'{var_name.upper()}_MEAN', f'{var_name.upper()}_STD']
    df_temp_now = pd.concat([df_out_2023, df_out_2024], axis = 0).reset_index(drop=True)
    gfs_temp_dataset = gfs_dataset.merge(df_temp_now, on = ['LATITUDE', 'LONGITUDE', 'WAVE'])
    all_data = gfs_temp_dataset.drop(columns = ['ID']).dropna()

    return all_data

def return_always_input_variable_list() -> list:
    """
    Return the list of core input variables that are always included in the model.

    This function defines a fixed set of demographic, socioeconomic, health,
    geographic, and environmental variables that serve as baseline inputs
    across all model specifications. The list is intended to ensure
    consistency and comparability when fitting models with different
    additional covariates or alternative environmental exposures.

    Returns
    -------
    list of str
        A list of variable (column) names including:
        - Geographic identifiers (country, latitude, longitude)
        - Survey wave indicator
        - Demographic and household characteristics
        - Socioeconomic status indicators
        - Physical and self-reported health measures
        - Social connection indicators
        - Environmental exposure summary statistics
          (e.g., LAI mean and standard deviation)

    Notes
    -----
    - All variables returned by this function are expected to be present
      in the merged analysis dataset.
    - Environmental variables (e.g., LAI_MEAN, LAI_STD) can be updated
      or replaced if alternative exposures are used, but the structure
      of this list should remain stable.
    - This function is designed to centralize model input definitions
      and reduce the risk of inconsistent variable selection.

    Examples
    --------
    >>> vars_in = return_always_input_variable_list()
    >>> 'LAI_MEAN' in vars_in
    True
    """
    return [
        'INCOME_REAL', 'COUNTRY', 'WAVE', 'LATITUDE', 'LONGITUDE', 'AGE',
        'EMPLOYMENT', 'MARITAL_STATUS', 'HAVE_CHILD', 'NUM_HOUSEHOLD_Y1',
        'OWN_RENT_HOME_Y1', 'URBAN_RURAL', 'EXPENSES', 'GENDER', 'EDUCATION_3',
        'CLOSE_TO', 
        'TAS_MEAN', 'TAS_STD'
    ]
    
def return_aim_mediate() -> list:
    return ['BODILY_PAIN',  'HEALTH_PROB', 'DAYS_EXERCISE']

def return_readable_variable_name():
    VARIABLE_MAP_RENAMED = {
        # Core Flourishing/Health Ratings (Scale 0-10)
        'MENTAL_HEALTH': 'Overall Mental Health (0-10)', 
        'PHYSICAL_HLTH': 'Overall Physical HEalth (0 - 10)',
        
        # Contextual/System Variables
        'COUNTRY': 'Country ID',                                # Country of Respondent [cite: 197, 198]
        'WAVE': 'Survey Wave',                            # Year 1, Year 2, etc. [cite: 260]
        'LATITUDE': 'Latitude',
        'LONGITUDE': 'Longitude',
        
        # Climate Data (As requested)
        'TAS_MEAN': 'Annual Mean Temperature',
        'TAS_STD': 'Annual Temperature Standard Deviation',
    
        # Demographic/Background Variables
        'AGE': 'Age of Respondent',                             # [cite: 262]
        'GENDER': 'Gender of Respondent',                       # [cite: 264, 42]
        'EDUCATION_3': 'Education Level (3-Categories)',        # Highest Completed Education [cite: 262]
        'MARITAL_STATUS': 'Marital Status',                     # [cite: 263]
        'URBAN_RURAL': 'Residential Area Type',                 # Urban/Rural [cite: 266]
        
        # Socioeconomic Variables
        'EMPLOYMENT': 'Employment Status',                       # [cite: 262]
        'INCOME_REAL': 'Household Income (USD)',              # Feelings about Household Income [cite: 263]
        'EXPENSES': 'Worry about Monthly Expenses (0-10)',       # 0=Worry all time, 10=Never worry [cite: 20, 11]
        'OWN_RENT_HOME_Y1': 'Housing Status (Own/Rent)',         # [cite: 266]
        'NUM_HOUSEHOLD_Y1': 'Number of Adults (18+) in Household', # [cite: 266]
        'HAVE_CHILD': 'Has Child Under 18 in Household',           # Assumed derived from 'NUM CHILDREN' [cite: 266]
        'DAYS_EXERCISE': 'Do Exercise at least Once a Week',
    
        # Health & Social Connection Variables
        'BODILY_PAIN': 'Bodily Pain in Past 4 Weeks',           # [cite: 13]
        'HEALTH_GROWUP_Y1': 'Health Status When Growing Up',     # [cite: 39]
        'HEALTH_PROB': 'Health Problems Restricting Activity',  # Prevents doing things people your age normally do [cite: 56]
        'CLOSE_TO': 'Knows Special Person Felt Close To',      # [cite: 15]  
    }
    
    return VARIABLE_MAP_RENAMED

def return_country_name_dict():
    COUNTRY_NAMES = {
        1: 'Argentina', 2:'Australia', 3:'Brazil', 
        4:'Egypt', 5:'Germany', 6:'India', 
        7:'Indonesia', 8:'Israel', 9:'Japan', 
        10:'Kenya', 11:'Mexico', 12:'Nigeria', 
        13:'the Philippines', 14:'Poland', 16:'South Africa', 
        17:'Spain', 18:'Tanzania', 19:'Turkey', 
        20:'the United Kingdom', 22:'the United States', 23:'Sweden', 
        24:'Hong Kong',
        15:'Others', 21: 'Others', 25: 'Others'}
    return COUNTRY_NAMES

def return_aim_variable_ph() -> str:
    return 'PHYSICAL_HLTH'
