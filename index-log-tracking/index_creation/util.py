import os
import re
import pandas as pd

def set_env_vars(ENV_VARS=None):
    """
    Set environment variables for Azure/OpenAI config.
    If a variable is set in ENV_VARS and is not a placeholder, use it as override.
    Otherwise, use the value from .env (already loaded via load_dotenv()).
    """
    if ENV_VARS is None:
        ENV_VARS = {}
    for k in ENV_VARS:
        value = ENV_VARS[k]
        # Use ENV_VARS value if not a placeholder, else keep .env value
        if value and not value.startswith('<') and not value.endswith('>'):
            os.environ[k] = value


def clean_metadata(df, schema_mapping_dict, index_name):
    """
    Cleans and standardizes metadata DataFrame columns using schema_mapping_dict.
    - Renames columns to standard schema.
    - Removes hash numbers from string columns.
    - Fills missing values with empty string.
    """
    # Rename columns based on schema mapping
    for key_std, key_df in schema_mapping_dict[index_name].items():
        if not key_df:
            df[key_std] = ""  # Add empty column if key_std is missing
            continue
        elif key_df not in df.columns:
            print("Warning: Column '{}' not found in DataFrame for index '{}'. Skipping renaming.".format(key_df, index_name))
            continue
        elif key_std in df.columns:
            continue
        df = df.rename(columns={key_df: key_std})

    # Remove hash numbers
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(remove_hash_number)
    df = df.fillna("")

    # Merge duplicate file names by concatenating differing values with ';'
    if 'Name' in df.columns:
        group_cols = [col for col in df.columns if col != 'Name']

        def merge_rows(group):
            merged = {}
            merged['Name'] = group.name
            for col in group_cols:
                merged[col] = '; '.join(sorted(set(val for val in group[col] if val)))
            return pd.Series(merged)

        df = df.groupby('Name').apply(merge_rows).reset_index(drop=True)
    return df


def remove_hash_number(text):
    if isinstance(text, str):
        try:
            return re.sub(r"#\d+", "", text)
        except:      
            return text