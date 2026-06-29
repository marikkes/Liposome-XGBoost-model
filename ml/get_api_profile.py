import sqlite3
import pandas as pd
from pathlib import Path

def load_api_properties(db_path: Path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM api_properties", conn)
    conn.close()

    return df

def get_api_profile(api_name, api_df):
    row = api_df[api_df["api"] == api_name].iloc[0]

    profile = {}
    for col in api_df.columns:
        if col != "api":
            profile[f"api_{col}"] = row[col]
            profile[f"api_{col}_missing"] = int(pd.isna(row[col]))

    return profile

def preprocess_api_profile(api_profile, X_reference):

    api_profile = api_profile.copy()

    for key in list(api_profile.keys()):

        if key.endswith("_missing"):
            continue

        if pd.isna(api_profile[key]):

            missing_key = key + "_missing"
            api_profile[missing_key] = 1

            if key in X_reference.columns:
                api_profile[key] = float(X_reference[key].median())
            else:
                raise KeyError(f"API profile key '{key}' not found in reference feature set")

    return api_profile