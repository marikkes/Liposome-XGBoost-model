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