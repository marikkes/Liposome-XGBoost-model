import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np

def load_lipid_properties(db_path: Path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM lipid_properties", conn)
    conn.close()

    return df

def get_lipid_profile(lipid_name, lipid_df):
    row = lipid_df[lipid_df["lipid_name"] == lipid_name].iloc[0]

    profile = {}
    for col in lipid_df.columns:
        if col != "lipid_name":
            profile[f"lipid_{col}"] = row[col]
            profile[f"lipid_{col}_missing"] = int(pd.isna(row[col]))

    return profile

def get_lipid_profiles(lipid_names, lipid_df):
    return [get_lipid_profile(name, lipid_df) for name in lipid_names]

# Not in use anymore, decided to use the lipid profiles separately instead of aggregating them into a single profile
def aggregate_lipid_profiles(lipid_profiles, fractions):
    df = pd.DataFrame(lipid_profiles)
    
    fractions = np.array(fractions)
    fractions = fractions / fractions.sum() # Fractions are supposed to sum up to 1

    agg = {}

    for col in df.columns:
        values = df[col].values

        # weighted mean
        mean = np.sum(values * fractions)
        agg[col] = mean

        # weighted std
        variance = np.sum(fractions * (values - mean) ** 2)
        agg[f"{col}_std"] = np.sqrt(variance)

    return agg

def build_lipid_slots(lipid_names, lipid_df, max_lipids=3):
    slots = {}

    lipid_names = lipid_names[:max_lipids]
    lipid_names = lipid_names + [None] * (max_lipids - len(lipid_names))

    for i, lipid in enumerate(lipid_names, start=1):
        prefix = f"lipid_{i}_"

        # Missing lipid slot
        if lipid is None or pd.isna(lipid):
            if i > 1:
                slots[f"lipid_{i}_missing"] = 1

            for col in lipid_df.columns:
                if col != "lipid_name":
                    slots[prefix + col] = np.nan
            continue

        # Try to find lipid
        match = lipid_df[lipid_df["lipid_name"] == lipid]

        if match.empty:
            print(f"⚠️ Lipid not found in DB: {lipid}")  # DEBUG

            if i > 1:
                slots[f"lipid_{i}_missing"] = 1

            for col in lipid_df.columns:
                if col != "lipid_name":
                    slots[prefix + col] = np.nan
            continue

        # Lipid exists
        if i > 1:
            slots[f"lipid_{i}_missing"] = 0

        row = match.iloc[0]

        for col in lipid_df.columns:
            if col != "lipid_name":
                slots[prefix + col] = row[col]

    return slots