
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from feature_engineering import load_and_engineer_features
from get_api_profile import load_api_properties
from get_lipid_profile import load_lipid_properties, build_lipid_slots

def make_dataset(db_path: Path, api_db_path: Path, lipid_db_path: Path):

    # Hent feature-engineeret DataFrame
    df = load_and_engineer_features(db_path)

    fraction_cols = [
        c for c in df.columns if c.startswith("lipid_") 
        and c.endswith("_fraction")
        and not c.startswith("lipid_1_")
        and not c.startswith("lipid_2_")
        and not c.startswith("lipid_3_")
        ]

    # ---------- Load API properties ----------
    api_df = load_api_properties(api_db_path)

    # ---------- Load lipid properties ----------
    lipid_df = load_lipid_properties(lipid_db_path)

    # ---------- Merge ----------
    df = df.merge(api_df, on="api", how="left")

    # legg "api_" foran alle API-kolonner
    api_cols = [c for c in api_df.columns if c != "api"]

    df = df.rename(columns={col: f"api_{col}" for col in api_cols})

    # Debug
    missing = df["api_molecular_weight"].isna().sum()
    print(f"Missing API properties: {missing}")

    # Compute lipid features for each row in the dataset
    lipid_features = df.apply(
    lambda row: compute_lipid_features(row, lipid_df),
    axis=1
    )

    df = pd.concat([df, lipid_features], axis=1)
    print("DF:", df.columns.tolist())

    api_feature_cols = [
        "api_molecular_weight",
        "api_rotatable_bond_count",
        "api_hbond_acceptor_count",
        "api_hbond_donor_count",
        "api_heavy_atom_count",
        "api_tpsa",
        "api_complexity"
    ]

    # Lag missing flags
    for col in api_feature_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    # Imputer med median
    for col in api_feature_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    missing_cols = [f"{col}_missing" for col in api_feature_cols]
    slot_fraction_cols = [
    "lipid_1_fraction",
    "lipid_2_fraction",
    "lipid_3_fraction"
    ]
    lipid_feature_cols = list(lipid_features.columns)

    feature_cols = ([
        "n_lipids",
        #"total_lipid_qty", fjernet denne siden det viktige er api to lipid ratio
        #"api_qty", fjernet denne for at ikke mengden api skal "telle" to ganger i modellen
        "api_to_lipid_ratio",
    ] 
    + fraction_cols
    + api_feature_cols
    + missing_cols
    + slot_fraction_cols
    + lipid_feature_cols
    )

    X = df[feature_cols]#.dropna()
    y = df.loc[X.index, "ee_mean"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

def compute_lipid_features(row, lipid_df):
    lipid_names = [row["lipid_1"], row["lipid_2"], row["lipid_3"]]
    qtys = [row["lipid_1_qty"], row["lipid_2_qty"], row["lipid_3_qty"]]

    # Normalize fractions
    total = sum([q for q in qtys if pd.notna(q) and q > 0])

    fractions = [
        (q / total) if pd.notna(q) and total > 0 else 0
        for q in qtys
    ]

    if total == 0:
        return pd.Series()

    # ---------- Build slots ----------
    features = build_lipid_slots(lipid_names, lipid_df)
 
    # ---------- Add slot fractions ----------
    #for i in range(3):
    #    features[f"lipid_{i+1}_fraction"] = fractions[i]

    # ----------  Weighted features ----------
    weighted_logp = 0
    weighted_hbond = 0
    logps = []

    for i in range(3):
        frac = fractions[i]
        logp = features.get(f"lipid_{i+1}_logp")
        hbond = features.get(f"lipid_{i+1}_hbond_donor_count")

        if pd.notna(logp):
            weighted_logp += frac * logp
            logps.append(logp)

        if pd.notna(hbond):
            weighted_hbond += frac * hbond
    
    # ---------- Max / Min ----------
    if logps:
        features["lipid_max_logp"] = max(logps)
        features["lipid_min_logp"] = min(logps)
    else:
        features["lipid_max_logp"] = np.nan
        features["lipid_min_logp"] = np.nan

    features["lipid_weighted_logp"] = weighted_logp
    features["lipid_weighted_hbond_donors"] = weighted_hbond

    # ---------- Lipid-lipid interactions ----------
    l1_logp = features.get("lipid_1_logp")
    l2_logp = features.get("lipid_2_logp")
    l3_logp = features.get("lipid_3_logp")

    if pd.notna(l1_logp) and pd.notna(l2_logp):
        features["lipid_1_2_logp_diff"] = l1_logp - l2_logp
    else:
        features["lipid_1_2_logp_diff"] = np.nan

    if pd.notna(l1_logp) and pd.notna(l3_logp):
        features["lipid_1_3_logp_diff"] = l1_logp - l3_logp
    else:
        features["lipid_1_3_logp_diff"] = np.nan

    # Tail diff
    l1_tail = features.get("lipid_1_tail_length_mean")
    l2_tail = features.get("lipid_2_tail_length_mean")

    if pd.notna(l1_tail) and pd.notna(l2_tail):
        features["lipid_1_2_tail_diff"] = l1_tail - l2_tail
    else:
        features["lipid_1_2_tail_diff"] = np.nan

    # ---------- API interactions ----------
    api_logp = row.get("api_logp")
    api_hbond = row.get("api_hbond_donor_count")

    if pd.notna(api_logp):
        if pd.notna(l1_logp):
            features["api_lipid_1_logp_diff"] = api_logp - l1_logp
            features["api_lipid_1_logp_ratio"] = api_logp / (l1_logp + 1e-6)
        else:
            features["api_lipid_1_logp_diff"] = np.nan
            features["api_lipid_1_logp_ratio"] = np.nan

        if pd.notna(l2_logp):
            features["api_lipid_2_logp_diff"] = api_logp - l2_logp
        else:
            features["api_lipid_2_logp_diff"] = np.nan

    if pd.notna(api_hbond) and pd.notna(features.get("lipid_1_hbond_donor_count")):
        features["api_lipid_1_hbond_donor_diff"] = (
            api_hbond - features["lipid_1_hbond_donor_count"]
        )
    else:
        features["api_lipid_1_hbond_donor_diff"] = np.nan

    return pd.Series(features)

def print_lipid_examples(df, lipid_feature_cols):

    for n in [1, 2, 3]:
        subset = df[df["n_lipids"] == n]

        if len(subset) == 0:
            print(f"\nNo rows found with n_lipids = {n}")
            continue

        row = subset.iloc[0]

        print(f"\n==================== n_lipids = {n} ====================")
        print(row[lipid_feature_cols].to_string())

if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"
    API_DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"
    LIPID_DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"

    X, y = make_dataset(DB_PATH, API_DB_PATH, LIPID_DB_PATH)

    #Feilsøking:
    df_debug = X.copy()
    lipid_feature_cols = [c for c in df_debug.columns if c.startswith("lipid_") or c.startswith("api_lipid")]
    print_lipid_examples(df_debug, lipid_feature_cols)

    # ---------- Lagre til CSV ----------
    output_dir = BASE_DIR / "ml" 
    output_dir.mkdir(parents=True, exist_ok=True)

    X_path = output_dir / "X.csv"
    y_path = output_dir / "y.csv"

    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False)

    print(f"Saved X to: {X_path}")
    print(f"Saved y to: {y_path}")

    print(X.head())
    print(y.head())

