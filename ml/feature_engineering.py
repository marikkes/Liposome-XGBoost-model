from pathlib import Path
import sqlite3
import pandas as pd

def load_and_engineer_features(db_path: Path) -> pd.DataFrame:
    """
    Leser rå formuleringer fra databasen og returnerer
    en DataFrame med feature-engineerede kolonner.
    """

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM formulations", conn)
    conn.close()

    # ---------- Velg rader med target ----------
    df = df[df["ee_mean"].notna()].copy()

    # ---------- Enkle numeriske features ----------
    df["n_lipids"] = df[["lipid_1", "lipid_2", "lipid_3"]].notna().sum(axis=1)

    df["total_lipid_qty"] = (
        df["lipid_1_qty"].fillna(0)
        + df["lipid_2_qty"].fillna(0)
        + df["lipid_3_qty"].fillna(0)
    )
    
    lipid_cols = ["lipid_1", "lipid_2", "lipid_3"]

    unique_lipids = (
        df[lipid_cols]
        .melt(value_name="lipid")["lipid"]
        .dropna()
        .unique()
    )

    for lipid in unique_lipids:
        qty = (
            (df["lipid_1"] == lipid) * df["lipid_1_qty"].fillna(0)
            + (df["lipid_2"] == lipid) * df["lipid_2_qty"].fillna(0)
            + (df["lipid_3"] == lipid) * df["lipid_3_qty"].fillna(0)
        )

        df[f"lipid_{lipid}_fraction"] = qty / df["total_lipid_qty"]
    
    # We impute api_qty where it is missing:
    df["api_qty_missing"] = df["api_qty"].isna().astype(int)
    df["api_qty"] = df["api_qty"].fillna(df["api_qty"].median())

    # ---------- Slot-based lipid fractions ----------
    df["lipid_1_fraction"] = df["lipid_1_qty"] / df["total_lipid_qty"]
    df["lipid_2_fraction"] = (df["lipid_2_qty"] / df["total_lipid_qty"]).fillna(0)
    df["lipid_3_fraction"] = (df["lipid_3_qty"] / df["total_lipid_qty"]).fillna(0)

    # Then calculate the ratio:
    df["api_to_lipid_ratio"] = df["api_qty"] / df["total_lipid_qty"]

    # ---------- Sjekk resultat ----------
    # print(df[[
    #     "n_lipids",
    #     "total_lipid_qty",
    #     "api_to_lipid_ratio",
    #     "ee_mean"
    # ]].head())

    return df