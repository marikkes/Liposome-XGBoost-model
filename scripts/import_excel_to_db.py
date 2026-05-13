import pandas as pd
import sqlite3
import re
from pathlib import Path

# ---------- 1. Hjelpefunksjon for EE parsing ----------
def parse_ee(value):
    """
    Returnerer (mean, std).
    Std blir None hvis den ikke finnes.
    """
    if pd.isna(value):
        return None, None

    value = str(value).strip()

    # Standardavvik rapportert
    match = re.search(r"([\d.]+)\s*[±+/-]\s*([\d.]+)", value)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Bare én verdi
    try:
        return float(value), None
    except ValueError:
        return None, None


# ---------- 2. Les Excel ----------
BASE_DIR = Path(__file__).resolve().parent.parent

excel_path = (
    BASE_DIR
    / "data"
    / "excel"
    / "Data collection final file for ML model.xlsx"
)

df = pd.read_excel(excel_path)
print(df.columns)


# Gi kolonnene ryddige navn
df = df.rename(columns={
    "Sr. no.": "sr_no",
    "L-1": "lipid_1",
    "Qty": "lipid_1_qty",
    "L-2": "lipid_2",
    "Qty.1": "lipid_2_qty",
    "L-3": "lipid_3",
    "Qty.2": "lipid_3_qty",
    "Drug": "drug_peptide",
    "Unnamed: 8": "drug_non_peptide",
    "Qty of drug": "api_qty",
    "EE (%)": "ee_raw"
})

# Kombiner peptide- og non-peptide API til én kolonne
df["api"] = df["drug_peptide"].combine_first(df["drug_non_peptide"])

# Fjern helt tomme kolonner (som 'Unnamed: 8')
df = df.drop(columns=["drug_peptide", "drug_non_peptide"])

# ---------- 3. Splitt EE ----------
df[["ee_mean", "ee_std"]] = df["ee_raw"].apply(
    lambda x: pd.Series(parse_ee(x))
)


# Definer hva som er en gyldig formulering
df = df.dropna(subset=["lipid_1", "api"], how="any")

# DOI kan fylles inn manuelt senere (foreløpig NULL)
df["doi"] = None

# ---------- 4. Velg relevante kolonner ----------
numeric_cols = [
    "lipid_1_qty", "lipid_2_qty", "lipid_3_qty",
    "api_qty", "ee_mean", "ee_std"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
   
df_db = df[
    [
        "lipid_1", "lipid_1_qty",
        "lipid_2", "lipid_2_qty",
        "lipid_3", "lipid_3_qty",
        "api", "api_qty",
        "ee_mean", "ee_std",
        "doi"
    ]
]

# ---------- 5. Skriv til SQLite ----------
work_db_path = BASE_DIR / "db" / "work" / "formulations_work.db"
conn = sqlite3.connect(work_db_path)

df_db.to_sql(
    "formulations",
    conn,
    if_exists="append",
    index=False
)

conn.close()

print("✅ Data importert til SQLite")
