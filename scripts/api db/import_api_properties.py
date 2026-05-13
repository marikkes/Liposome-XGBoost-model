from pathlib import Path
import pandas as pd
import sqlite3

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"

EXCEL_PATH = BASE_DIR / "data" / "excel" / "API properties.xlsx" 


def clean_numeric(series):
    """
    Konverterer '1144,4' -> 1144.4 osv.
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def main():
    # ---------- Les Excel ----------
    df = pd.read_excel(EXCEL_PATH)

    # ---------- Rename kolonner (match DB!) ----------
    df = df.rename(columns={
        "API name": "api",
        "Molecular weight (g/mol)": "molecular_weight",
        "Rotatable bond count": "rotatable_bond_count",
        "Hydrogen bond acceptor count": "hbond_acceptor_count",
        "Hydrogen bond donor count": "hbond_donor_count",
        "Heavy atom count": "heavy_atom_count",
        "Topological polar surface area (Å2)": "tpsa",
        "Complexity": "complexity"
    })

    # ---------- Rydd numeriske kolonner ----------
    numeric_cols = [
        "molecular_weight",
        "rotatable_bond_count",
        "hbond_acceptor_count",
        "hbond_donor_count",
        "heavy_atom_count",
        "tpsa",
        "complexity"
    ]

    for col in numeric_cols:
        df[col] = clean_numeric(df[col])

    # ---------- Koble til DB ----------
    conn = sqlite3.connect(DB_PATH)

    # ---------- Skriv til database ----------
    df.to_sql(
        "api_properties",
        conn,
        if_exists="replace", 
        index=False
    )

    conn.close()

    print("Data importert til database!")
    print(f"Antall rader: {len(df)}")


if __name__ == "__main__":
    main()