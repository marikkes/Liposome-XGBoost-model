from pathlib import Path
import pandas as pd
import sqlite3

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"

EXCEL_PATH = BASE_DIR / "data" / "excel" / "Lipid properties.xlsx" 


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
    df = df.dropna(how="all")

    # ---------- Rename kolonner (match DB!) ----------
    df = df.rename(columns={
        "Lipid name": "lipid_name",
        "Full name": "full_name",
        "Lipid class": "lipid_class",
        "Molecular weight (g/mol)": "molecular_weight",
        "Rotatable bond count": "rotatable_bond_count",
        "Hydrogen bond acceptor count": "hbond_acceptor_count",
        "Hydrogen bond donor count": "hbond_donor_count",
        "LogP": "logp",
        "Topological polar surface area (Å2)": "tpsa",
        "Formal charge": "formal_charge",
        "Number of tails": "number_of_tails", #Alltid 2 for fosfolipidene, fjerne?
        "Tail length mean": "tail_length_mean",
        "Tail length max": "tail_length_max",
        "Average double bonds per tail": "avg_double_bonds_per_tail",
        "Is mixture": "is_mixture",
        "Is zwitterionic": "is_zwitterionic",
        "Is cationic": "is_cationic",
        "Is anionic": "is_anionic"
    })

    df = df[df["lipid_name"].notna()]
    df = df[df["molecular_weight"].notna()]

    # Converting lipid class into boolean features
    df["is_phospholipid"] = (df["lipid_class"] == "Phospholipid").astype(int)
    df["is_sterol"] = (df["lipid_class"] == "Sterol").astype(int)
    # Add more classes here if needed!

    # Drop original lipid class text column and full name
    df = df.drop(columns=["lipid_class"])
    df = df.drop(columns=["full_name"])

    # ---------- Rydd numeriske kolonner ----------
    numeric_cols = [
        "molecular_weight",
        "rotatable_bond_count",
        "hbond_acceptor_count",
        "hbond_donor_count",
        "logp",
        "tpsa",
        "formal_charge",
        "number_of_tails",
        "tail_length_mean",
        "tail_length_max",
        "avg_double_bonds_per_tail"
    ]

    for col in numeric_cols:
        df[col] = clean_numeric(df[col])

    # ---------- Koble til DB ----------
    conn = sqlite3.connect(DB_PATH)

    # ---------- Skriv til database ----------
    df.to_sql(
        "lipid_properties",
        conn,
        if_exists="replace", 
        index=False
    )

    conn.close()

    print("Data importert til database!")
    print(f"Antall rader: {len(df)}")


if __name__ == "__main__":
    main()