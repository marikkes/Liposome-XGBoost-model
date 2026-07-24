from pathlib import Path
import pandas as pd
import sqlite3

#from utils import normalize_api_name


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "lipid_properties.db"
)

EXCEL_PATH = (
    BASE_DIR
    / "data"
    / "excel"
    / "lipid properties.xlsx"
)

def clean_numeric(series):

    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )

def import_backup_properties():

    # -----------------------
    # Load backup Excel
    # -----------------------

    backup = pd.read_excel(EXCEL_PATH)

    backup = backup.dropna(
        how="all"
    )

    backup = backup.rename(
        columns={
            "Lipid name": "lipid_name",
            "Molecular weight (g/mol)": "molecular_weight",
            "LogP": "logp",
            "Rotatable bond count": "rotatable_bond_count",
            "Hydrogen bond acceptor count": "hbond_acceptor_count",
            "Hydrogen bond donor count": "hbond_donor_count",
            #"Heavy atom count": "heavy_atom_count", Need to add this to the excel file
            "Topological polar surface area (Å2)": "tpsa",
            #"Complexity": "complexity", Need to add this to the excel file
            #"Defined atom stereocenter count": "defined_atom_stereocenter_count",
            #"Defined bond stereocenter count": "defined_bond_stereocenter_count",
            #"Number of rings": "number_of_rings"
        }
    )

    #backup["lipid_name"] = backup["lipid_name"].apply(normalize_api_name) Do we need this?

    numeric_cols = [
        "molecular_weight",
        "logp",
        "rotatable_bond_count",
        "hbond_acceptor_count",
        "hbond_donor_count",
        #"heavy_atom_count",
        "tpsa",
        #"complexity",
        #"defined_atom_stereocenter_count",
        #"defined_bond_stereocenter_count",
        #"number_of_rings"
    ]


    for col in numeric_cols:
        backup[col] = clean_numeric(
            backup[col]
        )



    with sqlite3.connect(DB_PATH) as conn:

        db = pd.read_sql(
            "SELECT * FROM lipid_properties",
            conn
        )

        # Only lipids missing SMILES
        missing = db[
            db["smiles"].isna()
        ]["lipid_name"]

        backup = backup[
            backup["lipid_name"].isin(missing)
        ]

        print(
            f"Backup values found for {len(backup)} lipids"
        )


        for _, row in backup.iterrows():

            columns = numeric_cols + [
                "descriptor_source"
            ]

            row["descriptor_source"] = "Manual backup"

            values = [
                row[col]
                for col in columns
            ]

            values.append(
                row["lipid_name"]
            )


            conn.execute(
                f"""
                UPDATE lipid_properties
                SET {", ".join(
                    f"{c}=?" for c in columns
                )}
                WHERE lipid_name = ?
                """,
                values
            )


        conn.commit()


    print("Backup import completed!")