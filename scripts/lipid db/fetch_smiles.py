from pathlib import Path
import sqlite3
import pandas as pd
import pubchempy as pcp

PUBCHEM_SYNONYMS = {
    "DAPC": [
        "1,2-diarachidoyl-sn-glycero-3-phosphocholine",
    ],
    "DPPC": [
        "1,2-dipalmitoyl-sn-glycero-3-phosphocholine",
    ],
    "DSPC": [
        "1,2-distearoyl-sn-glycero-3-phosphocholine",
    ],
    "DMPC": [
        "1,2-dimyristoyl-sn-glycero-3-phosphocholine",
    ],
    "DOPE": [
        "1,2-dioleoyl-sn-glycero-3-phosphoethanolamine",
    ],
    "DPPG": [
        "1,2-dipalmitoyl-sn-glycero-3-phosphoglycerol",
    ],
    "CHEMS": [
        "Cholesteryl hemisuccinate",
    ],
    "DOTAP": [
        "1,2-dioleoyl-3-(trimethylammonium)propane",
    ],
    "Chol": [
        "Cholesterol",
    ],
    "Egg PC": [
        "L-α-phosphatidylcholine (Egg, Chicken)",
        "L-alpha-phosphatidylcholine (Egg, Chicken)",
    ],
    "Soy PC": [
        "L-α-phosphatidylcholine (Soy)",
        "L-alpha-phosphatidylcholine (Soy)",
    ],
}

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "lipid_properties.db"
)


def get_smiles_from_pubchem(lipid_name):
    """
    Retrieve isomeric SMILES from PubChem.
    First tries the lipid name, then any known synonyms.
    """

    # Build list of search terms
    search_names = [lipid_name]

    if lipid_name in PUBCHEM_SYNONYMS:
        search_names.extend(PUBCHEM_SYNONYMS[lipid_name])

    # Try each search term
    for name in search_names:

        try:
            compounds = pcp.get_compounds(
                name,
                "name"
            )

            if compounds:
                print(f"  ✓ Found using '{name}'")
                return compounds[0].smiles

        except Exception:
            continue
    
    return None



def add_columns_if_missing(conn):

    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA table_info(lipid_properties)"
    )

    existing_columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    if "smiles" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE lipid_properties
            ADD COLUMN smiles TEXT
            """
        )


    if "smiles_source" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE lipid_properties
            ADD COLUMN smiles_source TEXT
            """
        )


    conn.commit()



def fetch_smiles():

    with sqlite3.connect(DB_PATH) as conn:

        # Add columns if needed
        add_columns_if_missing(conn)


        # Load lipids
        df = pd.read_sql(
            "SELECT * FROM lipid_properties",
            conn
        )


        # Only lookup lipids without SMILES
        missing_smiles = df["smiles"].isna()


        for idx in df[missing_smiles].index:

            lipid_name = df.loc[idx, "lipid_name"]

            smiles = get_smiles_from_pubchem(
                lipid_name
            )

            if smiles:

                df.loc[idx, "smiles"] = smiles
                df.loc[idx, "smiles_source"] = "PubChem"

                print(
                    f"Found SMILES: {lipid_name}"
                )

            else:

                df.loc[idx, "smiles_source"] = (
                    "Not found"
                )

                print(
                    f"No SMILES found: {lipid_name}"
                )


        # Update only the new columns
        for _, row in df.iterrows():

            conn.execute(
                """
                UPDATE lipid_properties
                SET smiles = ?,
                    smiles_source = ?
                WHERE lipid_name = ?
                """,
                (
                    row["smiles"],
                    row["smiles_source"],
                    row["lipid_name"]
                )
            )


        conn.commit()


    print("\nSMILES update completed!")