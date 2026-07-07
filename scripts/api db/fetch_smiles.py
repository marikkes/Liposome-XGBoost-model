from pathlib import Path
import sqlite3
import pandas as pd
import pubchempy as pcp

PUBCHEM_SYNONYMS = {
    "Cathelicidin (LL37)": [
        "LL-37",
        "cathelicidin LL-37",
    ]
}

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "api_properties.db"
)


def get_smiles_from_pubchem(api_name):
    """
    Retrieve isomeric SMILES from PubChem.
    First tries the API name, then any known synonyms.
    """

    # Build list of search terms
    search_names = [api_name]

    if api_name in PUBCHEM_SYNONYMS:
        search_names.extend(PUBCHEM_SYNONYMS[api_name])

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
        "PRAGMA table_info(api_properties)"
    )

    existing_columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    if "smiles" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE api_properties
            ADD COLUMN smiles TEXT
            """
        )


    if "smiles_source" not in existing_columns:

        cursor.execute(
            """
            ALTER TABLE api_properties
            ADD COLUMN smiles_source TEXT
            """
        )


    conn.commit()



def fetch_smiles():

    with sqlite3.connect(DB_PATH) as conn:

        # Add columns if needed
        add_columns_if_missing(conn)


        # Load APIs
        df = pd.read_sql(
            "SELECT * FROM api_properties",
            conn
        )


        # Only lookup APIs without SMILES
        missing_smiles = df["smiles"].isna()


        for idx in df[missing_smiles].index:

            api_name = df.loc[idx, "api"]

            smiles = get_smiles_from_pubchem(
                api_name
            )


            if smiles:

                df.loc[idx, "smiles"] = smiles
                df.loc[idx, "smiles_source"] = "PubChem"

                print(
                    f"Found SMILES: {api_name}"
                )

            else:

                df.loc[idx, "smiles_source"] = (
                    "Not found"
                )

                print(
                    f"No SMILES found: {api_name}"
                )


        # Update only the new columns
        for _, row in df.iterrows():

            conn.execute(
                """
                UPDATE api_properties
                SET smiles = ?,
                    smiles_source = ?
                WHERE api = ?
                """,
                (
                    row["smiles"],
                    row["smiles_source"],
                    row["api"]
                )
            )


        conn.commit()


    print("\nSMILES update completed!")