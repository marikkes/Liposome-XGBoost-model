import sqlite3
import pandas as pd
import pubchempy as pcp



def get_smiles_from_pubchem(
    name,
    synonyms=None
):
    """
    Retrieve SMILES from PubChem.
    Tries the main name first, then synonyms.
    """

    search_names = [name]

    if synonyms and name in synonyms:
        search_names.extend(
            synonyms[name]
        )


    for search_name in search_names:

        try:

            compounds = pcp.get_compounds(
                search_name,
                "name"
            )

            if compounds:

                print(
                    f"  ✓ Found using '{search_name}'"
                )

                # isomeric_smiles is preferred
                return compounds[0].isomeric_smiles


        except Exception:
            continue


    return None



def add_smiles_columns(
    conn,
    table_name
):

    cursor = conn.cursor()

    cursor.execute(
        f"PRAGMA table_info({table_name})"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    if "smiles" not in columns:

        cursor.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN smiles TEXT
            """
        )


    if "smiles_source" not in columns:

        cursor.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN smiles_source TEXT
            """
        )


    conn.commit()



def fetch_smiles(
    db_path,
    table_name,
    id_column,
    synonyms=None
):

    with sqlite3.connect(db_path) as conn:


        add_smiles_columns(
            conn,
            table_name
        )


        df = pd.read_sql(
            f"""
            SELECT *
            FROM {table_name}
            """,
            conn
        )


        missing_smiles = (
            df["smiles"].isna()
        )


        for idx in df[missing_smiles].index:


            name = df.loc[idx, id_column]


            smiles = get_smiles_from_pubchem(
                name,
                synonyms
            )


            if smiles:

                df.loc[idx, "smiles"] = smiles
                df.loc[idx, "smiles_source"] = "PubChem"

                print(
                    f"Found SMILES: {name}"
                )


            else:

                df.loc[idx, "smiles_source"] = (
                    "Not found"
                )

                print(
                    f"No SMILES found: {name}"
                )


        # Update database

        for _, row in df.iterrows():

            conn.execute(
                f"""
                UPDATE {table_name}
                SET smiles = ?,
                    smiles_source = ?
                WHERE {id_column} = ?
                """,
                (
                    row["smiles"],
                    row["smiles_source"],
                    row[id_column]
                )
            )


        conn.commit()


    print("\nSMILES update completed!")