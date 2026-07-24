from pathlib import Path
import sqlite3
import pandas as pd


# -----------------------
# Paths
# -----------------------

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
    / "Lipid properties.xlsx"
)

def add_column_if_missing(conn, column_name):

    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA table_info(lipid_properties)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if column_name not in columns:

        cursor.execute(
            f"""
            ALTER TABLE lipid_properties
            ADD COLUMN {column_name} REAL
            """
        )

    conn.commit()

def clean_numeric(series):

    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )



def import_manual_formulation_descriptors():

    # -----------------------
    # Load Excel backup
    # -----------------------

    backup = pd.read_excel(EXCEL_PATH)

    backup = backup.dropna(
        how="all"
    )


    # -----------------------
    # Rename columns
    # -----------------------

    backup = backup.rename(
        columns={
            "Lipid name": "lipid_name",
            "Tail length mean": "tail_length_mean",
            "Average double bonds per tail": "avg_double_bonds_per_tail",
            "Is mixture": "is_mixture",
            "Is zwitterionic": "is_zwitterionic",
        }
    )


    # -----------------------
    # Keep only relevant columns
    # -----------------------

    backup = backup[
        [
            "lipid_name",
            "tail_length_mean",
            "avg_double_bonds_per_tail",
            "is_mixture",
            "is_zwitterionic",
        ]
    ]


    # -----------------------
    # Clean numbers
    # -----------------------

    numeric_cols = [
        "tail_length_mean",
        "avg_double_bonds_per_tail",
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


        # -----------------------
        # Only update lipids missing values
        # -----------------------

        backup = backup[
            backup["lipid_name"].isin(
                db["lipid_name"]
            )
        ]


        print(
            f"Manual formulation values found for {len(backup)} lipids"
        )


        # -----------------------
        # Update database
        # -----------------------

        columns = [
            "tail_length_mean",
            "avg_double_bonds_per_tail",
            "is_mixture",
            "is_zwitterionic",
        ]

        for col in columns:
            add_column_if_missing(conn, col)

        for _, row in backup.iterrows():

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


    print(
        "Manual formulation descriptors import completed!"
    )



if __name__ == "__main__":
    import_manual_formulation_descriptors()

#Should we also add?
#number_of_tails
#tail_length_max
#Has tails
#Is cationic
#Is anionic
#lipid class