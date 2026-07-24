from pathlib import Path
import sqlite3
import pandas as pd

from db_utils.rdkit_descriptors import calculate_rdkit_descriptors
from db_utils.database_utils import add_column_if_missing


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "lipid_properties.db"
)

def calculate_database_descriptors(db_path, table_name, id_column):

    with sqlite3.connect(db_path) as conn:

        df = pd.read_sql(
            f"SELECT * FROM {table_name}",
            conn
        )

        # -----------------------
        # Calculate descriptors
        # -----------------------

        descriptor_df = (
            df["smiles"]
            .apply(calculate_rdkit_descriptors)
            .apply(pd.Series)
        )

        df.loc[
            descriptor_df.dropna(how="all").index,
            "descriptor_source"
        ] = "RDKit"

        # -----------------------
        # Add missing columns
        # -----------------------

        for column in descriptor_df.columns:
            add_column_if_missing(conn, table_name, column)

        # -----------------------
        # Merge descriptors
        # -----------------------

        df = pd.concat(
            [
                df,
                descriptor_df
            ],
            axis=1
        )

        # -----------------------
        # Update database
        # -----------------------

        calc_columns = descriptor_df.columns.tolist()

        calc_columns.append(
            "descriptor_source"
        )

        update_sql = f"""
            UPDATE {table_name}
            SET {", ".join(f"{col} = ?" for col in calc_columns)}
            WHERE {id_column} = ?
        """

        for _, row in df.iterrows():

            values = [row[col] for col in calc_columns]
            values.append(row[id_column])

            conn.execute(
                update_sql,
                values
            )

        conn.commit()