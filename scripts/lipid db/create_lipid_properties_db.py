from pathlib import Path
import sqlite3
import pandas as pd

from fetch_smiles import fetch_smiles
from calculate_lipid_descriptors import calculate_lipid_descriptors
from import_backup_properties import import_backup_properties
from import_manual_formulation_descriptors import import_manual_formulation_descriptors
#from utils import normalize_api_name

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


# -----------------------
# Lipids to include
# -----------------------

LIPID_LIST = [
    "DPPC",
    "DSPC",
    "DMPC",
    "DAPC",
    "Egg PC",
    "Soy PC",
    "DOPE",
    "DPPG",
    "CHEMS",
    "DOTAP",
    "Chol",
]

def create_empty_database():

    with sqlite3.connect(DB_PATH) as conn:

        df = pd.DataFrame(
            {
                "lipid_name": LIPID_LIST
            }
        )

        #df["lipid_name"] = df["lipid_name"].apply(normalize_api_name) Do we need this part?
        df["descriptor_source"] = None

        df.to_sql(
            "lipid_properties",
            conn,
            if_exists="replace",
            index=False
        )

# -----------------------
# Main
# -----------------------

def main():

    print("Creating empty lipid database...")
    create_empty_database()


    print("\nFetching SMILES from PubChem...")
    fetch_smiles()


    print("\nCalculating RDKit descriptors...")
    calculate_lipid_descriptors()

    #Not currently needed as all lipids have SMILES and descriptors calculated. If we add more lipids in the future, we can use this to fill in missing values.
    #print("\nAdding backup information...")
    #import_backup_properties()

    print("\nAdding manual formulation descriptors...")
    import_manual_formulation_descriptors()

    print("\nDatabase creation complete!")



if __name__ == "__main__":
    main()