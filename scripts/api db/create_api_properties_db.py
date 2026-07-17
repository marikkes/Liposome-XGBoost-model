from pathlib import Path
import sqlite3
import pandas as pd

from fetch_smiles import fetch_smiles
from calculate_api_descriptors import calculate_api_descriptors
from import_backup_properties import import_backup_properties
from utils import normalize_api_name

# -----------------------
# Paths
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "api_properties.db"
)


# -----------------------
# APIs to include
# -----------------------

API_LIST = [
    "Micrococcin P1",
    "Daptomycin",
    "Metformin",
    "Cathelicidin (LL37)",
    "Sirolimus",
    "Zinc sulfate",
    "Clarithromycin",
    "Doxorubicin",
    "Bortezomib",
    "Celecoxib",
    "Vinorelbine",
    "(CRA)- H P 228",
    "Silymarin",
    "Glatiramer acetate",
    "Apigenin",
    "5-Fluorouracil",
    #"Filgrastim",
    "Silibinin",
    "Curcumin",
    "Vancomycin",
    "Glutamine",
    "Allantoin",
    "Levamisole",
    "Albendazole",
    "Ceftriaxone",
    "Vancomycin HCl",
    "Naringenin",
    "Ritonavir",
    "Nisin",
    "Granulocyte-Colony Stimulating Factor",
    "Azithromycin dihydrate"
]

def create_empty_database():

    with sqlite3.connect(DB_PATH) as conn:

        df = pd.DataFrame(
            {
                "api": API_LIST
            }
        )

        df["api"] = df["api"].apply(normalize_api_name)
        df["descriptor_source"] = None

        df.to_sql(
            "api_properties",
            conn,
            if_exists="replace",
            index=False
        )

# -----------------------
# Main
# -----------------------

def main():

    print("Creating empty API database...")
    create_empty_database()


    print("\nFetching SMILES from PubChem...")
    fetch_smiles()


    print("\nCalculating RDKit descriptors...")
    calculate_api_descriptors()


    print("\nAdding backup information...")
    import_backup_properties()


    print("\nDatabase creation complete!")



if __name__ == "__main__":
    main()