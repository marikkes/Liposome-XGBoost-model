from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from db_utils.smiles_utils import fetch_smiles
from db_utils.pubchem_synonyms import PUBCHEM_LIPID_SYNONYMS

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "lipid_properties.db"
)

def fetch_lipid_smiles():

    fetch_smiles(
        db_path=DB_PATH,
        table_name="lipid_properties",
        id_column="lipid_name",
        synonyms=PUBCHEM_LIPID_SYNONYMS
    )