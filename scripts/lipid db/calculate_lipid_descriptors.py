from pathlib import Path

from db_utils.descriptor_pipeline import calculate_database_descriptors


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DB_PATH = (
    BASE_DIR
    / "db"
    / "work"
    / "lipid_properties.db"
)

def calculate_lipid_descriptors():

    calculate_database_descriptors(
        db_path=DB_PATH,
        table_name="lipid_properties",
        id_column="lipid_name"
    )