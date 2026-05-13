import sqlite3
from pathlib import Path

# NOT IN USE!!

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Opprett tabell
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_properties (
        api TEXT PRIMARY KEY,

        molecular_weight REAL,
        rotatable_bond_count INTEGER,

        hbond_acceptor_count INTEGER,
        hbond_donor_count INTEGER,

        heavy_atom_count INTEGER,
        tpsa REAL,  -- Topological Polar Surface Area

        complexity REAL
    );
    """)

    conn.commit()
    conn.close()
    print("Database og tabell opprettet!")

if __name__ == "__main__":
    create_database()