import sqlite3
from pathlib import Path

# NOT IN USE!!

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Opprett tabell
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lipid_properties (
        lipid_name TEXT PRIMARY KEY,
                   
        is_phospholipid INTEGER,
        is_sterol INTEGER,
        is_cationic_lipid INTEGER,

        molecular_weight REAL,
        rotatable_bond_count INTEGER,

        hbond_acceptor_count INTEGER,
        hbond_donor_count INTEGER,

        logp REAL,
        tpsa REAL,  -- Topological Polar Surface Area

        formal_charge INTEGER,
                   
        number_of_tails INTEGER,
        tail_length_mean REAL,
        tail_length_max INTEGER,
        avg_double_bonds_per_tail REAL,
        is_mixture  INTEGER,
        has_tails INTEGER,
        is_zwitterionic INTEGER,
        is_cationic INTEGER,
        is_anionic INTEGER                      
    );
    """)

    conn.commit()
    conn.close()
    print("Database og tabell opprettet!")

if __name__ == "__main__":
    create_database()