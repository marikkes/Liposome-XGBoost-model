import sqlite3
from pathlib import Path
from datetime import date

BASE_DIR = Path(__file__).resolve().parent.parent

today = date.today().isoformat()  # YYYY-MM-DD
db_path = BASE_DIR / "db" / "master" / f"formulations_{today}.db"

db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE formulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    lipid_1 TEXT,
    lipid_1_qty REAL,

    lipid_2 TEXT,
    lipid_2_qty REAL,

    lipid_3 TEXT,
    lipid_3_qty REAL,

    api TEXT,
    api_qty REAL,

    ee_mean REAL,
    ee_std REAL,

    doi TEXT
);
""")

conn.commit()
conn.close()

print(f"✅ Initial master opprettet: {db_path.name}")