import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
work_db_path = BASE_DIR / "db" / "work" / "formulations_work.db"
conn = sqlite3.connect(work_db_path)

# 1. Se de første radene
df = pd.read_sql("SELECT * FROM formulations", conn)
print("Første 5 rader:")
print(df.head())

print("\nAntall rader:", len(df))

# 2. Sjekk at EE er numerisk
print("\nEE-statistikk:")
print(df[["ee_mean", "ee_std"]].describe())

# 3. Eksempelspørring
print("\nMicrococcin P1:")
print(
    pd.read_sql(
        "SELECT api, ee_mean, ee_std FROM formulations "
        "WHERE api = 'Micrococcin P1'",
        conn
    )
)
print(df[df[["lipid_1", "api", "ee_mean"]].isna().all(axis=1)].head())


conn.close()