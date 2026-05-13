from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql("SELECT * FROM formulations", conn)
conn.close()

qty_cols = ["lipid_1_qty", "lipid_2_qty", "lipid_3_qty", "api_qty"]

for col in qty_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.head())
print(df.shape)

df["ee_mean"].hist(bins=20)
plt.xlabel("Encapsulation efficiency (%)")
plt.ylabel("Count")
plt.show()