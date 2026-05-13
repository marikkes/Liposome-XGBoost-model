import shutil
from pathlib import Path
from datetime import date
import re
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent

work_db = BASE_DIR / "db" / "work" / "formulations_work.db"
master_dir = BASE_DIR / "db" / "master"
archive_dir = BASE_DIR / "db" / "archive"

master_dir.mkdir(parents=True, exist_ok=True)
archive_dir.mkdir(parents=True, exist_ok=True)

today = date.today().isoformat()

# ---------- Finn neste versjonsnummer ----------
pattern = re.compile(rf"formulations_{today}_v(\d+)\.db")
existing_versions = []

for db in master_dir.glob(f"formulations_{today}_v*.db"):
    match = pattern.match(db.name)
    if match:
        existing_versions.append(int(match.group(1)))

next_version = max(existing_versions, default=0) + 1
version_str = f"v{next_version:03d}"

new_master = master_dir / f"formulations_{today}_{version_str}.db"

# ---------- Spør bruker om kommentar ----------
print("\n📝 Klargjør ny master-database")
comment = input("Beskriv endringene i denne versjonen:\n> ").strip()

if not comment:
    raise RuntimeError("❌ Kommentar er påkrevd for å promotere master")

# ---------- Kopier working → ny master ----------
shutil.copy(work_db, new_master)

# ---------- Åpne ny master og legg inn meta-informasjon ----------
conn = sqlite3.connect(new_master)
cur = conn.cursor()

# Sørg for at meta-tabellen finnes
cur.execute("""
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT,
    date TEXT,
    comment TEXT,
    n_formulations INTEGER
);
""")

# Tell antall formuleringer
cur.execute("SELECT COUNT(*) FROM formulations")
n_formulations = cur.fetchone()[0]

# Sett inn metadata
cur.execute("""
INSERT INTO meta (version, date, comment, n_formulations)
VALUES (?, ?, ?, ?)
""", (
    version_str,
    today,
    comment,
    n_formulations
))

conn.commit()
conn.close()

# ---------- Arkiver forrige master ----------
all_masters = sorted(master_dir.glob("formulations_*.db"))
if len(all_masters) > 1:
    shutil.copy(all_masters[-2], archive_dir / all_masters[-2].name)

print(f"\n✅ Ny master opprettet: {new_master.name}")
print(f"📊 Antall formuleringer: {n_formulations}")
print("📝 Kommentar lagret i meta-tabellen")