import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
master_dir = BASE_DIR / "db" / "master"
work_db = BASE_DIR / "db" / "work" / "formulations_work.db"

work_db.parent.mkdir(parents=True, exist_ok=True)

# Finn siste master basert på filnavn
latest_master = sorted(master_dir.glob("formulations_*.db"))[-1]

shutil.copy(latest_master, work_db)

print(f"✅ Working-database opprettet fra {latest_master.name}")
