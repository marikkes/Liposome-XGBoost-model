import json
import sqlite3
from datetime import datetime
from pathlib import Path


def get_run_db_path(base_dir: Path = None) -> Path:
    base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
    db_path = base_dir / "db" / "work" / "formulation_runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _create_run_table(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS formulation_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        api_name TEXT,
        comment TEXT,
        best_predicted_ee REAL,
        n_lipids INTEGER,
        api_ratio REAL,
        lipid_1 TEXT,
        w_1 REAL,
        lipid_2 TEXT,
        w_2 REAL,
        lipid_3 TEXT,
        w_3 REAL,
        model_type TEXT,
        training_mae REAL,
        training_r2 REAL,
        best_formulation TEXT,
        model_params TEXT
    );
    """)


def ensure_run_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    _create_run_table(cur)
    conn.commit()
    conn.close()


def _normalize_lipid_name(lipid_value):
    if not isinstance(lipid_value, str):
        return lipid_value

    lipid_value = lipid_value.strip()
    prefix = "lipid_"
    suffix = "_fraction"

    if lipid_value.startswith(prefix) and lipid_value.endswith(suffix):
        return lipid_value[len(prefix):-len(suffix)]

    return lipid_value


def _round_value(value):
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value

def _flatten_formulation_params(best_formulation: dict) -> tuple:
    return (
        best_formulation.get("n_lipids"),
        _round_value(best_formulation.get("api_ratio")),

        _normalize_lipid_name(best_formulation.get("lipid_0")),
        _normalize_lipid_name(best_formulation.get("lipid_1")),
        _normalize_lipid_name(best_formulation.get("lipid_2")),

        _round_value(best_formulation.get("w_0")),
        _round_value(best_formulation.get("w_1")),
        _round_value(best_formulation.get("w_2")),
    )

def save_run(
    db_path: Path,
    api_name: str,
    comment: str,
    best_predicted_ee: float,
    best_formulation: dict,
    model_params: dict,
    training_mae: float,
    training_r2: float,
):
    ensure_run_db(db_path)
    best_predicted_ee = _round_value(best_predicted_ee)
    n_lipids, api_ratio, lipid_1, lipid_2, lipid_3, w_1, w_2, w_3 = _flatten_formulation_params(best_formulation)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO formulation_runs (
            created_at,
            api_name,
            comment,
            best_predicted_ee,
            n_lipids,
            api_ratio,
            lipid_1,
            w_1,
            lipid_2,
            w_2,
            lipid_3,
            w_3,
            model_type,
            training_mae,
            training_r2,
            best_formulation,
            model_params
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            api_name,
            comment,
            best_predicted_ee,
            n_lipids,
            api_ratio,
            lipid_1,
            w_1,
            lipid_2,
            w_2,
            lipid_3,
            w_3,
            "xgboost",
            training_mae,
            training_r2,
            json.dumps(best_formulation, ensure_ascii=False),
            json.dumps(model_params, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()
