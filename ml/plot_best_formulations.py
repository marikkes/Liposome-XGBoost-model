import argparse
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "db" / "work" / "formulation_runs.db"


def load_run_history(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Run database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT id, created_at, best_predicted_ee FROM formulation_runs ORDER BY created_at", conn
    )
    conn.close()

    if df.empty:
        raise ValueError("No formulation runs found in the database.")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.sort_values("created_at")
    return df


def plot_best_formulations(db_path: Path, output_path: Path | None = None) -> None:
    df = load_run_history(db_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["created_at"], df["best_predicted_ee"], marker="o", linestyle="-", label="Predicted EE")

    if len(df) >= 3:
        df["rolling_mean"] = df["best_predicted_ee"].rolling(window=3, min_periods=1).mean()
        ax.plot(
            df["created_at"],
            df["rolling_mean"],
            linestyle="--",
            color="tab:orange",
            label="3-run moving average",
        )

    ax.set_title("Best Predicted EE Over Time")
    ax.set_xlabel("Run timestamp")
    ax.set_ylabel("Predicted EE")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the best predicted EE values from formulation run history."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to formulation_runs.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot image",
    )
    args = parser.parse_args()

    plot_best_formulations(args.db, args.output)


if __name__ == "__main__":
    main()
