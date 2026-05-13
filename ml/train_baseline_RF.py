from pathlib import Path

from make_dataset import make_dataset

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd


def main():
    # ---------- Finn database ----------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"

    # ---------- Last ML-klart datasett ----------
    X, y = make_dataset(DB_PATH)

    print("Dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()

    # ---------- Train / test split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------- Baseline-modell ----------
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ---------- Evaluering ----------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Baseline Random Forest")
    print("----------------------")
    print(f"MAE: {mae:.2f} EE%-poeng")
    print(f"R² : {r2:.3f}")
    print()

    # ---------- Feature importance ----------
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("Feature importance:")
    print(importances)


if __name__ == "__main__":
    main()