from pathlib import Path

from make_dataset import make_dataset
from get_api_profile import get_api_profile, load_api_properties
from formulation_run_db import get_run_db_path, save_run
from lipid_utils import get_lipid_type_fraction_columns, sort_lipids, build_formulation_row, LIPID_PRIORITY

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb
import optuna
import numpy as np
import pandas as pd
import joblib


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**params)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    return -np.mean(scores)

def generate_candidates(X_columns, api_profile, n_samples=5000):
    lipids = get_lipid_type_fraction_columns(X_columns)

    candidates = []

    for _ in range(n_samples):
        # velg 1-3 lipider, sorter dem basert på type
        chosen = np.random.choice(lipids, size=np.random.randint(1, 4), replace=False)
        chosen = sort_lipids(chosen)

        weights = np.random.dirichlet(np.ones(len(chosen)))

        row = build_formulation_row(
            X_columns=X_columns,
            chosen_lipids=chosen,
            weights=weights,
            api_ratio=np.random.uniform(0.05, 0.20),
            api_profile=api_profile,
        )

        candidates.append(row)

    return pd.DataFrame(candidates)


def suggest_formulations(model, X_columns, api_profile):
    candidates = generate_candidates(X_columns, api_profile)

    preds = model.predict(candidates)

    candidates["predicted_ee"] = preds

    top = candidates.sort_values("predicted_ee", ascending=False).head(5)

    return top

def formulation_objective(trial, model, X_columns):
    row = dict.fromkeys(X_columns, 0)

    lipids = get_lipid_type_fraction_columns(X_columns)

    # velg antall lipider
    n_lipids = trial.suggest_int("n_lipids", 1, 3)

    # optuna velger lipider
    chosen = [
        trial.suggest_categorical(f"lipid_{i}", lipids)
        for i in range(n_lipids)
    ]

    # Reject duplicates
    if len(set(chosen)) != len(chosen):
        return -1e6

    # fractions for original lipid order
    weights = np.array([
        trial.suggest_float(f"w_{i}", 0.01, 1.0)
        for i in range(len(chosen))
    ])

    weights = weights / weights.sum()

    # -----------------------------
    # Sort lipids AND weights together
    # -----------------------------
    pairs = sorted(
        zip(chosen, weights),
        key=lambda x: LIPID_PRIORITY.get(
            x[0].replace("lipid_", "").replace("_fraction", ""),
            999
    )
    )

    # unpack sorted pairs
    chosen = [p[0] for p in pairs]
    weights = np.array([p[1] for p in pairs])

    # API ratio (Optuna optimaliserer)
    row["api_to_lipid_ratio"] = trial.suggest_float("api_ratio", 0.05, 0.20)

    row["n_lipids"] = len(chosen)

    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    #trial.set_user_attr("formulation", row)

    penalty = 0.0

    # straff hvis én lipid dominerer for mye
    if max(weights) > 0.9:
        penalty += 5

    trial.set_user_attr(
    "formulation",
    {
        "n_lipids": len(chosen),
        "api_ratio": row["api_to_lipid_ratio"],

        "lipid_0": chosen[0] if len(chosen) > 0 else None,
        "lipid_1": chosen[1] if len(chosen) > 1 else None,
        "lipid_2": chosen[2] if len(chosen) > 2 else None,

        "w_0": float(weights[0]) if len(weights) > 0 else None,
        "w_1": float(weights[1]) if len(weights) > 1 else None,
        "w_2": float(weights[2]) if len(weights) > 2 else None,
    }
    )

    return pred - penalty


def main():
    # ---------- Finn database ----------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"
    API_DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"
    LIPID_DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"

    # ---------- Last datasett ----------
    X, y = make_dataset(DB_PATH, API_DB_PATH, LIPID_DB_PATH)

    print("Dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()

    # ---------- Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ---------- Bayesian optimization ----------
    print("Running Bayesian optimization...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

    print("Best parameters:")
    print(study.best_params)
    print()

    # ---------- Tren beste modell ----------
    best_model = xgb.XGBRegressor(
        **study.best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    best_model.fit(X_train, y_train)

    # ---------- Evaluering ----------
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Optimized XGBoost")
    print("------------------")
    print(f"MAE: {mae:.2f} EE%-poeng")
    print(f"R² : {r2:.3f}")
    print()

    # ---------- Feature importance ----------
    importances = pd.Series(
        best_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("Feature importance:")
    print(importances.head(20))

    print("\nTop suggested formulations:")
    # last API properties
    api_df = load_api_properties(API_DB_PATH)

    # velg API
    API_NAME = "Micrococcin P1"

    api_profile = get_api_profile(API_NAME, api_df)
    top = suggest_formulations(best_model, X.columns, api_profile)
    print(top.T)

    print("\nOptimizing formulations with Bayesian optimization...")

    formulation_study = optuna.create_study(direction="maximize")

    formulation_study.optimize(
        lambda trial: formulation_objective(trial, best_model, X.columns),
        n_trials=100
    )

    print("\nBest formulation found:")
    print(formulation_study.best_params)
    best_formulation = formulation_study.best_trial.user_attrs["formulation"]
    print(best_formulation)

    print("\nPredicted EE:")
    print(formulation_study.best_value)

    run_db_path = get_run_db_path(BASE_DIR)

    comment = input("Describe the changes from the previous run:\n> ").strip()
    if not comment:
        raise RuntimeError("A comment is required to save this run.")

    save_run(
        run_db_path,
        API_NAME,
        comment,
        float(formulation_study.best_value),
        best_formulation,
        {
            "training_model_params": study.best_params,
            "formulation_optimization_trials": formulation_study.best_trial.number,
        },
        float(mae),
        float(r2),
    )
    print(f"\n✅ Run saved to database: {run_db_path}")

    for i in range(5):
        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + i,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        MODEL_DIR = BASE_DIR / "models"
        joblib.dump(model, MODEL_DIR / f"xgb_model_{i}.pkl")


if __name__ == "__main__":
    main()