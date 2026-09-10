from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna

from make_dataset import make_dataset
from lipid_utils import get_available_lipids, lipid_name_from_column
from formulation_utils import choose_lipid_from_pca_trial, sort_lipid_weight_pairs, build_formulation_row, generate_lipid_weights, load_pca_model
from formulation_run_db import get_run_db_path, save_run
from classes.experiment_config import ExperimentConfig
from ml_utils import predict_ensemble

def formulation_objective(trial, config: ExperimentConfig):
    # Choose number of lipids (1, 2, or 3)
    n_lipids = trial.suggest_categorical("n_lipids", [1, 2, 3])

    # Choose lipids
    if config.lipid_selection_mode == "PCA":
        chosen = [
            choose_lipid_from_pca_trial(
                trial,
                i,
                config
            )
            for i in range(n_lipids)
        ]
    
    elif config.lipid_selection_mode == "RANDOM":
        lipid_columns = get_available_lipids(config.X_columns)

        lipids = [
            lipid_name_from_column(col)
            for col in lipid_columns
        ]
    
        chosen = [
            trial.suggest_categorical(f"lipid_{i}", lipids)
            for i in range(n_lipids)
        ]
    else:
        raise ValueError(
            "Unknown lipid selection mode"
        )

    # Reject duplicate lipids
    if len(set(chosen)) != len(chosen):
        return -1e6

    # Generate lipid weights
    weights = generate_lipid_weights(trial, n_lipids)

    # Sort lipid names and weights together
    chosen, weights = sort_lipid_weight_pairs(
        chosen,
        weights
        )

    # Range for api_to_lipid_ratio: 0.001 to 1.0 (log scale), change if needed
    api_ratio = trial.suggest_float(
        "api_ratio",
        config.api_ratio_min,
        config.api_ratio_max,
        log=True
    )

    # Build model input
    row = build_formulation_row(
        X_columns=config.X_columns,
        chosen_lipids=chosen,
        weights=weights,
        api_ratio=api_ratio,
        api_profile=config.api_profile
    )

    df = pd.DataFrame([row])

    # Predict EE using the ensemble of trained models
    pred = predict_ensemble(config, df)[0]

    # Formulation penalties
    penalty = 0.0

    # penalty if one lipid dominates too much (e.g., > 95% of total weight)
    if max(weights) > 0.95:
        penalty += 2

    # penalty if one lipid has too small a share
    if min(weights) < 0.05:
        penalty += 2

    # Save formulation details to the trial
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

    BASE_DIR = Path(__file__).resolve().parent.parent

    DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"
    API_DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"
    LIPID_DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"

    MODEL_DIR = BASE_DIR / "models"

    # -----------------------------
    # Load dataset
    # -----------------------------

    X, _, _ = make_dataset(
        DB_PATH,
        API_DB_PATH,
        LIPID_DB_PATH
    )

    # -----------------------------
    # Load trained models
    # -----------------------------

    models = []

    for i in range(5):

        model = joblib.load(
            MODEL_DIR / f"xgb_model_{i}.pkl"
        )

        models.append(model)

    print(f"Loaded {len(models)} trained models.")

    # -----------------------------
    # Create experiment config
    # -----------------------------

    config = ExperimentConfig(
        models=models,
        X_columns=X.columns,
        api_db_path=API_DB_PATH
    )

    # Load PCA model if needed
    if config.lipid_selection_mode == "PCA":

        config.pca_model = load_pca_model(
            config.n_pca_components
        )

    # -----------------------------
    # Optimize formulation
    # -----------------------------

    print("\nOptimizing formulation using Optuna...")

    formulation_study = optuna.create_study(direction="maximize")

    formulation_study.optimize(
        lambda trial: formulation_objective(trial, config),
        n_trials=config.n_formulation_trials
    )

    # -----------------------------
    # Print result
    # -----------------------------

    print("\nBest formulation found:")
    print(formulation_study.best_params)

    best_formulation = formulation_study.best_trial.user_attrs["formulation"]

    print("\nFormulation:")
    print(best_formulation)

    print("\nPredicted EE:")
    print(formulation_study.best_value)

    # -----------------------------
    # Optimization statistics
    # -----------------------------

    trials_df = formulation_study.trials_dataframe()

    valid_trials = trials_df[
        trials_df["value"].notna()
        & (trials_df["value"] > -100)
    ]

    print("\nPerformance by number of lipids:")

    print(
        valid_trials
        .groupby("params_n_lipids")["value"]
        .agg(
            n_trials="count",
            best="max",
            mean="mean",
            median="median",
            std="std"
        )
        .sort_index()
    )

    invalid_trials = trials_df[
        trials_df["value"] <= -100
    ]

    print("\nInvalid trials:")

    print(
        invalid_trials["params_n_lipids"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------
    # Optimization convergence
    # -----------------------------

    values = [
        t.value
        for t in formulation_study.trials
        if t.value > 0
    ]

    best_so_far = []

    current_best = -np.inf

    for value in values:
        current_best = max(current_best, value)
        best_so_far.append(current_best)


    plt.figure(figsize=(8,5))

    plt.plot(best_so_far)

    plt.xlabel("Valid trial")
    plt.ylabel("Best predicted EE so far")
    plt.title("Optuna convergence")

    plt.savefig(
            MODEL_DIR / "optuna_convergence.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close()

    run_db_path = get_run_db_path(BASE_DIR)

    comment = input("Describe the changes from the previous run:\n> ").strip()

    if not comment:
            raise RuntimeError("A comment is required to save this run.")

    save_run(
        run_db_path,
        config.api_name,
        "formulation",
        comment,
        float(formulation_study.best_value),
        best_formulation,
        {
            "formulation_params": formulation_study.best_params,
            "formulation_optimization_trials": formulation_study.best_trial.number,
            "n_formulation_trials": config.n_formulation_trials,
        },
        None,
        None,
    )

    print(f"\n✅ Run saved to database: {run_db_path}")

if __name__ == "__main__":
    main()