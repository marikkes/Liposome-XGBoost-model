from pathlib import Path

from make_dataset import make_dataset
from classes.experiment_config import ExperimentConfig
from formulation_utils import load_pca_model
from formulation_run_db import get_run_db_path, save_run
from ml_utils import predict_ensemble

from sklearn.model_selection import cross_val_score, GroupKFold
from train_test_splits import create_split
from sklearn.metrics import mean_absolute_error, r2_score

import xgboost as xgb
import optuna
import numpy as np
import pandas as pd
import joblib

def objective(trial, X, y, groups):
    params = {
        # The number of boosting rounds (trees) to build. A higher value can lead to better performance but may also increase the risk of overfitting.
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000), 
        # The maximum depth of each tree. Deeper trees can capture more complex patterns but may also overfit the training data.
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        # The learning rate controls the contribution of each tree to the final prediction. A lower learning rate may lead to better performance but requires more boosting rounds.
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        # The fraction of data to consider when building each tree. A lower value can help prevent overfitting but may also reduce model performance.
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        # The fraction of features to consider when building each tree. A lower value can help prevent overfitting but may also reduce model performance.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # Controls the minimum number of samples required to create a new node in the tree. A higher value can help prevent overfitting but may also reduce model performance.
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = xgb.XGBRegressor(**params)

    # Dataset is split into n=5 groups, samples within the same group are kept together in either the training or test set, and never split. This prevents data leakage and ensures that the model is evaluated on truly unseen data.
    cv = GroupKFold(n_splits=5)

    # Use cross-validation to evaluate the model's performance, better than a single train-test split that would be sensitive to the specific split.
    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    # We are trying to find hyperparameters to minimize the mean absolute error across the groups, so we return the negative of the mean score.
    return -np.mean(scores)

def main():
    # ---------- Finn database ----------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"
    API_DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"
    LIPID_DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"

    # ---------- Last datasett ----------
    X, y, groups = make_dataset(DB_PATH, API_DB_PATH, LIPID_DB_PATH)

    print("Dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()

    # ---------- Split ----------
    SPLIT_MODE = "within_api" # "random" or "api" or "within_api"

    train_idx, test_idx = create_split(
        X,
        y,
        groups,
        SPLIT_MODE
    )


    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]


    # TEST
    print("Train groups:")
    print(groups_train.value_counts())

    print("\nTest groups:")
    print(groups_test.value_counts())

    print("\nNumber of test samples:", len(test_idx))


    # ---------- Bayesian optimization ----------
    print("Running Bayesian optimization...")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, groups_train), n_trials=100)

    print("Best parameters:")
    print(study.best_params)
    print()

    # ---------- Tren beste modell ----------
    # best_model = xgb.XGBRegressor(
    #     **study.best_params,
    #     objective="reg:squarederror",
    #     random_state=42,
    #     n_jobs=-1
    # )

    # best_model.fit(X_train, y_train)

    # We can train multiple models with the same best parameters to create an ensemble for more robust predictions.
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    models = []

    for i in range(5):

        model = xgb.XGBRegressor(
            **study.best_params,
            objective="reg:squarederror",
            random_state=42+i,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        models.append(model)

        joblib.dump(
            model,
            MODEL_DIR / f"xgb_model_{i}.pkl"
        )

    config = ExperimentConfig(
            models=models,
            X_columns=X.columns,
            api_db_path=API_DB_PATH
        )
    
    if config.lipid_selection_mode == "PCA":
        config.pca_model = load_pca_model(config.n_pca_components)

    # ---------- Evaluering ----------
    y_pred = predict_ensemble(config.models, X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Optimized XGBoost")
    print("------------------")
    print(f"MAE: {mae:.2f} EE%-poeng")
    print(f"R² : {r2:.3f}")
    print()

    # ---------- Feature importance ----------
    feature_importances = np.array([
        model.feature_importances_
        for model in config.models
    ])

    mean_importances = feature_importances.mean(axis=0)
    std_importances = feature_importances.std(axis=0)

    importance_df = pd.DataFrame({
        "feature": config.X_columns,
        "importance": mean_importances,
        "std": std_importances
    }).sort_values(
        "importance",
        ascending=False
    )

    print("Feature importance:")
    print(importance_df.head(20))

    run_db_path = get_run_db_path(BASE_DIR)
    
    comment = input("Describe the changes from the previous run:\n> ").strip()

    if not comment:
        raise RuntimeError("A comment is required to save this run.")

    save_run(
        run_db_path,
        config.api_name,
        "training",
        comment,
        None,
        None,
        {
            "training_model_params": study.best_params,
            "split_mode": SPLIT_MODE,
            "n_trials": len(study.trials),
            "n_models": len(models),
        },
        float(mae),
        float(r2),
    )

    print(f"\n✅ Run saved to database: {run_db_path}")

if __name__ == "__main__":
    main()