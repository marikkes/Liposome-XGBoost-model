from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from get_api_profile import load_api_properties, get_api_profile
from make_dataset import make_dataset
from lipid_utils import get_lipid_type_fraction_columns, sort_lipids, build_formulation_row, extract_present_lipids, lipid_name_from_column
from formulation_utils import generate_candidates


def compute_novelty(candidates, X_train):

    # Fill NaNs ONLY for distance calculation
    candidates_filled = candidates.fillna(0)
    X_train_filled = X_train.fillna(0)
    
    distances = pairwise_distances(candidates_filled, X_train_filled)
    min_dist = distances.min(axis=1)  # nærmeste nabo

    return min_dist

# -----------------------------
# Ensemble prediction
# -----------------------------
def predict_with_uncertainty(models, X):
    preds = np.array([m.predict(X) for m in models])

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    return mean, std

# -----------------------------
# Normalization function
# -----------------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
# Mean, std and novelty have completely different scales and novelty will dominate if we dont normalize

# -----------------------------
# Acquisition function
# -----------------------------
def acquisition(mean, std, novelty, beta=1.5, gamma=0.5):
    return normalize(mean) + beta * normalize(std) + gamma * normalize(novelty)
#Want to obtain max EE and max uncertainty and difference from previous results

# -----------------------------
# Load ensemble models
# -----------------------------
def load_models(path, n_models=5):
    models = []
    for i in range(n_models):
        model = joblib.load(path / f"xgb_model_{i}.pkl")
        models.append(model)
    return models

# -----------------------------
# Selects 5 different experiments, added as we were getting 5 equal ones
# -----------------------------
def select_diverse_top(df, n_select=10, random_state=42):
    """
    Select diverse formulations using max-min distance sampling.
    
    Parameters:
        df (pd.DataFrame): full dataset
        n_select (int): number of formulations to select
    
    Returns:
        pd.DataFrame: selected subset
    """
    df = df.copy()
    
    # -------------------------
    # 1. Define feature columns
    # -------------------------
    X_columns = [
        col for col in df.columns
        if not col.endswith("_missing")
    ]
    
    # Remove anything non-numeric just in case
    X_columns = [col for col in X_columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # -------------------------
    # 2. Scale features
    # -------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(df[X_columns])
    
    # -------------------------
    # 3. Max-min selection
    # -------------------------
    rng = np.random.RandomState(random_state)
    
    # Start with a random point
    selected_idx = [rng.randint(len(df))]
    remaining_idx = list(set(range(len(df))) - set(selected_idx))
    
    for _ in range(n_select - 1):
        selected_points = X[selected_idx]
        remaining_points = X[remaining_idx]
        
        # Compute distance to closest selected point
        distances = pairwise_distances(remaining_points, selected_points)
        min_distances = distances.min(axis=1)
        
        # Pick the point farthest away
        next_idx = remaining_idx[np.argmax(min_distances)]
        
        selected_idx.append(next_idx)
        remaining_idx.remove(next_idx)
    
    return df.iloc[selected_idx]

# -----------------------------
# Suggest experiments
# -----------------------------
def suggest_next(models, X_train, X_columns, api_profile, n_suggestions=5, n_candidates=5000):

    candidates = generate_candidates(X_columns, api_profile, n_candidates)

    mean, std = predict_with_uncertainty(models, candidates)

    novelty = compute_novelty(candidates, X_train)

    candidates["mean_ee"] = mean
    candidates["uncertainty"] = std
    candidates["novelty"] = novelty

    candidates["score"] = acquisition(mean, std, novelty)

    top = select_diverse_top(candidates, n_suggestions)

    return top

# -----------------------------
# Format formulations
# -----------------------------
def format_formulations(df):
    lipid_cols = get_lipid_type_fraction_columns(df.columns)

    formatted = []

    for idx, row in df.iterrows():
        entry = {}

        # Basic info
        entry["api_to_lipid_ratio"] = round(row["api_to_lipid_ratio"], 3)

        # Extract ONLY lipids that are actually present
        present = extract_present_lipids(row, lipid_cols, threshold=0.01)

        sorted_lipids = sort_lipids(list(present.keys()))

        entry["lipids"] = {
            lipid_name_from_column(col): round(present[col], 3)
            for col in sorted_lipids
        }

        entry["n_lipids"] = len(sorted_lipids)

        # Results
        entry["mean_ee"] = round(row["mean_ee"], 2)
        entry["uncertainty"] = round(row["uncertainty"], 2)
        entry["novelty"] = round(row["novelty"], 1)
        entry["score"] = round(row["score"], 3)

        formatted.append(entry)

    return formatted

# -----------------------------
# MAIN
# -----------------------------
def main():

    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "db" / "work" / "formulations_work.db"
    API_DB_PATH = BASE_DIR / "db" / "work" / "api_properties.db"
    LIPID_DB_PATH = BASE_DIR / "db" / "work" / "lipid_properties.db"
    MODEL_DIR = BASE_DIR / "models"

    api_df = load_api_properties(BASE_DIR / "db" / "work" / "api_properties.db")
    api_profile = get_api_profile("Micrococcin P1", api_df)

    X, y = make_dataset(DB_PATH, API_DB_PATH, LIPID_DB_PATH)
    print("TEST TEST")
    print(X.isna().sum().sort_values(ascending=False).head(100))
    print(X.columns)

    print("Loading models...")
    models = load_models(MODEL_DIR, n_models=5)

    print("Generating next experiments...")
    top = suggest_next(models, X, X.columns, api_profile)

    print("\nTop next experiments:")
    formatted = format_formulations(top)

    for i, f in enumerate(formatted, 1):
        print(f"\n--- Experiment {i} ---")
        print(f"n_lipids: {f['n_lipids']}")
        print(f"api_to_lipid_ratio: {f['api_to_lipid_ratio']}")

        print("lipids:")
        for lipid, frac in f["lipids"].items():
            print(f"  - {lipid}: {frac}")

        print(f"mean_ee: {f['mean_ee']}")
        print(f"uncertainty: {f['uncertainty']}")
        print(f"novelty: {f['novelty']}")
        print(f"score: {f['score']}")


if __name__ == "__main__":
    main()