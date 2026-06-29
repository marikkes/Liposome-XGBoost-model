import numpy as np
import pandas as pd

from lipid_utils import get_available_lipids, sort_lipids, build_formulation_row

def normalize_weights(weights):
    weights = np.array(weights, dtype=float)
    return weights / weights.sum()

def sort_lipid_weight_pairs(lipids, weights):
    lipid_to_weight = dict(zip(lipids, weights))

    sorted_lipids = sort_lipids(lipids)

    sorted_weights = np.array([
        lipid_to_weight[lipid]
        for lipid in sorted_lipids
    ])

    return sorted_lipids, sorted_weights

# -----------------------------
# Candidate generation
# -----------------------------
def generate_candidates(X_columns, api_profile, n_samples=5000):
    lipids = get_available_lipids(X_columns)

    candidates = []

    for _ in range(n_samples):
        # choose 1–3 lipids, sort them based on type
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

def build_formulation_row(X_columns, chosen_lipids, weights, api_ratio, api_profile=None):
    row = dict.fromkeys(X_columns, 0.0)

    for lipid, w in zip(chosen_lipids, weights):
        row[lipid] = float(w)

    row["n_lipids"] = len(chosen_lipids)
    row["api_to_lipid_ratio"] = float(api_ratio)

    if api_profile is not None:
        for key, value in api_profile.items():
            if key in row:
                row[key] = value

    return row

# Add stronger constraints here if needed, e.g., minimum lipid fraction 0.05 etc. For now, we just ensure that the weights sum to 1.
def generate_lipid_weights(trial, n_lipids):
    """
    Generate lipid fractions using stick-breaking.
    Returns weights that sum to 1.
    """

    if n_lipids == 1:
        return np.array([1.0])

    elif n_lipids == 2:
        u1 = trial.suggest_float(
            "u1",
            0.05,
            0.95
        )

        return np.array([
            u1,
            1 - u1
        ])

    elif n_lipids == 3:
        u1 = trial.suggest_float(
            "u1",
            0.05,
            0.95
        )

        u2 = trial.suggest_float(
            "u2",
            0.05,
            0.95
        )

        return np.array([
            u1,
            (1-u1)*u2,
            (1-u1)*(1-u2)
        ])
    else:
        raise ValueError("Only 1-3 lipids supported")
