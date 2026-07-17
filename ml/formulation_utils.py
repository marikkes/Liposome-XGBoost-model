from pathlib import Path

import numpy as np
import pandas as pd

from lipid_utils import get_available_lipids, lipid_name_from_column, lipid_column_from_name, sort_lipids
from sklearn.neighbors import NearestNeighbors
from classes.experiment_config import ExperimentConfig

import joblib

from classes.pca_model import PCAModel

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

def load_pca_model(n_components):
    model_path = Path(__file__).resolve().parent.parent / "models" / "lipid_pca_model.joblib"
    pca_model = joblib.load(model_path)

    lipid_names = pca_model["lipid_names"]
    lipid_scores = pca_model["scores"][:, :n_components]
    pca = pca_model["pca"]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(lipid_scores)

    return PCAModel(
        lipid_names=lipid_names,
        lipid_scores=lipid_scores,
        pca=pca,
        nn=nn,
    )

def nearest_lipid(point, pca_model, used_lipids=None):
    """
    Returns the nearest unused lipid to a point in PCA space.
    """

    if used_lipids is None:
        used_lipids = []

    _, indices = pca_model.nn.kneighbors(
        point.reshape(1, -1),
        n_neighbors=len(pca_model.lipid_names)
    )
    
    for idx in indices[0]:
        lipid = pca_model.lipid_names[idx]

        if lipid not in used_lipids:
            return lipid

    raise RuntimeError("No unused lipids available.")

def choose_lipids_random(config: ExperimentConfig):
    lipid_columns = get_available_lipids(config.X_columns)
    lipids = [
            lipid_name_from_column(col)
            for col in lipid_columns
        ]
    chosen = np.random.choice(lipids, size=np.random.randint(1, 4), replace=False)

    return chosen


def choose_lipids_pca(n_lipids, config: ExperimentConfig):

    chosen = []

    # Standard deviation of each PC
    scales = np.sqrt(config.pca_model.pca.explained_variance_[:config.n_pca_components])

    for _ in range(n_lipids):

        point = np.random.normal(
            loc=0,
            scale=scales
        )

        lipid = nearest_lipid(
            point,
            config.pca_model,
            used_lipids=chosen
        )

        chosen.append(lipid)

    return np.array(chosen)

# -----------------------------
# Candidate generation
# -----------------------------
def generate_candidates(config: ExperimentConfig, n_samples=5000):
    candidates = []

    for _ in range(n_samples):
        # choose 1–3 lipids, sort them based on type
        chosen = choose_lipids(config)
        chosen = sort_lipids(chosen)

        weights = np.random.dirichlet(np.ones(len(chosen)))

        row = build_formulation_row(
            X_columns=config.X_columns,
            chosen_lipids=chosen,
            weights=weights,
            api_ratio=np.random.uniform(0.05, 0.20),
            api_profile=config.api_profile,
        )


        candidates.append(row)

    return pd.DataFrame(candidates)

def choose_lipids(config: ExperimentConfig):
    if config.lipid_selection_mode == "RANDOM":
        chosen = choose_lipids_random(config)
    elif config.lipid_selection_mode == "PCA":
        n_lipids = np.random.randint(1, 4)
        chosen = choose_lipids_pca(n_lipids, config)
    else:
        raise ValueError(
            f"Unknown lipid selection mode: {config.lipid_selection_mode}"
        )
    return chosen

def build_formulation_row(X_columns, chosen_lipids, weights, api_ratio, api_profile=None):
    row = dict.fromkeys(X_columns, 0.0)

    for lipid, w in zip(chosen_lipids, weights):
        lipid_column = lipid_column_from_name(lipid)

        if lipid_column not in row:
            raise ValueError(
                f"Lipid column {lipid_column} not found in X_columns"
            )
        row[lipid_column] = float(w)

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

def choose_lipid_from_pca_trial(
    trial,
    lipid_index,
    config
):
    """
    Let Optuna choose a point in PCA space,
    then return the nearest lipid.
    """

    n_components = config.n_pca_components

    # PCA ranges from actual lipid space
    scores = config.pca_model.lipid_scores

    chosen_point = []

    for i in range(n_components):

        pc_min = scores[:, i].min()
        pc_max = scores[:, i].max()

        value = trial.suggest_float(
            f"lipid_{lipid_index}_PC{i+1}",
            pc_min,
            pc_max
        )

        chosen_point.append(value)

    point = np.array(chosen_point)

    lipid = nearest_lipid(
        point,
        config.pca_model
    )

    return lipid