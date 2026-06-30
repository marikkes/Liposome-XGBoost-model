import numpy as np
import pandas as pd

import suggest_next_experiments as sne
from experiment_config import ExperimentConfig


class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def test_predict_with_uncertainty_returns_mean_and_std():
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    models = [DummyModel(0.2), DummyModel(0.8)]

    mean, std = sne.predict_with_uncertainty(models, X)

    assert np.allclose(mean, np.array([0.5, 0.5]))
    assert np.allclose(std, np.array([0.3, 0.3]))


def test_suggest_next_adds_scoring_columns(monkeypatch):
    X_columns = pd.Index(["api_to_lipid_ratio", "lipid_DPPC_fraction", "lipid_DOPE_fraction"])
    config = ExperimentConfig(
        models=[DummyModel(0.0)],
        X_columns=X_columns,
        api_profile={"api_molecular_weight": 1000.0},
        api_name="api",
        n_candidates=3,
        n_suggestions=2,
    )

    candidates = pd.DataFrame(
        {
            "api_to_lipid_ratio": [0.1, 0.2, 0.3],
            "lipid_DPPC_fraction": [0.8, 0.7, 0.6],
            "lipid_DOPE_fraction": [0.2, 0.3, 0.4],
        }
    )
    X_existing = candidates.copy()

    monkeypatch.setattr(sne, "generate_candidates", lambda *_args, **_kwargs: candidates.copy())
    monkeypatch.setattr(sne, "predict_with_uncertainty", lambda *_args, **_kwargs: (
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 2.0, 3.0]),
    ))
    monkeypatch.setattr(sne, "compute_novelty", lambda *_args, **_kwargs: np.array([0.5, 0.4, 0.3]))
    monkeypatch.setattr(sne, "select_diverse_top", lambda df, n_select: df.nlargest(n_select, "score"))

    top = sne.suggest_next(config, X_existing)

    assert len(top) == 2
    assert {"mean_ee", "uncertainty", "novelty", "score"}.issubset(top.columns)
    assert top["score"].notna().all()
    assert top.iloc[0]["mean_ee"] == 30.0
    assert (top["score"].values[:-1] >= top["score"].values[1:]).all()

def test_acquisition_prefers_high_prediction():
    mean = np.array([10.0, 20.0])
    std = np.array([0.1, 0.1])
    novelty = np.array([0.1, 0.1])

    score = sne.acquisition(
        mean,
        std,
        novelty,
        beta=0,
        gamma=0
    )

    assert score[1] > score[0]

def test_compute_novelty_returns_closest_distance():
    candidates = pd.DataFrame({"x":[0,10]})
    existing = pd.DataFrame({"x":[1]})

    novelty = sne.compute_novelty(candidates, existing)

    assert novelty[0] < novelty[1]