import numpy as np
import pandas as pd
import pytest

import train_xgboost as txgb
from experiment_config import ExperimentConfig


class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class DummyTrial:
    def __init__(self):
        self.attrs = {}

    def suggest_int(self, name, low, high):
        if name == "n_lipids":
            return 2
        return low

    def suggest_categorical(self, name, choices):
        return choices[0] if name.endswith("0") else choices[1]

    def suggest_float(self, name, low, high, log=False):
        return 0.1

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def test_predict_ensemble_raises_for_empty_models():
    config = ExperimentConfig(
        models=[],
        X_columns=pd.Index(["a"]),
        api_profile={},
        api_name="api",
    )

    with pytest.raises(ValueError, match="models is empty"):
        txgb.predict_ensemble(config, pd.DataFrame({"a": [1.0, 2.0]}))


def test_predict_ensemble_returns_mean_prediction():
    config = ExperimentConfig(
        models=[DummyModel(0.2), DummyModel(0.8)],
        X_columns=pd.Index(["a"]),
        api_profile={},
        api_name="api",
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    pred = txgb.predict_ensemble(config, X)

    assert np.allclose(pred, np.array([0.5, 0.5, 0.5]))


def test_formulation_objective_applies_penalty(monkeypatch):
    config = ExperimentConfig(
        models=[DummyModel(0.9)],
        X_columns=pd.Index([
            "api_to_lipid_ratio",
            "lipid_DPPC_fraction",
            "lipid_DOPE_fraction",
        ]),
        api_profile={"api_molecular_weight": 1000.0},
        api_name="api",
    )
    trial = DummyTrial()

    monkeypatch.setattr(txgb, "get_available_lipids", lambda cols: [
        "lipid_DPPC_fraction",
        "lipid_DOPE_fraction",
    ])
    monkeypatch.setattr(txgb, "generate_lipid_weights", lambda _trial, _n: [0.96, 0.04])
    monkeypatch.setattr(txgb, "sort_lipid_weight_pairs", lambda chosen, weights: (chosen, weights))
    monkeypatch.setattr(txgb, "build_formulation_row", lambda **kwargs: {
        "api_to_lipid_ratio": kwargs["api_ratio"],
        "lipid_DPPC_fraction": kwargs["weights"][0],
        "lipid_DOPE_fraction": kwargs["weights"][1],
    })
    monkeypatch.setattr(txgb, "predict_ensemble", lambda _config, _df: np.array([10.0]))

    score = txgb.formulation_objective(trial, config)
    formulation = trial.attrs["formulation"]

    # Both penalty branches should trigger: 10.0 - 2 - 2
    assert score == pytest.approx(6.0)
    assert "formulation" in trial.attrs
    assert trial.attrs["formulation"]["n_lipids"] == 2
    assert formulation["n_lipids"] == 2
    assert formulation["lipid_0"] is not None
    assert formulation["api_ratio"] == pytest.approx(0.1)
