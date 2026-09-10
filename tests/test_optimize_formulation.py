import numpy as np
import pandas as pd
import pytest

import optimize_formulation as opt
from classes.experiment_config import ExperimentConfig


class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(
            len(X),
            self.value,
            dtype=float,
        )


class DummyTrial:
    def __init__(self):
        self.attrs = {}

    def suggest_int(self, name, low, high):
        return low

    def suggest_categorical(self, name, choices):
        if name == "n_lipids":
            return 2
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        return 0.1

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def test_formulation_objective_applies_penalty(monkeypatch):

    config = ExperimentConfig(
        models=[DummyModel(0.9)],
        X_columns=pd.Index([
            "api_to_lipid_ratio",
            "lipid_DPPC_fraction",
            "lipid_DOPE_fraction",
        ]),
        api_db_path=None,
        api_profile={
            "api_molecular_weight": 1000.0
        },
        api_name="api",
    )

    trial = DummyTrial()

    monkeypatch.setattr(
        opt,
        "get_available_lipids",
        lambda cols: [
            "lipid_DPPC_fraction",
            "lipid_DOPE_fraction",
        ],
    )

    lipid_selection_index = [0]

    def mock_choose_lipid(*args):
        lipid_selection_index[0] += 1

        return (
            "lipid_DPPC_fraction"
            if lipid_selection_index[0] % 2 == 0
            else "lipid_DOPE_fraction"
        )

    monkeypatch.setattr(
        opt,
        "choose_lipid_from_pca_trial",
        mock_choose_lipid,
    )

    monkeypatch.setattr(
        opt,
        "generate_lipid_weights",
        lambda _trial, _n: [0.96, 0.04],
    )

    monkeypatch.setattr(
        opt,
        "sort_lipid_weight_pairs",
        lambda chosen, weights: (chosen, weights),
    )

    monkeypatch.setattr(
        opt,
        "build_formulation_row",
        lambda **kwargs: {
            "api_to_lipid_ratio": kwargs["api_ratio"],
            "lipid_DPPC_fraction": kwargs["weights"][0],
            "lipid_DOPE_fraction": kwargs["weights"][1],
        },
    )

    monkeypatch.setattr(
        opt,
        "predict_ensemble",
        lambda _models, _df: np.array([10.0]),
    )

    score = opt.formulation_objective(
        trial,
        config,
    )

    formulation = trial.attrs["formulation"]

    # Both penalty branches should trigger:
    # 10.0 - 2 - 2 = 6.0
    assert score == pytest.approx(6.0)

    assert "formulation" in trial.attrs
    assert formulation["n_lipids"] == 2
    assert formulation["lipid_0"] is not None
    assert formulation["api_ratio"] == pytest.approx(0.1)