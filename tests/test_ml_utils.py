import numpy as np
import pandas as pd
import pytest

from ml_utils import predict_ensemble


class DummyModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(
            len(X),
            self.value,
            dtype=float,
        )


def test_predict_ensemble_raises_for_empty_models():
    X = pd.DataFrame({
        "a": [1.0, 2.0]
    })

    with pytest.raises(
        ValueError,
        match="Models is empty"
    ):
        predict_ensemble([], X)


def test_predict_ensemble_returns_mean_prediction():
    models = [
        DummyModel(0.2),
        DummyModel(0.8),
    ]

    X = pd.DataFrame({
        "a": [1.0, 2.0, 3.0]
    })

    pred = predict_ensemble(models, X)

    assert np.allclose(
        pred,
        np.array([0.5, 0.5, 0.5])
    )