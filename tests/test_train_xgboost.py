import numpy as np
import pandas as pd

import train_xgboost as txgb


class DummyTrial:
    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


def test_objective_returns_cross_validation_mae(monkeypatch):
    X = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    groups = pd.Series([0, 0, 1, 1, 2])

    expected_scores = np.array([-1.0, -2.0, -3.0])

    def mock_cross_val_score(*args, **kwargs):
        return expected_scores

    monkeypatch.setattr(
        txgb,
        "cross_val_score",
        mock_cross_val_score,
    )

    trial = DummyTrial()

    result = txgb.objective(
        trial,
        X,
        y,
        groups,
    )

    # objective returns the negative mean of neg-MAE scores
    assert result == np.mean([1.0, 2.0, 3.0])