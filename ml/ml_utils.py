import numpy as np
import joblib

def predict_ensemble(models, X):
    if not models:
        raise ValueError("Models is empty; load/train models before calling predict_ensemble().")
    
    predictions = np.array([
        model.predict(X)
        for model in models
    ])

    return predictions.mean(axis=0)

def predict_with_uncertainty(models, X):
    preds = np.array([m.predict(X) for m in models])

    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    return mean, std

def load_models(path, n_models=5):
    models = []
    for i in range(n_models):
        model = joblib.load(path / f"xgb_model_{i}.pkl")
        models.append(model)
    return models
