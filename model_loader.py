"""
Loads the five trained LightGBM models (2 regressors, 3 classifiers)
and the feature name order they were trained with.
"""

from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent / "models"

_models = None


def get_models():
    global _models
    if _models is None:
        _models = {
            "tmax": joblib.load(MODELS_DIR / "tmax_model.joblib"),
            "tmin": joblib.load(MODELS_DIR / "tmin_model.joblib"),
            "heat": joblib.load(MODELS_DIR / "heat_model.joblib"),
            "freeze": joblib.load(MODELS_DIR / "freeze_model.joblib"),
            "rain": joblib.load(MODELS_DIR / "rain_model.joblib"),
            "features": joblib.load(MODELS_DIR / "features.joblib"),
        }
    return _models
