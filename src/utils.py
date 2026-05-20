"""
Helper functions to load saved models (Phase 6).
Used by gui.py in Phase 7.
"""

import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_models():
    """Load all trained models and scaler from models/ folder."""
    crop_model = joblib.load(MODELS_DIR / "crop_model.joblib")
    kmeans_model = joblib.load(MODELS_DIR / "kmeans_model.joblib")
    yield_model = joblib.load(MODELS_DIR / "yield_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return crop_model, kmeans_model, yield_model, scaler
