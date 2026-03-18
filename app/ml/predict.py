"""
Hybrid Scorer â€” combines IsolationForest (30%) + XGBoost (70%) into a
final 0-100 risk score, with graceful fallback to rules-only scoring.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.features import FEATURE_COLUMNS
from app.rules.rule_list import HIGH_THRESHOLD, MEDIUM_THRESHOLD

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE     = Path(__file__).resolve().parent.parent
XGB_PATH = BASE / "models" / "xgboost_model.pkl"
IF_PATH  = BASE / "models" / "isolation_forest.pkl"

# â”€â”€ Lazy model cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_xgb_model = None
_iso_model  = None


def _load_models() -> tuple[Any | None, Any | None]:
    """Load models from disk (cached in module-level vars)."""
    global _xgb_model, _iso_model
    if _xgb_model is None and XGB_PATH.exists():
        with open(XGB_PATH, 'rb') as f:
            _xgb_model = pickle.load(f)
    if _iso_model is None and IF_PATH.exists():
        with open(IF_PATH, 'rb') as f:
            _iso_model = pickle.load(f)
    return _xgb_model, _iso_model


def models_ready() -> bool:
    """Return True if both model files exist on disk."""
    return XGB_PATH.exists() and IF_PATH.exists()


def invalidate_cache() -> None:
    """Call after retraining to force reload on next predict."""
    global _xgb_model, _iso_model
    _xgb_model = None
    _iso_model  = None


# â”€â”€ Core scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def score_features(X: pd.DataFrame) -> np.ndarray:
    """
    Score a feature matrix and return final 0-100 scores (float array).

    Falls back to returning all zeros if models are not loaded
    (caller should then rely entirely on rule-based score).
    """
    xgb, iso = _load_models()
    if xgb is None or iso is None:
        return np.zeros(len(X))

    # Align and fill any missing feature columns
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X_aligned = X[FEATURE_COLUMNS].fillna(0)

    # IsolationForest: score_samples â†’ more negative = more anomalous
    # Convert to 0-1 (anomaly probability) via min-max rescaling
    if_raw  = iso.score_samples(X_aligned)
    if_min  = if_raw.min()
    if_max  = if_raw.max()
    if_span = (if_max - if_min) if (if_max - if_min) > 0 else 1.0
    # invert: high anomaly â†’ high score
    iso_prob = 1.0 - (if_raw - if_min) / if_span

    # XGBoost: probability of class 1 (laundering)
    xgb_prob = xgb.predict_proba(X_aligned)[:, 1]

    # Blend
    blended = 0.30 * iso_prob + 0.70 * xgb_prob

    # Scale to 0-100
    return np.clip(blended * 100, 0, 100)


def merge_ml_with_rules(rule_scores: pd.Series, ml_scores: np.ndarray,
                         ml_weight: float = 0.40) -> pd.Series:
    """
    Blend rule-based scores with ML scores.

    Weight: rules contribute (1 - ml_weight), ML contributes ml_weight.
    This keeps the interpretable rule score dominant while boosting with ML.
    """
    rule_arr = rule_scores.values.astype(float)
    combined = (1 - ml_weight) * rule_arr + ml_weight * ml_scores
    return pd.Series(np.clip(combined, 0, 100).round().astype(int),
                     index=rule_scores.index)


def classify(score: int | float) -> str:
    """Map a 0-100 score to LOW / MEDIUM / HIGH."""
    s = float(score)
    if s >= HIGH_THRESHOLD:
        return 'HIGH'
    if s >= MEDIUM_THRESHOLD:
        return 'MEDIUM'
    return 'LOW'
