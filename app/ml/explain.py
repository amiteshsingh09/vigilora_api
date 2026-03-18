"""
SHAP Explanations â€” produces human-readable reasons for each flagged transaction.

Uses shap.TreeExplainer on the XGBoost model.
Falls back gracefully to rule-based flags when model isn't trained.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.features import FEATURE_COLUMNS
from app.rules.rule_list import FEATURE_DISPLAY_NAMES

BASE     = Path(__file__).resolve().parent.parent
XGB_PATH = BASE / "models" / "xgboost_model.pkl"

_explainer = None   # lazy-loaded
_xgb_model  = None


def _load_explainer():
    global _explainer, _xgb_model
    if _explainer is not None:
        return _explainer
    if not XGB_PATH.exists():
        return None
    try:
        import shap
        with open(XGB_PATH, 'rb') as f:
            _xgb_model = pickle.load(f)
        _explainer = shap.TreeExplainer(_xgb_model)
        return _explainer
    except Exception:
        return None


def invalidate_cache():
    global _explainer, _xgb_model
    _explainer = None
    _xgb_model  = None


def explain_row(X_row: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """
    Return the top *top_n* SHAP contributors for a single-row feature DataFrame.

    Each item:
        {"feature": str, "label": str, "impact": float}

    Falls back to an empty list if SHAP/model is unavailable.
    """
    explainer = _load_explainer()
    if explainer is None:
        return []

    try:
        import shap
        # Align columns
        for col in FEATURE_COLUMNS:
            if col not in X_row.columns:
                X_row[col] = 0
        X_aligned = X_row[FEATURE_COLUMNS].fillna(0)

        shap_vals = explainer.shap_values(X_aligned)
        # For binary XGBoost, shap_values is a single array (log-odds space)
        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]   # class-1 SHAP values for first (only) row
        else:
            vals = shap_vals[0]

        # Pair with feature names, sort by |impact|
        contributions = sorted(
            zip(FEATURE_COLUMNS, vals),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:top_n]

        return [
            {
                "feature": feat,
                "label":   FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title()),
                "impact":  round(float(val), 4),
            }
            for feat, val in contributions
            if abs(val) > 0.001   # skip negligible contributions
        ]
    except Exception:
        return []


def explain_batch(X: pd.DataFrame, top_n: int = 3) -> list[list[dict]]:
    """
    Explain every row in *X*. Returns a list of explanation lists.
    Efficient: computes all SHAP values in one shot.
    """
    explainer = _load_explainer()
    if explainer is None:
        return [[] for _ in range(len(X))]

    try:
        import shap
        for col in FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0
        X_aligned = X[FEATURE_COLUMNS].fillna(0)

        shap_vals = explainer.shap_values(X_aligned)
        if isinstance(shap_vals, list):
            vals_matrix = shap_vals[1]
        else:
            vals_matrix = shap_vals

        results = []
        for row_vals in vals_matrix:
            contributions = sorted(
                zip(FEATURE_COLUMNS, row_vals),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:top_n]
            results.append([
                {
                    "feature": feat,
                    "label":   FEATURE_DISPLAY_NAMES.get(feat, feat.replace('_', ' ').title()),
                    "impact":  round(float(val), 4),
                }
                for feat, val in contributions
                if abs(val) > 0.001
            ])
        return results
    except Exception:
        return [[] for _ in range(len(X))]
