"""
Model Training â€” XGBoost + IsolationForest on IBM HI-Small dataset.

Usage (from repo root):
    python -m app.ml.train app/data/HI-Small_Trans.csv
Or via the /model/train API endpoint (which calls train_models()).
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from app.ml.features import build_features, load_ibm_csv, FEATURE_COLUMNS

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE = Path(__file__).resolve().parent.parent  # â†’ app/
MODELS_DIR   = BASE / "models"
DATA_DIR     = BASE / "data"
XGB_PATH     = MODELS_DIR / "xgboost_model.pkl"
IF_PATH      = MODELS_DIR / "isolation_forest.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# â”€â”€ Main training function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def train_models(csv_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load IBM dataset, engineer features, train both models, save to disk.

    Returns a dict with precision/recall/f1 and metadata.
    Raises FileNotFoundError if csv_path doesn't exist.
    """
    csv_path = Path(csv_path) if csv_path else DATA_DIR / "HI-Small_Trans.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"IBM dataset not found at {csv_path}. "
            "Please place HI-Small_Trans.csv in app/data/ and retry."
        )

    print(f"[train] Loading dataset: {csv_path}")
    t0 = time.time()

    df = load_ibm_csv(str(csv_path))
    X, y = build_features(df)

    # Keep only known feature columns (fill missing with 0)
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_COLUMNS]

    dataset_rows = len(df)
    print(f"[train] Dataset: {dataset_rows:,} rows, {X.shape[1]} features")

    # â”€â”€ IsolationForest (unsupervised â€” no labels needed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("[train] Training IsolationForest ...")
    iso = IsolationForest(
        n_estimators=100,      # reduced for memory
        max_samples=25600,     # bound memory on huge datasets
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    # Fit IF on a random million rows maximum if huge
    if len(X) > 1_000_000:
        X_if_train = X.sample(n=1_000_000, random_state=42)
    else:
        X_if_train = X
        
    iso.fit(X_if_train)
    with open(IF_PATH, 'wb') as f:
        pickle.dump(iso, f)
    print(f"[train] IsolationForest saved -> {IF_PATH}")

    # â”€â”€ XGBoost (supervised â€” needs Is_Laundering label) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    metrics: dict[str, Any] = {}

    if y is not None and y.sum() > 0:
        print(f"[train] Label distribution: {y.value_counts().to_dict()}")

        pos = int(y.sum())
        neg = int((y == 0).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        xgb = XGBClassifier(
            n_estimators=200,          # reduced for speed
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            tree_method='hist',        # CRITICAL for 5M rows
            device='cuda',             # UTILIZE NVIDIA GPU (RTX 5050)
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = xgb.predict(X_test)
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall    = float(recall_score(y_test, y_pred, zero_division=0))
        f1        = float(f1_score(y_test, y_pred, zero_division=0))

        print(f"[train] XGBoost metrics â€” P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")
        metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    else:
        # No labels â†’ train XGBoost on all data with synthetic labels from IF
        print("[train] No 'Is_Laundering' labels found â€” using IsolationForest as weak supervison...")
        if_scores = iso.score_samples(X)  # more negative = more anomalous
        y_synth   = (if_scores < np.percentile(if_scores, 2)).astype(int)

        xgb = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        xgb.fit(X, y_synth)
        metrics = {'precision': None, 'recall': None, 'f1': None}

    with open(XGB_PATH, 'wb') as f:
        pickle.dump(xgb, f)
    print(f"[train] XGBoost saved -> {XGB_PATH}")

    # â”€â”€ Persist metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from datetime import datetime
    trained_at = datetime.utcnow().isoformat() + 'Z'
    meta = {
        'trained_at':   trained_at,
        'dataset_rows': dataset_rows,
        'model_version': f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}",
        **metrics,
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = round(time.time() - t0, 1)
    print(f"[train] Done in {elapsed}s")
    return meta


# â”€â”€ CLI entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    result = train_models(path)
    print(json.dumps(result, indent=2))
