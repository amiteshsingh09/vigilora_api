"""
Model Router â€” train, metrics, and status endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.transaction import MetricsResponse, ModelStatus, TrainResponse

router = APIRouter(prefix="/model", tags=["Model"])

BASE         = Path(__file__).resolve().parent.parent   # â†’ app/
METRICS_PATH = BASE / "models" / "metrics.json"
XGB_PATH     = BASE / "models" / "xgboost_model.pkl"
IF_PATH      = BASE / "models" / "isolation_forest.pkl"
DATA_PATH    = BASE / "data"   / "HI-Small_Trans.csv"


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# â”€â”€ POST /model/train â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/train", response_model=TrainResponse, summary="Train / retrain on IBM dataset")
async def train(background_tasks: BackgroundTasks):
    """
    Trigger (re-)training of XGBoost + IsolationForest.
    Requires HI-Small_Trans.csv to be present at app/data/.
    Training runs synchronously (may take 1-3 minutes for large datasets).
    """
    if not DATA_PATH.exists():
        raise HTTPException(
            404,
            detail=(
                f"Dataset not found at {DATA_PATH}. "
                "Please upload HI-Small_Trans.csv to app/data/ and retry."
            )
        )

    try:
        from app.ml.train import train_models
        from app.ml.predict import invalidate_cache as inv_pred
        from app.ml.explain import invalidate_cache as inv_exp

        meta = train_models(str(DATA_PATH))

        # Bust cached model objects so next predict reloads from disk
        inv_pred()
        inv_exp()

        from datetime import datetime
        return TrainResponse(
            status="ok",
            dataset_rows=meta.get('dataset_rows', 0),
            precision=meta.get('precision'),
            recall=meta.get('recall'),
            f1=meta.get('f1'),
            trained_at=meta.get('trained_at', datetime.utcnow().isoformat() + 'Z'),
            message="Training complete. Models saved to app/models/.",
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")


# â”€â”€ GET /model/metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/metrics", response_model=MetricsResponse, summary="Model evaluation metrics")
async def get_metrics():
    """Return precision, recall, F1 from the most recent training run."""
    meta = _load_metrics()
    if not meta:
        raise HTTPException(404, "No training metrics found. Run /model/train first.")
    return MetricsResponse(
        precision=meta.get('precision'),
        recall=meta.get('recall'),
        f1=meta.get('f1'),
        dataset_rows=meta.get('dataset_rows'),
        trained_at=meta.get('trained_at'),
        model_version=meta.get('model_version', 'unknown'),
    )


# â”€â”€ GET /model/status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/status", response_model=ModelStatus, summary="Model readiness and version")
async def get_status():
    """Check whether trained model files exist and return metadata."""
    meta = _load_metrics()
    ready = XGB_PATH.exists() and IF_PATH.exists()
    return ModelStatus(
        ready=ready,
        model_version=meta.get('model_version', 'not-trained'),
        trained_at=meta.get('trained_at'),
        xgboost_path=str(XGB_PATH) if XGB_PATH.exists() else None,
        isolation_forest_path=str(IF_PATH) if IF_PATH.exists() else None,
        message=(
            "Models ready for inference."
            if ready else
            "No trained models found. POST /model/train with IBM dataset to enable ML scoring."
        ),
    )
