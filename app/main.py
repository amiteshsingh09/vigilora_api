"""
Vigilora AML â€” FastAPI Entry Point (v3 â€” ML-powered)

New modular backend:
    /transactions/*  â€” score transactions, batch CSV upload
    /alerts/*        â€” alert lifecycle management
    /model/*         â€” train, metrics, status

Legacy routes (Vercel frontend compatibility):
    /api/*           â€” all original api.py endpoints, mounted as sub-app

Run:
    uvicorn app.main:app --reload --port 5000
Docs at: http://localhost:5000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    import warnings
    warnings.filterwarnings("ignore")
    # Preload models to avoid cold starts on first request
    from app.ml.predict import _load_models
    from app.ml.explain import _load_explainer
    try:
        _load_models()
        _load_explainer()
        print("ML models loaded successfully at startup.")
    except Exception as e:
        print(f"Failed to preload ML models at startup: {e}")
    yield

from app.routes import transactions, alerts, model

# â”€â”€ New modular app â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI(
    title="Vigilora AML API",
    version="3.0",
    description=(
        "ML-powered Anti-Money Laundering backend with XGBoost + IsolationForest scoring, "
        "SHAP explanations, alert management, and full backward compatibility with v2 endpoints."
    ),
    lifespan=lifespan,
)

# â”€â”€ CORS (same origins as legacy api.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5001",
        "http://127.0.0.1:5001",
        "https://vigilora-dashboard.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ New route modules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app.include_router(transactions.router)
app.include_router(alerts.router)
app.include_router(model.router)

# â”€â”€ Mount legacy api.py under /api (backward compatibility) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from api import app as _legacy_app   # type: ignore[import]
    app.mount("/api", _legacy_app)
    _legacy_mounted = True
except Exception as e:
    _legacy_mounted = False
    print(f"[main] Warning: could not mount legacy api.py: {e}")

# â”€â”€ Root health check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/", tags=["Health"])
def root():
    return {
        "status":        "Vigilora AML API v3.9 â€” ML-powered",
        "docs":          "/docs",
        "legacy_api":    "/api (backward-compat)" if _legacy_mounted else "not mounted",
        "new_endpoints": ["/transactions", "/alerts", "/model"],
    }


@app.get("/health", tags=["Health"])
def health():
    from app.ml.predict import models_ready
    return {
        "status":      "ok",
        "ml_ready":    models_ready(),
    }
