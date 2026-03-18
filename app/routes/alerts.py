"""
Alerts Router â€” in-memory alert management for AML compliance workflow.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.ml.explain import explain_row
from app.ml.features import build_features
import pandas as pd

router = APIRouter(prefix="/alerts", tags=["Alerts"])

# Shared in-memory store â€” populated by transactions.py on batch upload
_alert_store: dict[str, dict] = {}


# â”€â”€ GET /alerts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("", summary="List all alerts")
async def list_alerts(
    tier:   Optional[str] = Query(None, description="Filter by risk level: HIGH | MEDIUM | LOW"),
    status: Optional[str] = Query(None, description="Filter by status: open | resolved | escalated"),
    limit:  int           = Query(50,   ge=1, le=500),
):
    alerts = list(_alert_store.values())

    if tier:
        alerts = [a for a in alerts if a.get('risk_level', '').upper() == tier.upper()]
    if status:
        alerts = [a for a in alerts if a.get('status', '') == status.lower()]

    # Sort: open first, then by score desc
    alerts = sorted(alerts, key=lambda a: (a.get('status') != 'open', -a.get('score', 0)))

    return JSONResponse({
        "total":  len(alerts),
        "alerts": alerts[:limit],
    })


# â”€â”€ POST /alerts/{id}/resolve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/{alert_id}/resolve", summary="Mark an alert as resolved")
async def resolve_alert(alert_id: str, note: Optional[str] = None):
    alert = _alert_store.get(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert['status']      = 'resolved'
    alert['resolved_at'] = datetime.utcnow().isoformat() + 'Z'
    if note:
        alert['resolution_note'] = note
    return JSONResponse({"status": "ok", "alert": alert})


# â”€â”€ POST /alerts/{id}/escalate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/{alert_id}/escalate", summary="Escalate alert for SAR filing")
async def escalate_alert(alert_id: str, note: Optional[str] = None):
    alert = _alert_store.get(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found.")
    alert['status']       = 'escalated'
    alert['escalated_at'] = datetime.utcnow().isoformat() + 'Z'
    alert['severity']     = 'danger'
    if note:
        alert['escalation_note'] = note
    return JSONResponse({"status": "ok", "alert": alert})


# â”€â”€ GET /alerts/{id}/explain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/{alert_id}/explain", summary="Full SHAP explanation for an alert")
async def explain_alert(alert_id: str):
    alert = _alert_store.get(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found.")

    # Return pre-computed SHAP if available
    if alert.get('shap_top3'):
        return JSONResponse({
            "alert_id":   alert_id,
            "ref":        alert.get('ref'),
            "score":      alert.get('score'),
            "risk_level": alert.get('risk_level'),
            "shap_top3":  alert.get('shap_top3'),
            "flags":      alert.get('flags'),
        })

    # Try to re-compute from all_fields
    all_fields = alert.get('all_fields')
    if all_fields:
        try:
            df = pd.DataFrame([all_fields])
            X, _ = build_features(df)
            shap_top3 = explain_row(X, top_n=5)
            alert['shap_top3'] = shap_top3
            return JSONResponse({
                "alert_id":   alert_id,
                "ref":        alert.get('ref'),
                "score":      alert.get('score'),
                "risk_level": alert.get('risk_level'),
                "shap_top3":  shap_top3,
                "flags":      alert.get('flags'),
            })
        except Exception as e:
            return JSONResponse({
                "alert_id":   alert_id,
                "shap_top3":  [],
                "flags":      alert.get('flags'),
                "note":       f"SHAP unavailable: {str(e)}",
            })

    return JSONResponse({"alert_id": alert_id, "shap_top3": [], "flags": alert.get('flags')})
