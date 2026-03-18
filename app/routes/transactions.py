"""
Transactions Router â€” score single transactions and batch CSV uploads.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.ml.explain import explain_batch, explain_row
from app.ml.features import build_features, FEATURE_COLUMNS
from app.ml.predict import classify, merge_ml_with_rules, models_ready, score_features
from app.rules.engine import compute_risk_scores, safe_col
from app.schemas.transaction import ScoredTransaction, TransactionIn

router = APIRouter(prefix="/transactions", tags=["Transactions"])

# Simple in-memory store (also written to by alerts.py)
_transaction_store: dict[str, dict] = {}


def get_store() -> dict[str, dict]:
    return _transaction_store


# â”€â”€ POST /transactions/score â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/score", response_model=ScoredTransaction, summary="Score a single transaction")
async def score_single(txn: TransactionIn):
    """
    Accept a single JSON transaction, run rule engine + ML hybrid scoring,
    and return the risk score with SHAP explanations.
    """
    # Build a one-row DataFrame from the input
    row = {
        'Transaction Amount':        txn.amount,
        'Merchant Category Code':    txn.merchant_category_code or '',
        'Merchant Name':             txn.merchant_name or '',
        'SENDER ADDRESS COUNTRY':    (txn.sender_country or '').upper(),
        'RECEIVER ADDRESS COUNTRY':  (txn.receiver_country or '').upper(),
        'CARD NUMBER':               txn.card_number or '',
        'Receiver First Name':       txn.receiver_first_name or '',
        'Receiver Last Name':        txn.receiver_last_name or '',
        'Transaction Date':          txn.transaction_date or datetime.utcnow().isoformat(),
        'Payment Format':            txn.payment_format or '',
        'Account':                   txn.sender_account or '',
        'Account.1':                 txn.receiver_account or '',
    }
    df = pd.DataFrame([row])

    # Rule scoring
    rule_scores = compute_risk_scores(df)
    rule_score  = int(rule_scores['score'].iloc[0])
    flags       = str(rule_scores['flags'].iloc[0])
    breakdown   = str(rule_scores['score_breakdown'].iloc[0])

    # ML scoring
    shap_top3 = []
    ml_avail  = models_ready()
    final_score = rule_score

    if ml_avail:
        try:
            X, _ = build_features(df)
            ml_scores = score_features(X)
            blended   = merge_ml_with_rules(rule_scores['score'], ml_scores)
            final_score = int(blended.iloc[0])
            shap_top3   = explain_row(X, top_n=3)
        except Exception:
            pass   # fall back to rule score

    risk_level = classify(final_score)
    txn_id     = str(uuid.uuid4())

    result = {
        "transaction_id": txn_id,
        "score":          final_score,
        "risk_level":     risk_level,
        "flags":          flags,
        "score_breakdown": breakdown,
        "shap_top3":      shap_top3,
        "ml_available":   ml_avail,
        "input":          txn.model_dump(),
    }
    _transaction_store[txn_id] = result
    return result


# â”€â”€ POST /transactions/upload-csv â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/upload-csv", summary="Batch-score an IBM-format CSV")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV (IBM HI-Small format or any compatible format).
    Returns full analysis payload including KPIs, alerts, SHAP reasons.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, "Only .csv files are supported on this endpoint.")

    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents), low_memory=False)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    try:
        rule_scores = compute_risk_scores(df_raw)
    except Exception as e:
        raise HTTPException(500, f"Rule scoring failed: {e}")

    # ML hybrid scoring
    ml_avail = models_ready()
    final_scores = rule_scores['score'].copy()
    shap_results: list[list[dict]] = [[] for _ in range(len(df_raw))]

    if ml_avail:
        try:
            from app.ml.features import build_features
            X, _ = build_features(df_raw)
            ml_sc   = score_features(X)
            blended = merge_ml_with_rules(rule_scores['score'], ml_sc)
            final_scores = blended
            shap_results = explain_batch(X, top_n=3)
        except Exception:
            pass

    # Assign back
    df_result = df_raw.copy()
    df_result['Risk Score']      = final_scores.values
    df_result['Risk Level']      = final_scores.apply(classify).values
    df_result['Risk Flags']      = rule_scores['flags'].values
    df_result['Score Breakdown'] = rule_scores['score_breakdown'].values
    for c in [col for col in rule_scores.columns if col.startswith('feature_')]:
        df_result[c] = rule_scores[c].values
    df_result['Z-Score'] = rule_scores['feature_zscore'].values

    total   = len(df_result)
    high_n  = int((df_result['Risk Level'] == 'HIGH').sum())
    med_n   = int((df_result['Risk Level'] == 'MEDIUM').sum())
    low_n   = int((df_result['Risk Level'] == 'LOW').sum())
    avg_sc  = float(df_result['Risk Score'].mean())

    # Build KPIs
    kpis = {
        "total": total, "high": high_n, "medium": med_n, "low": low_n,
        "avgScore": round(avg_sc, 1),
        "sarCandidates": high_n,
        "mlAvailable": ml_avail,
    }

    # Top flagged transactions (up to 200)
    ref_col  = safe_col(df_result, 'transaction_reference_number', 'Transaction Reference', 'Account')
    amt_col  = safe_col(df_result, 'Amount Paid', 'Transaction Amount', 'Amount')
    date_col = safe_col(df_result, 'Timestamp', 'Transaction Date', 'Date')

    def _safe(v):
        import numpy as np
        if v is None: return None
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if not isinstance(v, (bool, int, float, str, type(None))): return str(v)
        return v

    transactions = []
    for i, (_, row) in enumerate(df_result.head(200).iterrows()):
        entry = {k: _safe(v) for k, v in row.items()}
        entry['shap_top3'] = shap_results[i] if i < len(shap_results) else []
        entry['_txn_id']   = str(uuid.uuid4())
        transactions.append(entry)
        _transaction_store[entry['_txn_id']] = entry

    # Persist HIGH alerts to alert store
    from app.routes.alerts import _alert_store
    high_rows = df_result[df_result['Risk Level'] == 'HIGH']
    for i, (idx_val, row) in enumerate(high_rows.iterrows()):
        alert_id = str(uuid.uuid4())
        ref = str(row[ref_col]) if ref_col else f"Row-{idx_val}"
        _alert_store[alert_id] = {
            "id":             alert_id,
            "transaction_id": transactions[i]['_txn_id'] if i < len(transactions) else "",
            "ref":            ref,
            "score":          int(row['Risk Score']),
            "risk_level":     str(row['Risk Level']),
            "flags":          str(row['Risk Flags']),
            "status":         "open",
            "severity":       "danger",
            "created_at":     datetime.utcnow().isoformat() + 'Z',
            "shap_top3":      shap_results[i] if i < len(shap_results) else [],
            "all_fields":     {k: _safe(v) for k, v in row.items()},
        }

    return JSONResponse({
        "kpis":         kpis,
        "transactions": transactions,
        "mlAvailable":  ml_avail,
    })


# â”€â”€ GET /transactions/{id} â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/{txn_id}", summary="Get a cached scored transaction")
async def get_transaction(txn_id: str):
    """Retrieve a previously scored transaction by its UUID."""
    txn = _transaction_store.get(txn_id)
    if txn is None:
        raise HTTPException(404, "Transaction not found. Session may have expired.")
    return txn
