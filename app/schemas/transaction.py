"""
Pydantic request / response schemas for Vigilora AML.
"""
from __future__ import annotations
from typing import Optional, List, Any
from pydantic import BaseModel, Field


# â”€â”€ Inbound â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TransactionIn(BaseModel):
    """A single transaction submitted for scoring."""
    amount: float = Field(..., description="Transaction amount in USD")
    payment_format: Optional[str] = Field(None, description="Wire, ACH, Cash, etc.")
    sender_account: Optional[str] = None
    receiver_account: Optional[str] = None
    sender_country: Optional[str] = Field(None, description="ISO 3-letter country code")
    receiver_country: Optional[str] = Field(None, description="ISO 3-letter country code")
    merchant_category_code: Optional[str] = Field(None, alias="mcc")
    merchant_name: Optional[str] = None
    transaction_date: Optional[str] = Field(None, description="ISO 8601 timestamp")
    card_number: Optional[str] = None
    receiver_first_name: Optional[str] = None
    receiver_last_name: Optional[str] = None
    transaction_reference: Optional[str] = None

    model_config = {"populate_by_name": True}


# â”€â”€ SHAP explanation atom â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ShapContribution(BaseModel):
    feature: str
    label: str
    impact: float   # signed â€” positive = raises risk


# â”€â”€ Outbound â€” scored transaction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ScoredTransaction(BaseModel):
    transaction_id: str
    score: int                          # 0-100
    risk_level: str                     # LOW | MEDIUM | HIGH
    flags: str                          # pipe-separated rule flags
    score_breakdown: str                # human-readable breakdown
    shap_top3: List[ShapContribution] = []
    ml_available: bool = False          # True when models are loaded
    input: Optional[dict] = None        # echo the raw input back


# â”€â”€ Alert â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AlertOut(BaseModel):
    id: str
    transaction_id: str
    ref: str
    score: int
    risk_level: str
    flags: str
    status: str = "open"        # open | resolved | escalated
    severity: str = "warning"   # warning | danger
    created_at: str
    shap_top3: List[ShapContribution] = []
    all_fields: Optional[dict] = None


class AlertAction(BaseModel):
    note: Optional[str] = Field(None, description="Optional compliance note")


# â”€â”€ Model endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TrainResponse(BaseModel):
    status: str
    dataset_rows: int
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    trained_at: str
    message: str = ""


class MetricsResponse(BaseModel):
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    dataset_rows: Optional[int]
    trained_at: Optional[str]
    model_version: str


class ModelStatus(BaseModel):
    ready: bool
    model_version: str
    trained_at: Optional[str]
    xgboost_path: Optional[str]
    isolation_forest_path: Optional[str]
    message: str = ""
