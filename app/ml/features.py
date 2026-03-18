"""
Feature Engineering for IBM HI-Small Transactions Dataset.

Produces a clean feature matrix (X) and optional label vector (y)
that can be passed directly to train.py or predict.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.rules.rule_list import (
    FATF_HIGH_RISK, STRUCTURING_BAND_LOW, STRUCTURING_BAND_HIGH,
    IBM_COL_MAP,
)


# â”€â”€ Column aliases accepted from the IBM dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_TS_COLS    = ('Timestamp', 'Transaction Date', 'Date', 'timestamp')
_FROM_ACCT  = ('Account', 'Sender Account', 'From Account', 'sender_account')
_TO_ACCT    = ('Account.1', 'Receiver Account', 'To Account', 'receiver_account')
_FROM_BANK  = ('From Bank', 'Sender Bank', 'sender_country')
_TO_BANK    = ('To Bank', 'Receiver Bank', 'receiver_country')
_AMT_PAID   = ('Amount Paid', 'Transaction Amount', 'Amount', 'amount', 'amount_usd')
_AMT_RECV   = ('Amount Received',)
_PAY_FMT    = ('Payment Format', 'payment_format', 'Payment Type')
_LABEL      = ('Is Laundering', 'is_laundering', 'label')


def _find(df: pd.DataFrame, *candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_ibm_csv(path: str) -> pd.DataFrame:
    """Read and lightly clean the IBM HI-Small CSV with memory optimization."""
    # Read with memory-efficient engine and default types
    try:
        df = pd.read_csv(path, engine='pyarrow', dtype_backend='pyarrow')
    except Exception:
        df = pd.read_csv(path, low_memory=False)
        
    # Deduplicate column names (IBM CSV has two 'Account' columns)
    cols = [str(c).strip() for c in df.columns]
    seen = {}
    for i, c in enumerate(cols):
        if c in seen:
            seen[c] += 1
            cols[i] = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
    df.columns = cols

    # Downcast and normalize string columns
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype('category')
        
    # Downcast floats
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    return df


# â”€â”€ Stage 2: Feature Engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Engineer all ML features from *df*.

    Returns
    -------
    X : pd.DataFrame  â€” feature matrix
    y : pd.Series | None â€” label (Is_Laundering), or None if column absent
    """
    out = pd.DataFrame(index=df.index)

    amt_col   = _find(df, *_AMT_PAID)
    ts_col    = _find(df, *_TS_COLS)
    from_acct = _find(df, *_FROM_ACCT)
    to_acct   = _find(df, *_TO_ACCT)
    from_bank = _find(df, *_FROM_BANK)
    to_bank   = _find(df, *_TO_BANK)
    pay_col   = _find(df, *_PAY_FMT)
    label_col = _find(df, *_LABEL)

    # â”€â”€ Amount features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if amt_col:
        amt = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        out['amount_usd'] = amt

        # Z-score (handle zero-std edge case)
        mu, sigma = amt.mean(), amt.std()
        out['amount_zscore'] = ((amt - mu) / sigma).clip(-5, 5) if sigma > 0 else 0.0

        # Round-number flags
        out['is_round_1k']  = ((amt % 1000 == 0) & (amt >= 5000)).astype(int)
        out['is_round_500'] = ((amt % 500  == 0) & (amt >= 2000) & ~(amt % 1000 == 0)).astype(int)

        # Structuring / just-below-threshold
        out['is_just_below_threshold'] = (
            (amt >= STRUCTURING_BAND_LOW) & (amt <= STRUCTURING_BAND_HIGH)
        ).astype(int)
    else:
        for col in ('amount_usd', 'amount_zscore', 'is_round_1k',
                    'is_round_500', 'is_just_below_threshold'):
            out[col] = 0

    # â”€â”€ Time features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if ts_col:
        try:
            ts = pd.to_datetime(df[ts_col], errors='coerce')
            out['hour_of_day'] = ts.dt.hour.fillna(12).astype(int)
            out['is_weekend']  = ts.dt.dayofweek.isin([5, 6]).astype(int)
            # Odd hours: 23:00 - 05:00
            out['is_odd_hours'] = ts.dt.hour.apply(
                lambda h: 1 if (h >= 23 or h < 5) else 0
            ).fillna(0).astype(int)
        except Exception:
            out['hour_of_day'] = 12
            out['is_weekend']  = 0
            out['is_odd_hours'] = 0
    else:
        out['hour_of_day'] = 12
        out['is_weekend']  = 0
        out['is_odd_hours'] = 0

    # â”€â”€ Geographic / counterparty features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if from_bank and to_bank:
        sc = df[from_bank].astype(str).str.upper().str.strip()
        rc = df[to_bank].astype(str).str.upper().str.strip()
        out['is_cross_border'] = (sc != rc).astype(int)
        out['is_fatf_country'] = (
            sc.isin(FATF_HIGH_RISK.keys()) | rc.isin(FATF_HIGH_RISK.keys())
        ).astype(int)
    else:
        out['is_cross_border'] = 0
        out['is_fatf_country'] = 0

    # â”€â”€ New payee flag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if from_acct and to_acct:
        pair = df[from_acct].astype(str) + '-' + df[to_acct].astype(str)
        # For training data: first occurrence of a pair = new payee
        out['is_new_payee'] = (~pair.duplicated(keep='first')).astype(int)
    else:
        out['is_new_payee'] = 0

    # â”€â”€ Payment format encoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pay_col:
        le = LabelEncoder()
        out['payment_format_encoded'] = le.fit_transform(
            df[pay_col].fillna('Unknown').astype(str)
        )
    else:
        out['payment_format_encoded'] = 0

    # â”€â”€ Velocity features (rolling counts per sender account) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if from_acct and ts_col:
        try:
            ts_series = pd.to_datetime(df[ts_col], errors='coerce')
            acct_series = df[from_acct].astype(str)

            # Build a time-indexed helper df for rolling
            helper = pd.DataFrame({
                'ts':   ts_series,
                'acct': acct_series,
                'one':  1,
            }).sort_values('ts')

            helper = helper.set_index('ts')
            vel_1h  = helper.groupby('acct')['one'].transform(
                lambda s: s.rolling('1h',  min_periods=1).sum()
            )
            vel_6h  = helper.groupby('acct')['one'].transform(
                lambda s: s.rolling('6h',  min_periods=1).sum()
            )
            vel_24h = helper.groupby('acct')['one'].transform(
                lambda s: s.rolling('24h', min_periods=1).sum()
            )

            # Re-align to original index
            out['velocity_1hr']  = vel_1h.reindex(helper.index).values
            out['velocity_6hr']  = vel_6h.reindex(helper.index).values
            out['velocity_24hr'] = vel_24h.reindex(helper.index).values
        except Exception:
            out['velocity_1hr']  = 1
            out['velocity_6hr']  = 1
            out['velocity_24hr'] = 1
    else:
        out['velocity_1hr']  = 1
        out['velocity_6hr']  = 1
        out['velocity_24hr'] = 1

    # Fill any remaining NaN
    out = out.fillna(0)

    # â”€â”€ Label â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    y = None
    if label_col:
        y = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

    return out, y


# â”€â”€ Convenience: feature names in a stable order â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FEATURE_COLUMNS = [
    'amount_usd', 'amount_zscore', 'is_round_1k', 'is_round_500',
    'is_just_below_threshold',
    'hour_of_day', 'is_weekend', 'is_odd_hours',
    'is_cross_border', 'is_fatf_country', 'is_new_payee',
    'payment_format_encoded',
    'velocity_1hr', 'velocity_6hr', 'velocity_24hr',
]
