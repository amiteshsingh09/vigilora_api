"""
Rules Engine â€” applies all declarative AML rules to a DataFrame.

This is a clean refactor of compute_risk_scores() from api.py,
now reading constants from rule_list.py so rules are easy to update.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from app.rules.rule_list import (
    HIGH_RISK_MCC, FATF_HIGH_RISK,
    AMOUNT_SCORE, MCC_SCORE, CROSS_BORDER_SCORE, FATF_SCORE, FATF_CROSS_BONUS,
    ROUND_1K_SCORE, ROUND_500_SCORE, RECEIVER_VEL_SCORE, CARD_VEL_SCORE,
    ZSCORE_ANOMALY_SCORE, STRUCTURING_SCORE,
    STRUCTURING_BAND_LOW, STRUCTURING_BAND_HIGH,
    RECEIVER_REPEAT_THRESHOLD, CARD_REPEAT_THRESHOLD,
    HIGH_THRESHOLD, MEDIUM_THRESHOLD,
)


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def safe_col(df: pd.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def normalize_mcc(val) -> str:
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val).strip()


# â”€â”€ main engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all rule-based AML checks to *df* and return a scores DataFrame.

    Columns returned:
        score, risk_level, flags, score_breakdown,
        feature_amount, feature_high_risk_mcc, feature_cross_border,
        feature_fatf, feature_round_amount, feature_recv_velocity,
        feature_card_velocity, feature_zscore, feature_structuring
    """
    idx = df.index
    n   = len(df)

    # Locate flexible column names
    amt_col     = safe_col(df, 'Transaction Amount', 'Amount', 'Amount Paid', 'amount_usd')
    mcc_col     = safe_col(df, 'Merchant Category Code', 'MCC')
    sender_c    = safe_col(df, 'SENDER ADDRESS COUNTRY', 'Sender Country', 'From Bank')
    recv_c      = safe_col(df, 'RECEIVER ADDRESS COUNTRY', 'Receiver Country (Legacy)',
                            'Receiver Country', 'To Bank')
    card_col    = safe_col(df, 'CARD NUMBER', 'Card Number')
    recv_fn_col = safe_col(df, 'Receiver First Name', 'RECEIVER FIRST NAME')
    recv_ln_col = safe_col(df, 'Receiver Last Name',  'RECEIVER LAST NAME')

    total  = pd.Series(0, index=idx, dtype=int)
    flags  = [[] for _ in range(n)]
    bkdown = [[] for _ in range(n)]

    f_amt        = pd.Series(0.0, index=idx)
    f_mcc        = pd.Series(0,   index=idx)
    f_cross      = pd.Series(0,   index=idx)
    f_fatf       = pd.Series(0,   index=idx)
    f_round      = pd.Series(0,   index=idx)
    f_rvel       = pd.Series(0,   index=idx)
    f_cvel       = pd.Series(0,   index=idx)
    f_z          = pd.Series(0.0, index=idx)
    f_structure  = pd.Series(0,   index=idx)

    # â”€â”€ 1. Amount thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if amt_col:
        amt   = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        f_amt = amt
        pts   = pd.Series(0, index=idx)
        pts[amt > 50_000]                        = AMOUNT_SCORE['very_large']
        pts[(amt > 10_000) & (amt <= 50_000)]    = AMOUNT_SCORE['large']
        pts[(amt >  5_000) & (amt <= 10_000)]    = AMOUNT_SCORE['elevated']
        total += pts
        for i, (p, a) in enumerate(zip(pts, amt)):
            if p == AMOUNT_SCORE['very_large']:
                flags[i].append('Very Large Amount (>$50K)')
                bkdown[i].append(f'Large Amount +{p}')
            elif p == AMOUNT_SCORE['large']:
                flags[i].append('Large Amount (>$10K)')
                bkdown[i].append(f'Large Amount +{p}')
            elif p == AMOUNT_SCORE['elevated']:
                flags[i].append('Elevated Amount (>$5K)')
                bkdown[i].append(f'Elevated Amount +{p}')

    # â”€â”€ 2. Structuring / smurfing (just-below $10K) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if amt_col:
        amt        = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        is_struct  = (amt >= STRUCTURING_BAND_LOW) & (amt <= STRUCTURING_BAND_HIGH)
        f_structure = is_struct.astype(int)
        pts        = is_struct.astype(int) * STRUCTURING_SCORE
        total     += pts
        for i, (flag, p) in enumerate(zip(is_struct, pts)):
            if flag:
                flags[i].append(f'Structuring Detected ($9Kâ€“$10K band)')
                bkdown[i].append(f'Structuring +{p}')

    # â”€â”€ 3. High-risk MCC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if mcc_col:
        mcc_s  = df[mcc_col].apply(normalize_mcc)
        is_hr  = mcc_s.isin(HIGH_RISK_MCC.keys())
        f_mcc  = is_hr.astype(int)
        pts    = is_hr.astype(int) * MCC_SCORE
        total += pts
        for i, (flag, m, p) in enumerate(zip(is_hr, mcc_s, pts)):
            if flag:
                flags[i].append(f'High-Risk MCC {m} ({HIGH_RISK_MCC[m]})')
                bkdown[i].append(f'High-Risk MCC +{p}')
    else:
        mcc_s = pd.Series(['N/A'] * n, index=idx)

    # â”€â”€ 4. Cross-border + FATF â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if sender_c and recv_c:
        sc_vals  = df[sender_c].astype(str).str.upper().str.strip()
        rc_vals  = df[recv_c].astype(str).str.upper().str.strip()
        is_cross = sc_vals != rc_vals
        is_fs    = sc_vals.isin(FATF_HIGH_RISK.keys())
        is_fr    = rc_vals.isin(FATF_HIGH_RISK.keys())
        is_fatf  = is_fs | is_fr
        f_cross  = is_cross.astype(int)
        f_fatf   = is_fatf.astype(int)
        pts = (
            is_cross.astype(int) * CROSS_BORDER_SCORE
            + is_fatf.astype(int) * FATF_SCORE
            + (is_cross & is_fatf).astype(int) * FATF_CROSS_BONUS
        ).clip(upper=30)
        total += pts
        sc_list = sc_vals.tolist()
        for i, (cross, fatf, p, r, s) in enumerate(zip(is_cross, is_fatf, pts, rc_vals, sc_list)):
            if fatf:
                cname = FATF_HIGH_RISK.get(str(r), '') or FATF_HIGH_RISK.get(str(s), str(r))
                flags[i].append(f'FATF High-Risk Country ({cname})')
                bkdown[i].append(f'FATF +{p}')
            elif cross:
                flags[i].append('Cross-Border Transaction')
                bkdown[i].append(f'Cross-Border +{p}')

    # â”€â”€ 5. Round amounts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if amt_col:
        amt    = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        is_1k  = (amt % 1000 == 0) & (amt >= 5000)
        is_5h  = (amt % 500  == 0) & (amt >= 2000) & ~is_1k
        f_round = is_1k.astype(int) * 2 + is_5h.astype(int)
        pts    = is_1k.astype(int) * ROUND_1K_SCORE + is_5h.astype(int) * ROUND_500_SCORE
        total += pts
        for i, (k, h, p) in enumerate(zip(is_1k, is_5h, pts)):
            if k:
                flags[i].append('Round Amount ($1,000 multiple)')
                bkdown[i].append(f'Round Amount +{p}')
            elif h:
                flags[i].append('Round Amount ($500 multiple)')
                bkdown[i].append(f'Round Amount +{p}')

    # â”€â”€ 6. Receiver velocity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if recv_fn_col and recv_ln_col:
        rkey   = df[recv_fn_col].astype(str) + '_' + df[recv_ln_col].astype(str)
        rfreq  = rkey.map(rkey.value_counts())
        f_rvel = rfreq.fillna(0).astype(int)
        is_hrr = rfreq > RECEIVER_REPEAT_THRESHOLD
        pts    = is_hrr.astype(int) * RECEIVER_VEL_SCORE
        total += pts
        for i, (flag, p, cnt) in enumerate(zip(is_hrr, pts, rfreq)):
            if flag:
                flags[i].append(f'High Receiver Velocity ({int(cnt)}x same receiver)')
                bkdown[i].append(f'Receiver Velocity +{p}')

    # â”€â”€ 7. Card velocity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if card_col:
        cfreq  = df[card_col].map(df[card_col].value_counts())
        f_cvel = cfreq.fillna(0).astype(int)
        is_hrc = cfreq > CARD_REPEAT_THRESHOLD
        pts    = is_hrc.astype(int) * CARD_VEL_SCORE
        total += pts
        for i, (flag, p, cnt) in enumerate(zip(is_hrc, pts, cfreq)):
            if flag:
                flags[i].append(f'High Card Usage ({int(cnt)}x same card)')
                bkdown[i].append(f'Card Velocity +{p}')

    # â”€â”€ 8. Z-score statistical anomaly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if amt_col:
        amt  = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        mean = amt.mean()
        std  = amt.std()
        if std > 0:
            z    = (amt - mean) / std
            f_z  = z.round(2)
            anom = z.abs() > 2
            pts  = anom.astype(int) * ZSCORE_ANOMALY_SCORE
            total += pts
            for i, (flag, p, zv) in enumerate(zip(anom, pts, z)):
                if flag:
                    flags[i].append(f'Statistical Anomaly (Z={zv:.2f})')
                    bkdown[i].append(f'Z-Score Anomaly +{p}')

    total = total.clip(upper=100)
    level = total.apply(
        lambda s: 'HIGH' if s >= HIGH_THRESHOLD else ('MEDIUM' if s >= MEDIUM_THRESHOLD else 'LOW')
    )

    out = pd.DataFrame(index=idx)
    out['score']                 = total
    out['risk_level']            = level
    out['flags']                 = [' | '.join(f) if f else 'No flags' for f in flags]
    out['score_breakdown']       = [', '.join(b) if b else 'No risk points scored' for b in bkdown]
    out['feature_amount']        = f_amt
    out['feature_high_risk_mcc'] = f_mcc
    out['feature_cross_border']  = f_cross
    out['feature_fatf']          = f_fatf
    out['feature_round_amount']  = f_round
    out['feature_recv_velocity'] = f_rvel
    out['feature_card_velocity'] = f_cvel
    out['feature_zscore']        = f_z
    out['feature_structuring']   = f_structure
    return out


def classify(score: int) -> str:
    """Convert a raw 0-100 score to a risk-level string."""
    if score >= HIGH_THRESHOLD:
        return 'HIGH'
    if score >= MEDIUM_THRESHOLD:
        return 'MEDIUM'
    return 'LOW'
