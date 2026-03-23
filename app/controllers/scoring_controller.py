import pandas as pd
import numpy as np

def safe_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

def fmt_amount(val):
    try:
        return f"${float(val):,.2f}"
    except (ValueError, TypeError):
        return str(val) if val is not None else 'N/A'

def normalize_mcc(val):
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val).strip()

def load_excel(f) -> pd.DataFrame:
    try:
        df = pd.read_excel(f, engine='openpyxl')
    except Exception:
        df = pd.read_excel(f)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).replace({'nan': '', 'None': ''})
    return df

def safe_val(v):
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)
    if isinstance(v, (pd.NaT.__class__,)):
        return None
    if not isinstance(v, (bool, int, float, str, type(None))):
        return str(v)
    return v

def row_to_dict(row):
    return {k: safe_val(v) for k, v in row.items()}

def merge_results(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    r = df.copy()
    r['Risk Score']      = scores['score'].values
    r['Risk Level']      = scores['risk_level'].values
    r['Risk Flags']      = scores['flags'].values
    r['Score Breakdown'] = scores['score_breakdown'].values
    for c in [col for col in scores.columns if col.startswith('feature_')]:
        r[c] = scores[c].values
    r['Z-Score'] = scores['feature_zscore'].values
    return r

def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    from app.models.constants import HIGH_RISK_MCC, FATF_HIGH_RISK
    idx = df.index
    n   = len(df)

    amt_col     = safe_col(df, 'Transaction Amount', 'Amount')
    mcc_col     = safe_col(df, 'Merchant Category Code', 'MCC')
    sender_c    = safe_col(df, 'SENDER ADDRESS COUNTRY', 'Sender Country')
    recv_c      = safe_col(df, 'RECEIVER ADDRESS COUNTRY', 'Receiver Country (Legacy)', 'Receiver Country', 'RECEIVER ADDRESS COUNTRY')
    card_col    = safe_col(df, 'CARD NUMBER', 'Card Number')
    recv_fn_col = safe_col(df, 'Receiver First Name', 'RECEIVER FIRST NAME')
    recv_ln_col = safe_col(df, 'Receiver Last Name', 'RECEIVER LAST NAME')

    total  = pd.Series(0, index=idx, dtype=int)
    flags  = [[] for _ in range(n)]
    bkdown = [[] for _ in range(n)]

    f_amt   = pd.Series(0.0, index=idx)
    f_mcc   = pd.Series(0, index=idx)
    f_cross = pd.Series(0, index=idx)
    f_fatf  = pd.Series(0, index=idx)
    f_round = pd.Series(0, index=idx)
    f_rvel  = pd.Series(0, index=idx)
    f_cvel  = pd.Series(0, index=idx)
    f_z     = pd.Series(0.0, index=idx)

    if amt_col:
        amt   = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        f_amt = amt
        pts   = pd.Series(0, index=idx)
        pts[amt > 50000] = 25
        pts[(amt > 10000) & (amt <= 50000)] = 20
        pts[(amt > 5000)  & (amt <= 10000)] = 10
        total += pts
        for i, (p, a) in enumerate(zip(pts, amt)):
            if p == 25:   flags[i].append('Very Large Amount (>$50K)');  bkdown[i].append(f'Large Amount +{p}')
            elif p == 20: flags[i].append('Large Amount (>$10K)');       bkdown[i].append(f'Large Amount +{p}')
            elif p == 10: flags[i].append('Elevated Amount (>$5K)');     bkdown[i].append(f'Elevated Amount +{p}')

    if mcc_col:
        mcc_s  = df[mcc_col].apply(normalize_mcc)
        is_hr  = mcc_s.isin(HIGH_RISK_MCC.keys())
        f_mcc  = is_hr.astype(int)
        pts    = is_hr.astype(int) * 20
        total += pts
        for i, (flag, m, p) in enumerate(zip(is_hr, mcc_s, pts)):
            if flag:
                flags[i].append(f'High-Risk MCC {m} ({HIGH_RISK_MCC[m]})')
                bkdown[i].append(f'High-Risk MCC +{p}')
    else:
        mcc_s = pd.Series(['N/A'] * n, index=idx)

    if sender_c and recv_c:
        sc_vals  = df[sender_c].astype(str).str.upper().str.strip()
        rc_vals  = df[recv_c].astype(str).str.upper().str.strip()
        is_cross = sc_vals != rc_vals
        is_fs    = sc_vals.isin(FATF_HIGH_RISK.keys())
        is_fr    = rc_vals.isin(FATF_HIGH_RISK.keys())
        is_fatf  = is_fs | is_fr
        f_cross  = is_cross.astype(int)
        f_fatf   = is_fatf.astype(int)
        pts = (is_cross.astype(int)*10 + is_fatf.astype(int)*15 + (is_cross & is_fatf).astype(int)*5).clip(upper=30)
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

    if amt_col:
        amt    = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        is_1k  = (amt % 1000 == 0) & (amt >= 5000)
        is_5h  = (amt % 500  == 0) & (amt >= 2000) & ~is_1k
        f_round = is_1k.astype(int) * 2 + is_5h.astype(int)
        pts    = is_1k.astype(int) * 10 + is_5h.astype(int) * 5
        total += pts
        for i, (k, h, p) in enumerate(zip(is_1k, is_5h, pts)):
            if k:   flags[i].append('Round Amount ($1,000 multiple)'); bkdown[i].append(f'Round Amount +{p}')
            elif h: flags[i].append('Round Amount ($500 multiple)');   bkdown[i].append(f'Round Amount +{p}')

    if recv_fn_col and recv_ln_col:
        rkey  = df[recv_fn_col].astype(str) + '_' + df[recv_ln_col].astype(str)
        rfreq = rkey.map(rkey.value_counts())
        f_rvel = rfreq.fillna(0).astype(int)
        is_hrr = rfreq > 5
        pts    = is_hrr.astype(int) * 10
        total += pts
        for i, (flag, p, cnt) in enumerate(zip(is_hrr, pts, rfreq)):
            if flag:
                flags[i].append(f'High Receiver Velocity ({int(cnt)}x same receiver)')
                bkdown[i].append(f'Receiver Velocity +{p}')

    if card_col:
        cfreq  = df[card_col].map(df[card_col].value_counts())
        f_cvel = cfreq.fillna(0).astype(int)
        is_hrc = cfreq > 10
        pts    = is_hrc.astype(int) * 5
        total += pts
        for i, (flag, p, cnt) in enumerate(zip(is_hrc, pts, cfreq)):
            if flag:
                flags[i].append(f'High Card Usage ({int(cnt)}x same card)')
                bkdown[i].append(f'Card Velocity +{p}')

    if amt_col:
        amt  = pd.to_numeric(df[amt_col], errors='coerce').fillna(0)
        mean = amt.mean(); std = amt.std()
        if std > 0:
            z   = (amt - mean) / std
            f_z = z.round(2)
            anom = z.abs() > 2
            pts  = anom.astype(int) * 15
            total += pts
            for i, (flag, p, zv) in enumerate(zip(anom, pts, z)):
                if flag:
                    flags[i].append(f'Statistical Anomaly (Z={zv:.2f})')
                    bkdown[i].append(f'Z-Score Anomaly +{p}')

    total = total.clip(upper=100)
    level = total.apply(lambda s: 'HIGH' if s >= 61 else ('MEDIUM' if s >= 31 else 'LOW'))

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
    return out
