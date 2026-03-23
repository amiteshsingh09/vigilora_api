import pandas as pd
from app.controllers.scoring_controller import safe_col, fmt_amount, normalize_mcc, row_to_dict, safe_val
from app.models.constants import FATF_HIGH_RISK, HIGH_RISK_MCC

def build_analysis_response(df_result: pd.DataFrame, cache_key: str) -> dict:
    def get_elevated_mean(scores):
        if scores is None or scores.empty:
            return 0.0
        elevated = scores[scores >= 40]
        return float(elevated.mean()) if not elevated.empty else float(scores.mean())

    total      = len(df_result)
    high_n     = int((df_result['Risk Level'] == 'HIGH').sum())
    med_n      = int((df_result['Risk Level'] == 'MEDIUM').sum())
    low_n      = int((df_result['Risk Level'] == 'LOW').sum())
    avg_score  = get_elevated_mean(df_result['Risk Score'])

    sc_col  = safe_col(df_result, 'SENDER ADDRESS COUNTRY', 'Sender Country')
    rc_col  = safe_col(df_result, 'RECEIVER ADDRESS COUNTRY', 'Receiver Country (Legacy)', 'Receiver Country')
    amt_col = safe_col(df_result, 'Transaction Amount', 'Amount')
    ref_col = safe_col(df_result, 'transaction_reference_number', 'Transaction Reference')
    date_col= safe_col(df_result, 'Transaction Date', 'Date')
    mcc_col = safe_col(df_result, 'Merchant Category Code', 'MCC')

    cross_count = 0
    cross_pct   = 0.0
    if sc_col and rc_col and total > 0:
        cross_count = int((df_result[sc_col].astype(str).str.upper() != df_result[rc_col].astype(str).str.upper()).sum())
        cross_pct   = round(cross_count / total * 100, 1)

    fatf_pct = 0.0
    if rc_col and total > 0:
        fatf_pct = round(df_result[rc_col].astype(str).str.upper().isin(FATF_HIGH_RISK.keys()).sum() / total * 100, 1)

    kpis = {
        "total": total, "high": high_n, "medium": med_n, "low": low_n,
        "avgScore": round(avg_score, 1), "sarCandidates": high_n,
        "customerRiskPct":    round(high_n / total * 100, 1) if total > 0 else 0,
        "transactionRiskPct": round((high_n + med_n) / total * 100, 1) if total > 0 else 0,
        "geographicRiskPct":  cross_pct,
        "fatfPct": fatf_pct,
    }

    # --- Multi-granularity Trend ---
    trend_daily = []
    trend_weekly = []
    trend_monthly = []
    default_view = "daily"

    df_copy = df_result.copy()
    if date_col:
        try:
            df_copy['_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_dated = df_copy.dropna(subset=['_date']).copy()

            if not df_dated.empty:
                date_range_days = (df_dated['_date'].max() - df_dated['_date'].min()).days

                if date_range_days >= 60:
                    default_view = "monthly"
                elif date_range_days >= 14:
                    default_view = "weekly"
                else:
                    default_view = "daily"

                grp_d = df_dated.groupby(df_dated['_date'].dt.date)
                sc_d  = grp_d['Risk Score'].apply(get_elevated_mean)
                hi_d  = grp_d['Risk Level'].apply(lambda x: (x == 'HIGH').sum())
                am_d  = grp_d[amt_col].mean() if amt_col else None
                for day, score in sc_d.items():
                    p = {"label": str(day), "score": round(float(score), 1), "highCount": int(hi_d.get(day, 0))}
                    if am_d is not None:
                        p["avgAmount"] = round(float(am_d.get(day, 0)), 2)
                    trend_daily.append(p)

                df_dated['_week'] = df_dated['_date'].dt.to_period('W').dt.start_time.dt.date
                grp_w = df_dated.groupby('_week')
                sc_w  = grp_w['Risk Score'].apply(get_elevated_mean)
                hi_w  = grp_w['Risk Level'].apply(lambda x: (x == 'HIGH').sum())
                am_w  = grp_w[amt_col].mean() if amt_col else None
                for wk, score in sc_w.items():
                    p = {"label": f"W/{str(wk)[5:]}", "score": round(float(score), 1), "highCount": int(hi_w.get(wk, 0))}
                    if am_w is not None:
                        p["avgAmount"] = round(float(am_w.get(wk, 0)), 2)
                    trend_weekly.append(p)

                df_dated['_month'] = df_dated['_date'].dt.to_period('M').dt.start_time.dt.date
                grp_m = df_dated.groupby('_month')
                sc_m  = grp_m['Risk Score'].apply(get_elevated_mean)
                hi_m  = grp_m['Risk Level'].apply(lambda x: (x == 'HIGH').sum())
                am_m  = grp_m[amt_col].mean() if amt_col else None
                for mo, score in sc_m.items():
                    p = {"label": str(mo)[:7], "score": round(float(score), 1), "highCount": int(hi_m.get(mo, 0))}
                    if am_m is not None:
                        p["avgAmount"] = round(float(am_m.get(mo, 0)), 2)
                    trend_monthly.append(p)
        except Exception:
            pass

    if not trend_daily:
        bsz = max(1, total // 8)
        for i in range(min(8, total)):
            chunk = df_copy.iloc[i*bsz:(i+1)*bsz]
            p = {"label": f"Batch {i+1}", "score": round(get_elevated_mean(chunk['Risk Score']), 1),
                 "highCount": int((chunk['Risk Level'] == 'HIGH').sum())}
            if amt_col:
                p["avgAmount"] = round(float(pd.to_numeric(chunk[amt_col], errors='coerce').mean()), 2)
            trend_daily.append(p)
        trend_weekly  = trend_daily
        trend_monthly = trend_daily

    trend = [{"day": p["label"], "score": p["score"]} for p in trend_daily]

    # --- Transactions (top 200) with Top Contributors standard keys ---
    tx_cols = [c for c in [ref_col, date_col, amt_col, sc_col, rc_col,
               'Risk Score', 'Risk Level', 'Risk Flags', 'Score Breakdown', 'Z-Score',
               safe_col(df_result, 'Merchant Name'), mcc_col,
               safe_col(df_result, 'Receiver First Name', 'RECEIVER FIRST NAME'),
               safe_col(df_result, 'Receiver Last Name', 'RECEIVER LAST NAME'),
               safe_col(df_result, 'Receiver Account')]
               if c and c in df_result.columns]

    transactions = []
    for _, row in df_result[tx_cols].head(200).iterrows():
        base_dict = row_to_dict(row)
        
        # Determine the Receiver/Merchant string
        fn_val = base_dict.get("Receiver First Name") or base_dict.get("RECEIVER FIRST NAME") or ""
        ln_val = base_dict.get("Receiver Last Name") or base_dict.get("RECEIVER LAST NAME") or ""
        full_name = f"{fn_val} {ln_val}".strip()

        merch = base_dict.get("Merchant Name") or ""
        if full_name and merch and merch != "Unknown":
            recv = f"{full_name} ({merch})"
        elif full_name:
            recv = full_name
        elif merch and merch != "Unknown":
            recv = merch
        else:
            recv = base_dict.get("Receiver Account") or "Unknown"
        
        # Add the explicit standardized keys required by Top Contributors table
        base_dict["Reference"] = safe_val(row.get(ref_col) if ref_col else "N/A")
        base_dict["Date"] = safe_val(row.get(date_col) if date_col else "N/A")
        base_dict["Receiver Account"] = safe_val(recv)
        base_dict["Amount"] = safe_val(row.get(amt_col) if amt_col else 0)
        base_dict["Risk Score"] = safe_val(row.get("Risk Score", 0))
        base_dict["Risk Level"] = safe_val(row.get("Risk Level", "LOW"))
        base_dict["Risk Flags"] = safe_val(row.get("Risk Flags", "None"))
        
        transactions.append(base_dict)

    # --- Overview: risk distribution buckets ---
    score_buckets = {}
    for v in df_result['Risk Score']:
        b = f"{(int(v)//10)*10}-{(int(v)//10)*10+10}"
        score_buckets[b] = score_buckets.get(b, 0) + 1
    histogram = [{"range": k, "count": v} for k, v in sorted(score_buckets.items())]

    risk_summary = []
    for lvl in ['HIGH', 'MEDIUM', 'LOW']:
        sub = df_result[df_result['Risk Level'] == lvl]['Risk Score']
        if not sub.empty:
            risk_summary.append({
                "level": lvl, "count": int(len(sub)),
                "avg": round(float(sub.mean()), 1),
                "max": int(sub.max()), "min": int(sub.min()),
            })

    def make_alert_cards(rows, max_n=40):
        cards = []
        for _, row in rows.head(max_n).iterrows():
            ref = row[ref_col] if ref_col else str(row.name)
            amt = fmt_amount(row[amt_col]) if amt_col else 'N/A'
            flags = row['Risk Flags']
            flag_count = flags.count('|') + 1 if flags != 'No flags' else 0
            cards.append({
                "ref": str(ref), "amount": amt,
                "score": int(row['Risk Score']), "level": str(row['Risk Level']),
                "flags": str(flags), "breakdown": str(row['Score Breakdown']),
                "zScore": round(float(row.get('Z-Score', 0)), 2),
                "flagCount": flag_count,
            })
        return cards

    high_alerts   = make_alert_cards(df_result[df_result['Risk Level'] == 'HIGH'])
    medium_alerts = make_alert_cards(df_result[df_result['Risk Level'] == 'MEDIUM'])

    anomalies = []
    anom_df = df_result[df_result['Z-Score'].abs() > 2].sort_values('Z-Score', key=abs, ascending=False).head(20)
    for _, row in anom_df.iterrows():
        ref = row[ref_col] if ref_col else str(row.name)
        anomalies.append({
            "ref": str(ref), "zScore": round(float(row['Z-Score']), 2),
            "score": int(row['Risk Score']), "flags": str(row['Risk Flags'])[:80],
        })

    merchant_data = []
    if mcc_col:
        mcc_norm = df_result[mcc_col].apply(normalize_mcc)
        top_mcc  = mcc_norm.value_counts().head(12).reset_index()
        top_mcc.columns = ['mcc', 'count']
        for _, row in top_mcc.iterrows():
            sub = df_result[mcc_norm == row['mcc']]
            merchant_data.append({
                "mcc": str(row['mcc']),
                "description": HIGH_RISK_MCC.get(str(row['mcc']), 'Standard Merchant'),
                "count": int(row['count']),
                "isHighRisk": str(row['mcc']) in HIGH_RISK_MCC,
                "avgScore": round(float(sub['Risk Score'].mean()), 1),
            })

    fatf_transactions = []
    if rc_col:
        fatf_df = df_result[df_result[rc_col].astype(str).str.upper().isin(FATF_HIGH_RISK.keys())].head(30)
        for _, row in fatf_df.iterrows():
            ctry = str(row[rc_col]).upper()
            fatf_transactions.append({
                "country": ctry, "countryName": FATF_HIGH_RISK.get(ctry, ctry),
                "amount": fmt_amount(row[amt_col]) if amt_col else 'N/A',
                "score": int(row['Risk Score']), "level": str(row['Risk Level']),
                "ref": str(row[ref_col]) if ref_col else 'N/A',
            })

    country_data = []
    if rc_col:
        top_c = df_result[rc_col].value_counts().head(15).reset_index()
        top_c.columns = ['country', 'count']
        for _, row in top_c.iterrows():
            country_data.append({
                "country": str(row['country']), "count": int(row['count']),
                "isFatf": str(row['country']).upper() in FATF_HIGH_RISK,
            })

    network = {"nodes": [], "edges": []}
    if ref_col:
        sn_col = safe_col(df_result, 'Sender Account') or safe_col(df_result, 'Originator Account')
        rn_col = safe_col(df_result, 'Receiver Account') or safe_col(df_result, 'Beneficiary Account')
        if sn_col and rn_col:
            edges_df = df_result.groupby([sn_col, rn_col])['Risk Score'].max().reset_index()
            edges_df = edges_df.sort_values('Risk Score', ascending=False).head(25)
            node_set = set(edges_df[sn_col]).union(set(edges_df[rn_col]))
            node_map = {}
            for n in node_set:
                sn_max = df_result[df_result[sn_col] == n]['Risk Score'].max() if n in df_result[sn_col].values else 0
                rn_max = df_result[df_result[rn_col] == n]['Risk Score'].max() if n in df_result[rn_col].values else 0
                max_score = max(sn_max, rn_max)
                risk_level = 'high' if max_score >= 61 else 'medium' if max_score >= 31 else 'low'
                node_map[n] = {"id": str(n), "label": str(n)[:8], "risk": risk_level}
            network["nodes"] = list(node_map.values())
            network["edges"] = [{"from": str(r[sn_col]), "to": str(r[rn_col])} for _, r in edges_df.iterrows()]

    sar_candidates = []
    high_sar = df_result[df_result['Risk Level'] == 'HIGH'].copy()
    for i, (idx_val, row) in enumerate(high_sar.iterrows()):
        ref = row[ref_col] if ref_col else f"Row {idx_val}"
        flags_short = str(row['Risk Flags'])[:60] + ('...' if len(str(row['Risk Flags'])) > 60 else '')
        sar_candidates.append({
            "index": i, "rowIndex": int(idx_val), "ref": str(ref),
            "score": int(row['Risk Score']), "zScore": round(float(row.get('Z-Score', 0)), 2),
            "flagCount": str(row['Risk Flags']).count('|') + 1 if row['Risk Flags'] != 'No flags' else 0,
            "flagsShort": flags_short, "breakdown": str(row['Score Breakdown']), "allFields": row_to_dict(row),
        })

    alerts = []
    if fatf_transactions: alerts.append({"id": "fatf", "icon": "globe", "title": f"{len(fatf_transactions)} FATF high-risk jurisdiction transactions detected", "severity": "danger", "timestamp": "Just now"})
    if high_n > 0: alerts.append({"id": "sar", "icon": "clipboard", "title": f"{high_n} SAR candidate(s) pending MLRO review", "severity": "warning", "timestamp": "Just now"})
    if anomalies: alerts.append({"id": "anomaly", "icon": "bar-chart", "title": f"{len(anomalies)} statistical anomalies detected (|Z|>2)", "severity": "warning", "timestamp": "Just now"})
    alerts.append({"id": "kyc", "icon": "alert", "title": "KYC Verification required for HIGH risk account holders", "severity": "danger", "timestamp": "Just now"})

    fn_col = safe_col(df_result, 'SENDER FIRST NAME', 'Sender First Name')
    ln_col = safe_col(df_result, 'SENDER LAST NAME', 'Sender Last Name')
    customers_dict = {}
    for _, row in df_result.iterrows():
        fname = str(row[fn_col]).strip() if pd.notna(row.get(fn_col)) else ""
        lname = str(row[ln_col]).strip() if pd.notna(row.get(ln_col)) else ""
        full_name = f"{fname} {lname}".strip()
        ref = str(row.get(ref_col, ''))
        cust_id = full_name if full_name else ref
        if not cust_id: continue
        score = float(row.get('Risk Score', 0))
        h_rc = str(row.get(rc_col, '')).upper()
        h_sc = str(row.get(sc_col, '')).upper()
        is_fatf = h_rc in FATF_HIGH_RISK or h_sc in FATF_HIGH_RISK
        tx_date = str(row.get(date_col, '2026-02-14'))[:10]
        flags = str(row.get('Risk Flags', 'None'))
        mcc = str(row.get(mcc_col, ''))
        
        if cust_id not in customers_dict:
            addr_line = str(row.get('SENDER ADDRESS LINE', ''))
            city = str(row.get('SENDER ADDRESS CITY', ''))
            state = str(row.get('SENDER ADDRESS STATE', ''))
            zip_code = str(row.get('SENDER ADDRESS POSTAL', ''))
            card = str(row.get('CARD NUMBER', 'XXXX'))
            acct_type = str(row.get('CARD PRODUCT TYPE', 'Checking'))
            employer = str(row.get('Merchant Name', 'MegaGrocer Inc'))
            customers_dict[cust_id] = {
                "id": f"cust-{ref if ref else hash(cust_id)}",
                "customerNumber": ref if ref else f"CUST-{abs(hash(cust_id))%1000000}",
                "name": full_name if full_name else "Unknown Customer",
                "dob": "1980-01-01", "joiningDate": "2020-01-01",
                "ssn": f"XXX-XX-{ref[-4:] if ref else str(hash(cust_id))[-4:]}",
                "phone": f"1-800-555-{ref[-4:] if ref else str(hash(cust_id))[-4:]}",
                "email": f"{full_name.replace(' ', '').lower()}@example.com" if full_name else "customer@example.com",
                "riskScore": score, "hasFatf": is_fatf,
                "addresses": [{"label": "Current", "line1": addr_line, "line2": "", "city": city, "state": state, "zip": zip_code, "country": h_sc}],
                "accounts": [{"number": card, "type": acct_type, "currency": str(row.get('Transaction Currency Code', 'USD')), "status": "Active", "opened": "2020-01-01"}],
                "work": {"employer": employer, "industry": mcc, "occupation": "Private Client", "since": "2020", "annualIncome": "Confidential"},
                "cases": [], "alertHistory": []
            }
        cust = customers_dict[cust_id]
        if score > cust["riskScore"]: cust["riskScore"] = score
        if is_fatf: cust["hasFatf"] = True
        if score >= 40:
            cust["alertHistory"].append({"ref": ref, "date": tx_date, "decision": "Under Review", "score": score, "flags": flags})
            
    customers = []
    for cd in customers_dict.values():
        if cd["riskScore"] >= 80:
            c_ref = cd["customerNumber"][-4:]
            cd["cases"].append({"id": f"CASE-2026-{c_ref}", "type": "AML Investigation", "status": "Open", "opened": "2026-02-14", "description": "Triggered by high risk rule engine scoring."})
        customers.append(cd)

    return {
        "cacheKey":        cache_key,
        "kpis":            kpis,
        "trend":           trend,
        "transactions":    transactions,
        "alerts":          alerts,
        "countryData":     country_data,
        "columns":         tx_cols,
        "histogram":       histogram,
        "riskSummary":     risk_summary,
        "highAlerts":      high_alerts,
        "mediumAlerts":    medium_alerts,
        "anomalies":       anomalies,
        "merchantData":    merchant_data,
        "fatfTransactions": fatf_transactions,
        "sarCandidates":   sar_candidates,
        "crossBorderCount": cross_count,
        "trendDaily":      trend_daily,
        "trendWeekly":     trend_weekly,
        "trendMonthly":    trend_monthly,
        "defaultView":     default_view,
        "network":         network,
        "customers":       customers,
    }
