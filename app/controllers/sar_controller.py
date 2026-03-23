from datetime import datetime
from app.controllers.scoring_controller import fmt_amount, normalize_mcc
from app.models.constants import HIGH_RISK_MCC, FATF_HIGH_RISK

def generate_sar_text(row: dict, total: int, avg_score: float) -> str:
    ref     = row.get('transaction_reference_number') or row.get('Transaction Reference') or 'N/A'
    amt_raw = None
    for k in row:
        if 'amount' in k.lower():
            amt_raw = row[k]; break
    amount  = fmt_amount(amt_raw)
    score   = row.get('Risk Score', 'N/A')
    level   = row.get('Risk Level', 'N/A')
    flags   = row.get('Risk Flags', 'N/A')
    bkdown  = row.get('Score Breakdown', 'N/A')
    z       = row.get('Z-Score', 0)
    date    = row.get('Transaction Date') or row.get('Date') or 'N/A'
    sender_country   = row.get('SENDER ADDRESS COUNTRY') or row.get('Sender Country') or 'N/A'
    receiver_country = row.get('RECEIVER ADDRESS COUNTRY') or row.get('Receiver Country') or row.get('Receiver Country (Legacy)') or 'N/A'
    mcc     = row.get('Merchant Category Code') or row.get('MCC') or 'N/A'
    merchant= row.get('Merchant Name') or 'N/A'
    card    = row.get('CARD NUMBER') or row.get('Card Number') or 'XXXX-XXXX-XXXX-XXXX'
    recv_fn = row.get('Receiver First Name') or row.get('RECEIVER FIRST NAME') or ''
    recv_ln = row.get('Receiver Last Name')  or row.get('RECEIVER LAST NAME')  or ''
    receiver_name = f"{recv_fn} {recv_ln}".strip() or 'N/A'

    now  = datetime.now()
    text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               VIGILORA AML — SUSPICIOUS ACTIVITY REPORT (SAR) DRAFT         ║
║                    ** NOT FOR FILING WITHOUT MLRO APPROVAL **                ║
╚══════════════════════════════════════════════════════════════════════════════╝

REPORT METADATA
───────────────────────────────────────────────────────────────────────────────
Report Generated     : {now.strftime('%Y-%m-%d %H:%M:%S')} UTC
SAR Reference        : SAR-{now.strftime('%Y%m%d')}-{str(ref)[:10]}
Status               : DRAFT — PENDING MLRO REVIEW
Reporting Institution: [FILL IN INSTITUTION NAME]
MLRO Name            : [FILL IN MLRO NAME]
Jurisdiction         : [FILL IN JURISDICTION]

SUBJECT TRANSACTION DETAILS
───────────────────────────────────────────────────────────────────────────────
Transaction Reference: {ref}
Transaction Date/Time: {date}
Transaction Amount   : {amount}
Merchant Name        : {merchant}
Merchant Category    : {mcc}
Card Number (masked) : {str(card)[:4]}XXXXXXXX{str(card)[-4:] if len(str(card)) >= 4 else '****'}
Sender Country       : {sender_country}
Receiver Country     : {receiver_country}
Receiver Name        : {receiver_name}

AI RISK ASSESSMENT
───────────────────────────────────────────────────────────────────────────────
AML Risk Score       : {score}/100
Risk Classification  : {level}
Z-Score Anomaly      : {float(z):.2f}σ  {'[ANOMALOUS — |Z|>2]' if abs(float(z)) > 2 else '[Within normal range]'}
Portfolio Context    : Batch avg score {avg_score:.1f}/100 across {total:,} transactions

Flags Triggered:
{chr(10).join(f'  → {f}' for f in str(flags).split(' | ')) if flags != 'No flags' else '  (None)'}

Score Breakdown:
{chr(10).join(f'  • {b}' for b in str(bkdown).split(', ')) if bkdown != 'No risk points scored' else '  (No points scored)'}

NARRATIVE
───────────────────────────────────────────────────────────────────────────────
On {date}, a transaction of {amount} (Reference: {ref}) was identified by the
Vigilora AML AI scoring engine as HIGH RISK with a score of {score}/100.

The transaction originated from {sender_country} and was directed to {receiver_country}.
{'This transaction involves a FATF high-risk jurisdiction, indicating elevated money laundering risk.' if receiver_country in FATF_HIGH_RISK else 'This is a cross-border transaction subject to enhanced due diligence.'}

{'The transaction amount exhibits a statistical anomaly (Z-Score: ' + str(float(z))[:5] + 'σ), significantly deviating from the portfolio average, indicating possible structuring or unusual activity.' if abs(float(z)) > 2 else ''}

The merchant category code {mcc} {'(' + HIGH_RISK_MCC.get(normalize_mcc(mcc), '') + ') is classified as high-risk under AML typologies.' if normalize_mcc(mcc) in HIGH_RISK_MCC else 'was reviewed as part of the overall risk assessment.'}

Based on the foregoing indicators, this transaction is recommended for enhanced
due diligence review and formal SAR filing consideration.

REQUIRED ACTIONS
───────────────────────────────────────────────────────────────────────────────
  [ ] MLRO to review and validate all risk indicators above
  [ ] Verify customer identity and source of funds
  [ ] Check sanctions and PEP screening results
  [ ] Determine whether formal SAR filing is required
  [ ] If filing: Submit to regulatory authority within statutory deadline
  [ ] Document decision and rationale in case management system

DISCLAIMER
───────────────────────────────────────────────────────────────────────────────
This SAR draft is auto-generated by the Vigilora AI AML system and must be
reviewed, validated, and approved by an authorized Money Laundering Reporting
Officer (MLRO) before filing. Do not share without MLRO authorization.

═════════════════════════ END OF SAR DRAFT ═════════════════════════
""".strip()
    return text
