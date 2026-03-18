"""
AML Rule Definitions â€” single source of truth for all rule constants.

Add new rules here; the engine.py imports them automatically.
"""

# â”€â”€ High-Risk Merchant Category Codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HIGH_RISK_MCC: dict[str, str] = {
    '6012': 'Financial Institution Merchandise',
    '7995': 'Gambling / Betting',
    '6211': 'Securities Dealers',
    '5993': 'Cigar / Tobacco Stores',
    '5912': 'Drug Stores / Pharmacies',
    '7801': 'Government-Licensed Casinos',
    '6051': 'Non-Financial Institutions (Crypto/Currency)',
    '6010': 'Manual Cash Disbursements',
    '4829': 'Wire Transfers / Money Orders',
    '7800': 'Government-Licensed Gambling',
    '7802': 'State Lotteries',
}

# â”€â”€ FATF High-Risk Jurisdictions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FATF_HIGH_RISK: dict[str, str] = {
    'IRN': 'Iran',        'PRK': 'North Korea', 'MMR': 'Myanmar',
    'AFG': 'Afghanistan', 'IRQ': 'Iraq',        'SYR': 'Syria',
    'YEM': 'Yemen',       'LBY': 'Libya',       'SOM': 'Somalia',
    'VEN': 'Venezuela',   'NGA': 'Nigeria',     'PAK': 'Pakistan',
    'HTI': 'Haiti',       'RUS': 'Russia',      'BLR': 'Belarus',
    'CUB': 'Cuba',
}

# â”€â”€ Structuring / smurfing thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STRUCTURING_BAND_LOW  = 9_000   # just-below-threshold band start
STRUCTURING_BAND_HIGH = 9_999
CTR_THRESHOLD         = 10_000  # Cash Transaction Reporting threshold (US)

# â”€â”€ Velocity thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
VELOCITY_1HR_HIGH  = 5    # more than N txns from same account in 1 hr  â†’ flag
VELOCITY_6HR_HIGH  = 15
VELOCITY_24HR_HIGH = 30

RECEIVER_REPEAT_THRESHOLD = 5   # same receiver first+last > N times â†’ flag
CARD_REPEAT_THRESHOLD     = 10  # same card number > N times â†’ flag

# â”€â”€ Score weights â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
AMOUNT_SCORE = {
    'very_large':  25,   # > 50,000
    'large':       20,   # > 10,000
    'elevated':    10,   # > 5,000
}
MCC_SCORE            = 20
CROSS_BORDER_SCORE   = 10
FATF_SCORE           = 15
FATF_CROSS_BONUS     = 5
ROUND_1K_SCORE       = 10
ROUND_500_SCORE      = 5
RECEIVER_VEL_SCORE   = 10
CARD_VEL_SCORE       = 5
ZSCORE_ANOMALY_SCORE = 15
STRUCTURING_SCORE    = 20   # just-below-threshold band

# â”€â”€ Score bands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HIGH_THRESHOLD   = 70   # ML-adjusted (was 61 in legacy)
MEDIUM_THRESHOLD = 40   # ML-adjusted (was 31 in legacy)

# IBM HI-Small dataset column names â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IBM_COL_MAP = {
    'timestamp':        'Timestamp',
    'from_bank':        'From Bank',
    'from_account':     'Account',
    'to_bank':          'To Bank',
    'to_account':       'Account.1',
    'amount_received':  'Amount Received',
    'receiving_currency': 'Receiving Currency',
    'amount_paid':      'Amount Paid',
    'payment_currency': 'Payment Currency',
    'payment_format':   'Payment Format',
    'is_laundering':    'Is Laundering',
}

# Feature column names produced by features.py (used by explain.py labels)
FEATURE_DISPLAY_NAMES = {
    'amount_usd':               'Transaction Amount',
    'amount_zscore':            'Z-Score Anomaly',
    'is_round_1k':              'Round Amount ($1K multiple)',
    'is_round_500':             'Round Amount ($500 multiple)',
    'is_just_below_threshold':  'Structuring (just-below $10K)',
    'is_high_risk_mcc':         'High-Risk MCC',
    'is_cross_border':          'Cross-Border Transaction',
    'is_fatf_country':          'FATF High-Risk Jurisdiction',
    'velocity_1hr':             'Account Velocity (1-hour)',
    'velocity_6hr':             'Account Velocity (6-hour)',
    'velocity_24hr':            'Account Velocity (24-hour)',
    'receiver_velocity':        'Receiver Velocity',
    'card_velocity':            'Card-Number Velocity',
    'hour_of_day':              'Hour of Day',
    'is_weekend':               'Weekend Transaction',
    'is_odd_hours':             'Odd-Hours Transaction (11pmâ€“5am)',
    'is_new_payee':             'New / First-Time Payee',
    'payment_format_encoded':   'Payment Format',
}
