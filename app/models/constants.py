import pandas as pd

HIGH_RISK_MCC = {
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

FATF_HIGH_RISK = {
    'IRN': 'Iran',        'PRK': 'North Korea', 'MMR': 'Myanmar',
    'AFG': 'Afghanistan', 'IRQ': 'Iraq',        'SYR': 'Syria',
    'YEM': 'Yemen',       'LBY': 'Libya',       'SOM': 'Somalia',
    'VEN': 'Venezuela',   'NGA': 'Nigeria',     'PAK': 'Pakistan',
    'HTI': 'Haiti',       'RUS': 'Russia',      'BLR': 'Belarus',
    'CUB': 'Cuba',
}

# In-memory store (per request - stateless)
_result_cache: dict[str, pd.DataFrame] = {}
