{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
\
\
def add_basic_features(combined: pd.DataFrame) -> pd.DataFrame:\
    """\
    Add basic lag and simple volatility features to the combined returns DataFrame.\
    Operates in-place and also returns the DataFrame for convenience.\
    """\
    # Lagged returns (1-minute)\
    combined["lag_ES"] = combined["ret_ES"].shift(1)\
    combined["lag_USDJPY"] = combined["ret_USDJPY"].shift(1)\
    combined["lag_USDCAD"] = combined["ret_USDCAD"].shift(1)\
    combined["lag_VIX"] = combined["ret_VIX"].shift(1)\
    combined["lag_GVZ"] = combined["ret_GVZ"].shift(1)\
\
    # Simple ES volatility feature\
    combined["vol_ES_60"] = combined["ret_ES"].rolling(60).std()\
\
    return combined\
\
\
def add_advanced_features(combined: pd.DataFrame) -> pd.DataFrame:\
    """\
    Add the richer volatility, correlation, momentum, and regime features.\
    """\
    # Ensure basic features exist\
    if "lag_ES" not in combined.columns:\
        combined = add_basic_features(combined)\
\
    # 1. Volatility features for multiple assets\
    vol_windows = [60, 120, 300]\
    for w in vol_windows:\
        combined[f"vol_ES_\{w\}"] = combined["ret_ES"].rolling(w).std()\
        combined[f"vol_USDJPY_\{w\}"] = combined["ret_USDJPY"].rolling(w).std()\
        combined[f"vol_USDCAD_\{w\}"] = combined["ret_USDCAD"].rolling(w).std()\
        combined[f"vol_VIX_\{w\}"] = combined["ret_VIX"].rolling(w).std()\
        combined[f"vol_GVZ_\{w\}"] = combined["ret_GVZ"].rolling(w).std()\
\
    # 2. Rolling correlations with ES\
    corr_windows = [60, 300]\
    for w in corr_windows:\
        combined[f"corr_ES_VIX_\{w\}"] = combined["ret_ES"].rolling(w).corr(\
            combined["ret_VIX"]\
        )\
        combined[f"corr_ES_USDJPY_\{w\}"] = combined["ret_ES"].rolling(w).corr(\
            combined["ret_USDJPY"]\
        )\
        combined[f"corr_ES_USDCAD_\{w\}"] = combined["ret_ES"].rolling(w).corr(\
            combined["ret_USDCAD"]\
        )\
\
    # 3. Momentum features for ES\
    mom_windows = [5, 10, 30]\
    for w in mom_windows:\
        combined[f"mom_ES_\{w\}"] = combined["ret_ES"].rolling(w).sum()\
\
    # 4. Volatility regime based on ES 300-min volatility\
    regime_vol_col = "vol_ES_300"\
    vol_threshold = combined[regime_vol_col].quantile(0.7)\
    combined["regime_high_vol"] = (\
        combined[regime_vol_col] > vol_threshold\
    ).astype(int)\
\
    return combined\
}