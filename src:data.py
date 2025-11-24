{\rtf1\ansi\ansicpg1252\cocoartf2818
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import pandas as pd\
\
# You can override this from a notebook or script if needed\
DATA_DIR = "/content/drive/MyDrive/project_data"\
\
PRICE_DIRS = \{\
    'crypto':  f"\{DATA_DIR\}/crypto",\
    'equity':  f"\{DATA_DIR\}/equity",\
    'etf':     f"\{DATA_DIR\}/etf",\
    'futures': f"\{DATA_DIR\}/futures",\
    'fx':      f"\{DATA_DIR\}/fx",\
    'index':   f"\{DATA_DIR\}/index",\
\}\
\
\
def load_metadata():\
    """\
    Load symbol metadata and drop junk columns if present.\
    """\
    meta_path = os.path.join(DATA_DIR, "symbol_metadata.csv")\
    meta = pd.read_csv(meta_path)\
    if "Unnamed: 0" in meta.columns:\
        meta = meta.drop(columns=["Unnamed: 0"])\
    return meta\
\
\
def load_asset(ticker: str, meta: pd.DataFrame) -> pd.DataFrame:\
    """\
    Load a single asset (by ticker) using the metadata to find its asset class.\
    Returns a DataFrame with a clean 'timestamp' column and a 'ret' column.\
    """\
    row = meta[meta["Ticker"] == ticker]\
    if row.empty:\
        raise ValueError(f"Ticker \{ticker\} not found in metadata")\
\
    asset_class = row["PrimaryAssetClass"].values[0].lower()\
    folder = PRICE_DIRS.get(asset_class)\
    if folder is None:\
        raise ValueError(f"No folder mapping for asset class: \{asset_class\}")\
\
    parquet_path = os.path.join(folder, f"\{ticker\}.parquet")\
    csv_path = os.path.join(folder, f"\{ticker\}.csv")\
\
    if os.path.exists(parquet_path):\
        df = pd.read_parquet(parquet_path)\
    elif os.path.exists(csv_path):\
        df = pd.read_csv(csv_path)\
    else:\
        raise FileNotFoundError(f"No .parquet or .csv found for \{ticker\} in \{folder\}")\
\
    # Standardize timestamps\
    if "ts" in df.columns:\
        df = df.rename(columns=\{"ts": "timestamp"\})\
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)\
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)\
\
    df = df.sort_values("timestamp").reset_index(drop=True)\
\
    # Per-minute returns\
    df["ret"] = df["close"].pct_change()\
\
    return df\
\
\
def build_combined_returns(tickers, meta: pd.DataFrame) -> pd.DataFrame:\
    """\
    Given a list of tickers, load each and merge their returns on timestamp.\
    Returns a DataFrame with one 'timestamp' column and 'ret_<ticker>' columns.\
    """\
    dfs = \{\}\
    for t in tickers:\
        df_t = load_asset(t, meta)[["timestamp", "ret"]].rename(\
            columns=\{"ret": f"ret_\{t\}"\}\
        )\
        dfs[t] = df_t\
\
    # Merge on timestamp\
    combined = dfs[tickers[0]]\
    for t in tickers[1:]:\
        combined = combined.merge(dfs[t], on="timestamp", how="inner")\
\
    return combined\
}