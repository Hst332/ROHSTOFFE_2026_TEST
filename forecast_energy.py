#!/usr/bin/env python3
"""
CODE A – MARKET FORECAST
Energy, Commodities & Indices

Ruhig. Robust. Professionell.
Gas = ML
Oil = Rule-based
Other Assets = ML
One TXT output
"""

# =======================
# IMPORTS
# =======================
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# =======================
# CONFIG
# =======================
START_DATE_GAS = "2014-01-01"
START_DATE_OIL = "2015-01-01"
import os
OUT_TXT = os.path.join(os.getcwd(), "forecast_output.txt")


GAS_SYMBOL = "NG=F"
SYMBOL_BRENT = "BZ=F"
SYMBOL_WTI = "CL=F"

UP_THRESHOLD = 0.60
DOWN_THRESHOLD = 0.40

# =======================
# -------- GAS ----------
# =======================
def load_gas_prices():
    df = yf.download(GAS_SYMBOL, start=START_DATE_GAS, auto_adjust=True, progress=False)
    df = df[["Close"]].rename(columns={"Close": "Gas_Close"})
    df.dropna(inplace=True)
    return df

def load_eia_storage():
    try:
        s = pd.read_csv("eia_storage.csv", parse_dates=["Date"])
        s.sort_values("Date", inplace=True)
        return s
    except Exception:
        return None

def build_gas_features(price_df, storage_df):
    df = price_df.copy()
    df["ret"] = df["Gas_Close"].pct_change()
    df["trend_5"] = df["Gas_Close"].pct_change(5)
    df["trend_20"] = df["Gas_Close"].pct_change(20)
    df["vol_10"] = df["ret"].rolling(10).std()
    df["Target"] = (df["ret"].shift(-1) > 0).astype(int)

    if storage_df is not None:
        storage_df["surprise"] = storage_df["Storage"] - storage_df["FiveYearAvg"]
        storage_df["surprise_z"] = (storage_df["surprise"] - storage_df["surprise"].rolling(52).mean()) / storage_df["surprise"].rolling(52).std()
        storage_df = storage_df[["Date", "surprise_z"]]
        df = df.merge(storage_df, left_index=True, right_on="Date", how="left")
        df["surprise_z"].ffill(inplace=True)
        df.set_index("Date", inplace=True)
    else:
        df["surprise_z"] = 0.0

    df.dropna(inplace=True)
    return df

def train_gas_model(df):
    features = ["trend_5","trend_20","vol_10","surprise_z"]
    X = df[features]
    y = df["Target"]

    tscv = TimeSeriesSplit(n_splits=5)
    acc = []

    for tr, te in tscv.split(X):
        if len(te) == 0:
            continue
        m = LogisticRegression(max_iter=200)
        m.fit(X.iloc[tr], y.iloc[tr])
        acc.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    last = df.iloc[-1]
    prob_up = model.predict_proba(last[features].values.reshape(1, -1))[0][1]
    signal = "UP" if prob_up >= UP_THRESHOLD else "DOWN" if prob_up <= DOWN_THRESHOLD else "NO_TRADE"

    return {
        "name": "NATURAL GAS",
        "date": last.name.date().isoformat(),
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "signal": signal,
        "cv_mean": float(np.mean(acc)),
        "cv_std": float(np.std(acc)),
        "close": float(last["Gas_Close"].iloc[0])
    }

# =======================
# -------- OIL ----------
# =======================
def load_oil_prices():
    brent = yf.download(SYMBOL_BRENT, start=START_DATE_OIL, progress=False)
    wti = yf.download(SYMBOL_WTI, start=START_DATE_OIL, progress=False)
    df = pd.DataFrame(index=brent.index)
    df["Brent_Close"] = brent["Close"]
    df["WTI_Close"] = wti["Close"]
    df.dropna(inplace=True)
    return df

def build_oil_signal(df):
    df = df.copy()
    df["Brent_Trend"] = df["Brent_Close"] > df["Brent_Close"].rolling(20).mean()
    df["WTI_Trend"] = df["WTI_Close"] > df["WTI_Close"].rolling(20).mean()
    df["Spread"] = df["Brent_Close"] - df["WTI_Close"]
    df["Spread_Z"] = (df["Spread"] - df["Spread"].rolling(60).mean()) / df["Spread"].rolling(60).std()
    df.dropna(inplace=True)
    last = df.iloc[-1]

    prob_up = 0.50
    if last["Brent_Trend"] and last["WTI_Trend"]:
        prob_up += 0.07
    if last["Spread_Z"] > 0.5:
        prob_up += 0.03
    elif last["Spread_Z"] < -0.5:
        prob_up -= 0.03
    prob_up = max(0.0, min(1.0, prob_up))

    if prob_up >= 0.57:
        signal = "UP"
    elif prob_up <= 0.43:
        signal = "DOWN"
    else:
        signal = "NO_TRADE"

    return {
        "date": last.name.date().isoformat(),
        "brent": float(last["Brent_Close"]),
        "wti": float(last["WTI_Close"]),
        "spread": float(last["Spread"]),
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "signal": signal,
    }

# =======================
# -------- GENERIC ML ----------
# =======================
def build_trend_vol_features(df, price_col="Close"):
    df = df.copy()
    df["ret"] = df[price_col].pct_change()
    df["trend_5"] = df[price_col].pct_change(5)
    df["trend_20"] = df[price_col].pct_change(20)
    df["vol_10"] = df["ret"].rolling(10).std()
    return df

def train_ml_model(df, price_col="Close", up_threshold=0.57, down_threshold=0.43, min_rows=30):
    df = build_trend_vol_features(df, price_col=price_col)
    df["Target"] = (df[price_col].shift(-1) > df[price_col]).astype(int)
    features = [c for c in df.columns if "trend" in c or "vol" in c]

    df = df.dropna()
    if len(df) < min_rows or df.empty:
        last_date = df.index[-1].date().isoformat() if len(df) > 0 else datetime.utcnow().date().isoformat()
        last_close = float(df[price_col].iloc[-1]) if len(df) > 0 else 0.0
        return {
            "date": last_date,
            "prob_up": 0.5,
            "prob_down": 0.5,
            "signal": "NO_TRADE",
            "cv_mean": 0.0,
            "cv_std": 0.0,
            "close": last_close
        }

    X = df[features]
    y = df["Target"]
    if X.empty or y.empty:
        last_date = df.index[-1].date().isoformat()
        last_close = float(df[price_col].iloc[-1])
        return {
            "date": last_date,
            "prob_up": 0.5,
            "prob_down": 0.5,
            "signal": "NO_TRADE",
            "cv_mean": 0.0,
            "cv_std": 0.0,
            "close": last_close
        }

    tscv = TimeSeriesSplit(n_splits=5)
    acc = []
    for tr, te in tscv.split(X):
        if len(te) == 0:
            continue
        m = LogisticRegression(max_iter=200)
        m.fit(X.iloc[tr], y.iloc[tr])
        acc.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    last = df.iloc[-1]
    prob_up = model.predict_proba(last[features].values.reshape(1, -1))[0][1]
    signal = "UP" if prob_up >= up_threshold else "DOWN" if prob_up <= down_threshold else "NO_TRADE"

    return {
        "date": last.name.date().isoformat(),
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "signal": signal,
        "cv_mean": np.mean(acc) if acc else 0.0,
        "cv_std": np.std(acc) if acc else 0.0,
        "close": float(last[price_col].iloc[0])
    }

# =======================
# -------- TXT OUTPUT ----------
# =======================
def write_output_txt(all_assets, filename=OUT_TXT):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===================================\n")
        f.write("   MARKET FORECAST – DAILY UPDATE\n")
        f.write("===================================\n")
        f.write(f"Run time (UTC): {datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}\n\n")
        for asset in all_assets:
            f.write(f"--------- {asset['name'].upper()} ---------\n")
            f.write(f"Data date : {asset['date']}\n")
            f.write(f"Close     : {asset.get('close', 0.0):.2f}\n")
            if 'cv_mean' in asset:
                f.write(f"Model CV  : {asset['cv_mean']:.2%} ± {asset['cv_std']:.2%}\n")
            f.write(f"Prob UP   : {asset['prob_up']:.2%}\n")
            f.write(f"Prob DOWN : {asset['prob_down']:.2%}\n")
            f.write(f"Signal    : {asset['signal']}\n\n")
        f.write("===================================\n")
    print(f"[OK] Forecast TXT written: {filename}")
    print(f"[DEBUG] Forecast TXT path: {os.path.abspath(filename)}")

# =======================
# -------- MAIN ----------
# =======================
def main():
    all_assets = []

    # --- Gas ---
    gas_prices = load_gas_prices()
    storage = load_eia_storage()
    gas_df = build_gas_features(gas_prices, storage)
    gas_res = train_gas_model(gas_df)
    all_assets.append(gas_res)

    # --- Oil ---
    oil_df = load_oil_prices()
    oil_res = build_oil_signal(oil_df)
    all_assets.append({
        "name": "OIL (BRENT/WTI)",
        **oil_res,
        "close": oil_res["brent"]
    })

    # --- Weitere Assets ---
    symbols = {
        "GOLD":"GC=F",
        "SILVER":"SI=F",
        "COPPER":"HG=F",
        "SP500":"^GSPC",
        "DAX":"^GDAXI"
    }

    for name, sym in symbols.items():
        df = yf.download(sym, start="2015-01-01", progress=False)
        if df.empty:
            continue
        asset_res = train_ml_model(df, price_col="Close")
        asset_res["name"] = name
        all_assets.append(asset_res)

    # --- TXT erstellen ---
    write_output_txt(all_assets)

if __name__ == "__main__":
    main()
