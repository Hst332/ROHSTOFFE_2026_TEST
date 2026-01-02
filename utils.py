import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# =======================
# TXT Output für mehrere Assets
# =======================
def write_output_txt(filename, results):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("===================================\n")
        f.write("   ENERGY FORECAST – CODE A\n")
        f.write("===================================\n")
        f.write(f"Run time (UTC): {datetime.utcnow():%Y-%m-%d %H:%M:%S UTC}\n\n")
        for asset in results:
            f.write(f"--------- {asset['name']} ---------\n")
            f.write(f"Data date : {asset['date']}\n")
            if "cv_mean" in asset:
                f.write(f"Model CV  : {asset['cv_mean']:.2%} ± {asset['cv_std']:.2%}\n")
            if "close" in asset:
                f.write(f"Close      : {asset['close']:.2f}\n")
            if "spread" in asset:
                f.write(f"Spread     : {asset['spread']:.2f}\n")
            f.write(f"Prob UP   : {asset['prob_up']:.2%}\n")
            f.write(f"Prob DOWN : {asset['prob_down']:.2%}\n")
            f.write(f"Signal    : {asset['signal']}\n\n")
        f.write("===================================\n")

# =======================
# ML Feature Builder
# =======================
def build_trend_vol_features(df, price_col="Close", trend_windows=[5,20], vol_window=10):
    df = df.copy()
    df["ret"] = df[price_col].pct_change()
    for w in trend_windows:
        df[f"trend_{w}"] = df[price_col].pct_change(w)
    df[f"vol_{vol_window}"] = df["ret"].rolling(vol_window).std()
    df.dropna(inplace=True)
    return df

def train_ml_model(df, price_col="Close", up_threshold=0.57, down_threshold=0.43):
    df = build_trend_vol_features(df, price_col=price_col)
    df["Target"] = (df[price_col].shift(-1) > df[price_col]).astype(int)
    features = [c for c in df.columns if "trend" in c or "vol" in c]
    
    X = df[features]
    y = df["Target"]
    
    tscv = TimeSeriesSplit(n_splits=5)
    acc = []
    for tr, te in tscv.split(X):
        m = LogisticRegression(max_iter=200)
        m.fit(X.iloc[tr], y.iloc[tr])
        acc.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
    
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    
    last = df.iloc[-1:]
    prob_up = model.predict_proba(last[features])[0][1]
    signal = "UP" if prob_up >= up_threshold else "DOWN" if prob_up <= down_threshold else "NO_TRADE"
    
    return {
        "date": last.index[0].date().isoformat(),
        "prob_up": prob_up,
        "prob_down": 1.0 - prob_up,
        "signal": signal,
        "cv_mean": np.mean(acc),
        "cv_std": np.std(acc),
        "close": float(last[price_col])
    }
