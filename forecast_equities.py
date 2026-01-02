from utils import build_trend_vol_features
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def load_equity(symbol, start="2015-01-01"):
    df = yf.download(symbol, start=start, progress=False)
    df = df[["Close"]].rename(columns={"Close": symbol})
    df.dropna(inplace=True)
    return df

def train_equity_model(df):
    df = build_trend_vol_features(df, price_col=df.columns[0])
    df["Target"] = (df[df.columns[0]].shift(-1) > df[df.columns[0]]).astype(int)
    features = [c for c in df.columns if "trend" in c or "vol" in c]
    
    X = df[features]
    y = df["Target"]
    
    tscv = TimeSeriesSplit(n_splits=5)
    acc = []
    for tr, te in tscv.split(X):
        m = LogisticRegression(max_iter=200)
        m.fit(X.iloc[tr], y.iloc[tr])
        acc.append(np.mean(m.predict(X.iloc[te]) == y.iloc[te]))
    
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    
    last = df.iloc[-1:]
    prob_up = model.predict_proba(last[features])[0][1]
    signal = ("UP" if prob_up >= 0.57 else "DOWN" if prob_up <= 0.43 else "NO_TRADE")
    
    return {
        "name": df.columns[0],
        "date": last.index[0].date().isoformat(),
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "signal": signal,
        "cv_mean": np.mean(acc),
        "cv_std": np.std(acc)
    }
