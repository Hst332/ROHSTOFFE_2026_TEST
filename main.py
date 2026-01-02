from forecast_gas import train_gas_model, load_gas_prices, load_eia_storage, build_gas_features
from forecast_oil import build_oil_signal, load_oil_prices
from forecast_metals import load_metal, train_metal_model
from forecast_equities import load_equity, train_equity_model
from utils import write_output_txt

OUT_TXT = "output/energy_forecast_output.txt"

def main():
    results = []

    # GAS
    gas_prices = load_gas_prices()
    storage = load_eia_storage()
    gas_df = build_gas_features(gas_prices, storage)
    model, features, cv_mean, cv_std = train_gas_model(gas_df)
    last = gas_df.iloc[-1:]
    prob_up = model.predict_proba(last[features])[0][1]
    signal = ("UP" if prob_up >= 0.54 else "DOWN" if prob_up <= 0.46 else "NO_TRADE")
    results.append({
        "name": "NATURAL GAS",
        "date": last.index[0].date().isoformat(),
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "signal": signal,
        "cv_mean": cv_mean,
        "cv_std": cv_std
    })

    # OIL
    oil_df = load_oil_prices()
    oil_res = build_oil_signal(oil_df)
    results.append(oil_res)

    # METALS
    for symbol in ["GC=F", "SI=F", "HG=F"]:  # Gold, Silber, Kupfer
        df = load_metal(symbol)
        res = train_metal_model(df)
        results.append(res)

    # EQUITIES
    for symbol in ["^GSPC", "^GDAXI"]:  # SP500, DAX
        df = load_equity(symbol)
        res = train_equity_model(df)
        results.append(res)

    # WRITE TXT
    write_output_txt(OUT_TXT, results)
    print("[OK] Energy forecast written to", OUT_TXT)

if __name__ == "__main__":
    main()
