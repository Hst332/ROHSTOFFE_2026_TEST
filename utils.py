import numpy as np
from datetime import datetime, timezone

# --------------------------------------------------
# Signal-Logik mit asset-spezifischen Trade-Schwellen
# --------------------------------------------------

def build_signal(prob_up: float, asset: str):
    thresholds = {
        "NATURAL GAS": 0.56,
        "OIL (BRENT/WTI)": 0.595
    }

    trade_level = thresholds.get(asset, 1.0)  # Default: kein Trade

    if prob_up >= trade_level:
        return "TRADE_UP", trade_level
    elif (1.0 - prob_up) >= trade_level:
        return "TRADE_DOWN", trade_level
    else:
        return "NO_TRADE", trade_level


# --------------------------------------------------
# Einheitlicher Forecast-Result-Builder
# --------------------------------------------------

def build_result(asset, date, close, prob_up, cv_mean=0.0, cv_std=0.0):
    signal, trade_level = build_signal(prob_up, asset)

    return {
        "asset": asset,
        "date": date,
        "close": float(close),
        "prob_up": float(prob_up),
        "prob_down": float(1.0 - prob_up),
        "cv_mean": float(cv_mean),
        "cv_std": float(cv_std),
        "signal": signal,
        "trade_level": trade_level
    }


# --------------------------------------------------
# TXT Daily Report Writer (FINAL)
# --------------------------------------------------

def write_output_txt(results, filepath="MARKET_FORECAST_DAILY.txt"):
    runtime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("===================================\n")
        f.write("   MARKET FORECAST – DAILY UPDATE\n")
        f.write("===================================\n")
        f.write(f"Run time (UTC): {runtime}\n\n")

        for r in results:
            f.write(f"--------- {r['asset']} ---------\n")
            f.write(f"Data date : {r['date']}\n")
            f.write(f"Close     : {r['close']:.2f}\n")

            if r["cv_mean"] > 0:
                f.write(
                    f"Model CV  : {r['cv_mean']*100:.2f}% ± {r['cv_std']*100:.2f}%\n"
                )

            f.write(f"Prob UP   : {r['prob_up']*100:.2f}%\n")
            f.write(f"Prob DOWN : {r['prob_down']*100:.2f}%\n")

            if r["signal"] == "NO_TRADE" and r["trade_level"] < 1:
                f.write(
                    f"Signal    : NO_TRADE      (Trade erst bei >= {r['trade_level']*100:.1f}%)\n"
                )
            else:
                f.write(f"Signal    : {r['signal']}\n")

            f.write("\n")

        f.write("===================================\n")


# --------------------------------------------------
# Beispiel: tägliche Ergebnis-Erzeugung
# (wird normalerweise aus deinen Modellen gespeist)
# --------------------------------------------------

if __name__ == "__main__":
    results = [
        build_result("NATURAL GAS Handen bei +/-7%", "2026-01-02", 3.63, 0.5368, 0.5044, 0.0167),
        build_result("OIL (BRENT/WTI)", "2026-01-02", 60.72, 0.4700),
        build_result("GOLD", "2026-01-02", 4330.90, 0.50),
        build_result("SILVER", "2026-01-02", 71.73, 0.50),
        build_result("COPPER", "2026-01-02", 5.69, 0.50),
        build_result("SP500", "2026-01-02", 6856.86, 0.50),
        build_result("DAX", "2026-01-02", 24539.34, 0.50)
    ]

    write_output_txt(results)
