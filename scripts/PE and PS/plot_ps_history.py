from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

VAL_DIR = Path("../data/valuation")
OUT_DIR = Path("../figures/ps/history")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]

SHOW = False  # set True if you also want pop-up windows

for t in TICKERS:
    f = VAL_DIR / f"{t}_ps_monthly.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()

    ps = df["PS"].dropna()
    if ps.empty:
        print(f"{t}: PS series empty, skipping")
        continue

    med = ps.median()
    p90 = ps.quantile(0.9)

    plt.figure(figsize=(10, 5))
    plt.plot(ps, label="P/S")
    plt.axhline(med, linestyle="--", label="Median")
    plt.axhline(p90, linestyle=":", label="90th pct")
    plt.title(f"{t} Price-to-Sales")
    plt.ylabel("P/S")
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"{t}_ps_history.png"
    plt.savefig(out_path, dpi=300)
    if SHOW:
        plt.show()
    plt.close()

    print(f"Saved {out_path}")
