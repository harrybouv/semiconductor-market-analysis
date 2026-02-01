# plot_pe_history.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]
SHOW = False

HERE = Path(__file__).resolve()

def find_root() -> Path:
    for p in [HERE.parent, *HERE.parents]:
        prices_dir = p / "data" / "prices"
        if prices_dir.exists() and any(prices_dir.glob("*_monthly_prices.csv")):
            return p
    raise FileNotFoundError("Could not find root containing data/prices/*_monthly_prices.csv")

ROOT = find_root()

VAL_DIR = ROOT / "data" / "valuation"
OUT_DIR = ROOT / "figures" / "pe" / "history"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for t in TICKERS:
    f = VAL_DIR / f"{t}_pe_monthly.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()

    pe = df["PE"].dropna()
    if pe.empty:
        print(f"{t}: PE series empty, skipping")
        continue

    med = pe.median()
    p90 = pe.quantile(0.9)

    plt.figure(figsize=(10, 5))
    plt.plot(pe, label="P/E")
    plt.axhline(med, linestyle="--", label="Median")
    plt.axhline(p90, linestyle=":", label="90th pct")
    plt.title(f"{t} Price-to-Earnings")
    plt.ylabel("P/E")
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / f"{t}_pe_history.png"
    plt.savefig(out_path, dpi=300)
    if SHOW:
        plt.show()
    plt.close()

    print(f"Saved {out_path}")
print(t, "rows:", len(df), "cols:", df.columns.tolist())
print(t, "PE non-null:", df["PE"].notna().sum())
