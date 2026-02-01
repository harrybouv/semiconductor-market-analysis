# plot_pe_comparison.py
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
OUT_DIR = ROOT / "figures" / "pe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

series = []

for t in TICKERS:
    f = VAL_DIR / f"{t}_pe_monthly.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()

    pe = df["PE"].dropna()
    if pe.empty:
        print(f"{t}: PE series empty, skipping")
        continue

    pe_norm = (pe / pe.median()).rename(t)  # 1.0 = “typical” for that firm
    series.append(pe_norm)

panel = pd.concat(series, axis=1).dropna(how="all")

plt.figure(figsize=(11, 6))
for col in panel.columns:
    plt.plot(panel.index, panel[col], label=col)

plt.axhline(1.0, linewidth=1)
plt.title("Normalised P/E (each firm divided by its own median)")
plt.ylabel("P/E relative to own median")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / "pe_comparison.png"
plt.savefig(out_path, dpi=300)
if SHOW:
    plt.show()
plt.close()

print(f"Saved {out_path}")
