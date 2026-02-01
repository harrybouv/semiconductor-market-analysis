from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

VAL_DIR = Path("../data/valuation")
OUT_DIR = Path("../figures/ps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]

SHOW = False  # set True if you also want pop-up windows

series = []

for t in TICKERS:
    f = VAL_DIR / f"{t}_ps_monthly.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()

    ps = df["PS"].dropna()
    if ps.empty:
        print(f"{t}: PS series empty, skipping")
        continue

    ps_norm = (ps / ps.median()).rename(t)  # 1.0 = “typical” for that firm
    series.append(ps_norm)

panel = pd.concat(series, axis=1).dropna(how="all")

plt.figure(figsize=(11, 6))
for col in panel.columns:
    plt.plot(panel.index, panel[col], label=col)

plt.axhline(1.0, linewidth=1)  # median baseline
plt.title("Normalised P/S (each firm divided by its own median)")
plt.ylabel("P/S relative to own median")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / "ps_comparison.png"
plt.savefig(out_path, dpi=300)
if SHOW:
    plt.show()
plt.close()

print(f"Saved {out_path}")
