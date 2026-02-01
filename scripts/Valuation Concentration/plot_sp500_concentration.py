# plot_sp500_concentration.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()

def find_root() -> Path:
    for p in [HERE.parent, *HERE.parents]:
        if (p / "data").exists():
            return p
    raise FileNotFoundError("Could not find project root (no /data folder found)")

ROOT = find_root()

DATA = ROOT / "data" / "concentration" / "sp500_topn_share.csv"
if not DATA.exists():
    raise FileNotFoundError(f"Missing {DATA} (run build_sp500_concentration.py successfully first)")

OUT_DIR = ROOT / "figures" / "concentration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=[0], index_col=0).sort_index()

plt.figure(figsize=(11, 6))
plt.plot(df.index, df["top_5_share"], label="Top 5 share")
plt.plot(df.index, df["top_10_share"], label="Top 10 share")
plt.title("S&P 500 Market Cap Concentration (Top-N Share)")
plt.ylabel("Share of total market cap")
plt.legend()
plt.tight_layout()

out_path = OUT_DIR / "sp500_topn_concentration.png"
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Saved: {out_path}")
