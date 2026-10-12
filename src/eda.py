
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "city_day.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Figure 1: AQI Category Distribution
order = [
    "Moderate", "Satisfactory", "Poor",
    "Very Poor", "Good", "Severe"
]

counts = df["AQI_Bucket"].dropna().str.strip().str.title()
counts = counts.value_counts().reindex(order, fill_value=0)

fig, ax = plt.subplots(figsize=(8, 4.5))
counts.plot(kind="bar", color="steelblue", ax=ax)
ax.set_title("AQI Category Distribution")
ax.set_xlabel("AQI Category")
ax.set_ylabel("Number of Records")
ax.tick_params(axis="x", labelrotation=35)
fig.tight_layout()
fig.savefig(OUT / "aqi_distribution.png", dpi=200)
plt.close(fig)

# Figure 2: PM2.5 Distribution
pm25 = pd.to_numeric(df["PM2.5"], errors="coerce").dropna()

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(pm25, bins=40, kde=True, ax=ax)
ax.set_title("PM2.5 Distribution")
ax.set_xlabel("PM2.5")
ax.set_ylabel("Frequency")
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(OUT / "pm25_distribution.png", dpi=200)
plt.close(fig)

# Figure 3: Pollutant Correlation Matrix (without AQI)
pollutants = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene",
    "Toluene", "Xylene"
]

available = [c for c in pollutants if c in df.columns]
numeric = df[available].apply(pd.to_numeric, errors="coerce")
corr = numeric.corr()

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    ax=ax
)
ax.set_title("Pollutant Correlation Heatmap")
fig.tight_layout()
fig.savefig(
    OUT / "pollutant_correlation.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

print("Three EDA graphs saved in the results folder.")