"""Generate charts for the project report."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/dataset.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['is_weekend'] = df['datetime'].dt.dayofweek >= 5

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── 1. EDA overview ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('French Day-Ahead Electricity Price — EDA (Jan 2023 – Apr 2026)',
             fontsize=14, fontweight='bold', y=1.01)

# Monthly avg time series
ax = axes[0, 0]
monthly = df.set_index('datetime')['price'].resample('ME').mean()
ax.plot(monthly.index, monthly.values, linewidth=2, color='#c0392b')
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#c0392b')
ax.set_title('Monthly Average Price', fontweight='bold')
ax.set_ylabel('EUR/MWh')
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=30)

# Distribution
ax = axes[0, 1]
ax.hist(df['price'], bins=80, color='#2980b9', edgecolor='white', linewidth=0.3, alpha=0.85)
ax.axvline(df['price'].mean(), color='red', linestyle='--', linewidth=1.5,
           label=f"Mean {df['price'].mean():.0f}")
ax.axvline(df['price'].median(), color='orange', linestyle='--', linewidth=1.5,
           label=f"Median {df['price'].median():.0f}")
ax.set_title('Price Distribution', fontweight='bold')
ax.set_xlabel('EUR/MWh')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Intraday weekday vs weekend
ax = axes[0, 2]
hourly = df.groupby(['hour', 'is_weekend'])['price'].mean().unstack()
ax.plot(hourly.index, hourly[False], marker='o', markersize=4, label='Weekday',
        color='#2c3e50', linewidth=2)
ax.plot(hourly.index, hourly[True], marker='s', markersize=4, label='Weekend',
        color='#e67e22', linewidth=2, linestyle='--')
ax.set_title('Intraday Price Pattern', fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('EUR/MWh')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 24, 3))

# Seasonal pattern
ax = axes[1, 0]
m_mean = df.groupby('month')['price'].mean()
m_std  = df.groupby('month')['price'].std()
ax.bar(range(1,13), m_mean, yerr=m_std, color='#27ae60', alpha=0.75,
       error_kw={'linewidth':1, 'capsize':3}, edgecolor='white')
ax.set_xticks(range(1,13))
ax.set_xticklabels(MONTHS, fontsize=8)
ax.set_title('Seasonal Pattern', fontweight='bold')
ax.set_ylabel('EUR/MWh')
ax.grid(True, alpha=0.3, axis='y')

# Net load vs price
ax = axes[1, 1]
sc = ax.scatter(df['net_load_mw']/1000, df['price'], alpha=0.08, s=4,
                c=df['renewable_penetration_pct'], cmap='RdYlGn_r', vmin=0, vmax=0.8)
ax.set_title('Net Load vs Price\n(color = renewable penetration)', fontweight='bold')
ax.set_xlabel('Net Load (GW)')
ax.set_ylabel('EUR/MWh')
ax.set_ylim(-150, 300)
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Renew. %', fontsize=8)

# Feature importance
ax = axes[1, 2]
fi = pd.read_csv(ROOT / "research_notebooks/reports/xgboost_feature_importance.csv")
top = fi.head(10)
cat_colors = {
    'Historical Lags': '#c0392b', 'Rolling Stats': '#e74c3c',
    'Calendar': '#3498db', 'Weather': '#27ae60',
    'Renewable Production': '#f39c12', 'Other': '#95a5a6'
}
bar_colors = [cat_colors.get(c, '#95a5a6') for c in top['category']]
ax.barh(range(len(top)), top['importance'], color=bar_colors, edgecolor='white')
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f[:24] for f in top['feature']], fontsize=8)
ax.invert_yaxis()
ax.set_title('XGBoost — Top 10 Features', fontweight='bold')
ax.set_xlabel('Importance')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(DOCS / "eda_overview.png", dpi=150, bbox_inches='tight')
print("Saved eda_overview.png")
plt.close()

# ── 2. Model comparison ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Walk-Forward Model Comparison (2023–2026)', fontsize=14, fontweight='bold')

metrics = {
    'Ridge':    {'mae': 14.77, 'rmse': 19.62, 'r2': 0.806, 'dir': 78.5},
    'XGBoost':  {'mae': 13.41, 'rmse': 18.34, 'r2': 0.831, 'dir': 76.7},
    'LightGBM\n(P50)': {'mae': 13.67, 'rmse': 19.05, 'r2': 0.819, 'dir': 76.4},
}
models = list(metrics.keys())
colors = ['#3498db', '#e74c3c', '#f39c12']

# MAE & RMSE
ax = axes[0]
x = np.arange(len(models))
w = 0.35
mae_vals  = [metrics[m]['mae']  for m in models]
rmse_vals = [metrics[m]['rmse'] for m in models]
b1 = ax.bar(x - w/2, mae_vals,  w, label='MAE',  color=colors, alpha=0.8)
b2 = ax.bar(x + w/2, rmse_vals, w, label='RMSE', color=colors, alpha=0.4, hatch='//')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('EUR/MWh')
ax.set_title('MAE & RMSE', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

# R²
ax = axes[1]
r2_vals = [metrics[m]['r2'] for m in models]
bars = ax.bar(models, r2_vals, color=colors, alpha=0.85, edgecolor='white')
ax.set_ylim(0.78, 0.85)
ax.set_ylabel('R²')
ax.set_title('R² Score', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Direction accuracy
ax = axes[2]
dir_vals = [metrics[m]['dir'] for m in models]
bars = ax.bar(models, dir_vals, color=colors, alpha=0.85, edgecolor='white')
ax.set_ylim(70, 82)
ax.set_ylabel('%')
ax.set_title('Direction Accuracy', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(50, color='gray', linestyle=':', linewidth=1.2, label='Random baseline')
ax.legend(fontsize=9)
for bar, v in zip(bars, dir_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(DOCS / "model_comparison.png", dpi=150, bbox_inches='tight')
print("Saved model_comparison.png")
plt.close()

# ── 3. LightGBM quantile intervals ───────────────────────────────────────────
preds = pd.read_csv(ROOT / "research_notebooks/reports/lightgbm_quantile_predictions.csv")
preds['datetime'] = pd.to_datetime(preds['datetime'])

# Sample 2 weeks
sample = preds.iloc[:24*14].copy()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('LightGBM Quantile Regression — Prediction Intervals', fontsize=13, fontweight='bold')

ax = axes[0]
ax.fill_between(sample['datetime'], sample['pred_p10'], sample['pred_p90'],
                alpha=0.25, color='#3498db', label='P10–P90 interval')
ax.plot(sample['datetime'], sample['pred_p50'], linewidth=1.8, color='#2980b9', label='P50 (median)')
ax.plot(sample['datetime'], sample['actual'], linewidth=1.5, color='#c0392b',
        alpha=0.85, label='Actual')
ax.set_title('2-Week Sample: Actual vs Quantile Predictions', fontweight='bold')
ax.set_ylabel('EUR/MWh')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=30)

# Interval width distribution
ax = axes[1]
width = preds['interval_width']
ax.hist(width, bins=60, color='#3498db', alpha=0.8, edgecolor='white')
ax.axvline(width.mean(), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean width: {width.mean():.1f} EUR/MWh')
ax.set_title('Distribution of P10–P90 Interval Width', fontweight='bold')
ax.set_xlabel('Interval Width (EUR/MWh)')
ax.set_ylabel('Count')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DOCS / "lightgbm_quantile.png", dpi=150, bbox_inches='tight')
print("Saved lightgbm_quantile.png")
plt.close()

print("\nAll charts generated in docs/")
