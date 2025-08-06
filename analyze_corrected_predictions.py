import pandas as pd
import numpy as np

# Read the corrected file with actual regimes
df = pd.read_csv('forecasts/regime_forecast_hmm_weekly_with_actual_regimes_corrected.csv')

print(f"Total records: {len(df):,}")
print(f"Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")

# Remove NaN regimes for analysis
df_clean = df.dropna(subset=['Regime']).copy()
print(f"Records after removing NaN regimes: {len(df_clean):,}")

print("\n=== REGIME DISTRIBUTION COMPARISON ===")

print("\nActual Regime Distribution:")
actual_counts = df_clean['Regime'].value_counts().sort_index()
total = len(df_clean)
for regime, count in actual_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {int(regime)}: {count:,} ({percentage:.2f}%)")

print("\nUpside Predicted Regime Distribution:")
upside_counts = df_clean['upside_regime'].value_counts().sort_index()
for regime, count in upside_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count:,} ({percentage:.2f}%)")

print("\nDownside Predicted Regime Distribution:")
downside_counts = df_clean['downside_regime'].value_counts().sort_index()
for regime, count in downside_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count:,} ({percentage:.2f}%)")

# Calculate accuracy metrics
print("\n=== PREDICTION ACCURACY ===")

# Upside prediction accuracy
upside_correct = (df_clean['Regime'] == df_clean['upside_regime']).sum()
upside_accuracy = (upside_correct / total) * 100
print(f"Upside prediction accuracy: {upside_correct:,}/{total:,} ({upside_accuracy:.2f}%)")

# Downside prediction accuracy  
downside_correct = (df_clean['Regime'] == df_clean['downside_regime']).sum()
downside_accuracy = (downside_correct / total) * 100
print(f"Downside prediction accuracy: {downside_correct:,}/{total:,} ({downside_accuracy:.2f}%)")

# Confusion matrix for upside predictions
print("\n=== UPSIDE PREDICTION CONFUSION MATRIX ===")
confusion_upside = pd.crosstab(df_clean['Regime'], df_clean['upside_regime'], margins=True)
print(confusion_upside)

print("\n=== DOWNSIDE PREDICTION CONFUSION MATRIX ===")
confusion_downside = pd.crosstab(df_clean['Regime'], df_clean['downside_regime'], margins=True)
print(confusion_downside)

# Per-regime accuracy
print("\n=== PER-REGIME ACCURACY (Upside) ===")
for regime in sorted(df_clean['Regime'].unique()):
    regime_mask = df_clean['Regime'] == regime
    regime_total = regime_mask.sum()
    regime_correct = ((df_clean['Regime'] == regime) & (df_clean['upside_regime'] == regime)).sum()
    regime_accuracy = (regime_correct / regime_total) * 100 if regime_total > 0 else 0
    print(f"Regime {int(regime)}: {regime_correct:,}/{regime_total:,} ({regime_accuracy:.2f}%)")

print("\n=== PER-REGIME ACCURACY (Downside) ===")
for regime in sorted(df_clean['Regime'].unique()):
    regime_mask = df_clean['Regime'] == regime
    regime_total = regime_mask.sum()
    regime_correct = ((df_clean['Regime'] == regime) & (df_clean['downside_regime'] == regime)).sum()
    regime_accuracy = (regime_correct / regime_total) * 100 if regime_total > 0 else 0
    print(f"Regime {int(regime)}: {regime_correct:,}/{regime_total:,} ({regime_accuracy:.2f}%)")
