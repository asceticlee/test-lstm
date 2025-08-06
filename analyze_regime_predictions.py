import pandas as pd
import numpy as np

# Read the file with actual regimes
df = pd.read_csv('forecasts/regime_forecast_hmm_weekly_with_actual_regimes.csv')

print(f"Total records: {len(df):,}")
print(f"Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")

# Compare predicted vs actual regime distributions
print("\n=== REGIME DISTRIBUTION COMPARISON ===")

print("\nActual Regime Distribution:")
actual_counts = df['Regime'].value_counts().sort_index()
total = len(df)
for regime, count in actual_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count:,} ({percentage:.2f}%)")

print("\nUpside Predicted Regime Distribution:")
upside_counts = df['upside_regime'].value_counts().sort_index()
for regime, count in upside_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count:,} ({percentage:.2f}%)")

print("\nDownside Predicted Regime Distribution:")
downside_counts = df['downside_regime'].value_counts().sort_index()
for regime, count in downside_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count:,} ({percentage:.2f}%)")

# Calculate accuracy metrics
print("\n=== PREDICTION ACCURACY ===")

# Upside prediction accuracy
upside_correct = (df['Regime'] == df['upside_regime']).sum()
upside_accuracy = (upside_correct / total) * 100
print(f"Upside prediction accuracy: {upside_correct:,}/{total:,} ({upside_accuracy:.2f}%)")

# Downside prediction accuracy  
downside_correct = (df['Regime'] == df['downside_regime']).sum()
downside_accuracy = (downside_correct / total) * 100
print(f"Downside prediction accuracy: {downside_correct:,}/{total:,} ({downside_accuracy:.2f}%)")

# Confusion matrix for upside predictions
print("\n=== UPSIDE PREDICTION CONFUSION MATRIX ===")
confusion_upside = pd.crosstab(df['Regime'], df['upside_regime'], margins=True)
print(confusion_upside)

print("\n=== DOWNSIDE PREDICTION CONFUSION MATRIX ===")
confusion_downside = pd.crosstab(df['Regime'], df['downside_regime'], margins=True)
print(confusion_downside)

# Per-regime accuracy
print("\n=== PER-REGIME ACCURACY (Upside) ===")
for regime in sorted(df['Regime'].unique()):
    regime_mask = df['Regime'] == regime
    regime_total = regime_mask.sum()
    regime_correct = ((df['Regime'] == regime) & (df['upside_regime'] == regime)).sum()
    regime_accuracy = (regime_correct / regime_total) * 100 if regime_total > 0 else 0
    print(f"Regime {regime}: {regime_correct:,}/{regime_total:,} ({regime_accuracy:.2f}%)")

print("\n=== PER-REGIME ACCURACY (Downside) ===")
for regime in sorted(df['Regime'].unique()):
    regime_mask = df['Regime'] == regime
    regime_total = regime_mask.sum()
    regime_correct = ((df['Regime'] == regime) & (df['downside_regime'] == regime)).sum()
    regime_accuracy = (regime_correct / regime_total) * 100 if regime_total > 0 else 0
    print(f"Regime {regime}: {regime_correct:,}/{regime_total:,} ({regime_accuracy:.2f}%)")
