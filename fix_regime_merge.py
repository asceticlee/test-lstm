import pandas as pd

# Read the forecast file
forecast_df = pd.read_csv('forecasts/regime_forecast_hmm_weekly_20200106_20250809.csv')
print(f"Forecast file shape: {forecast_df.shape}")
print(f"Forecast columns: {forecast_df.columns.tolist()}")

# Read the regime assignments file
regime_df = pd.read_csv('regime_analysis/regime_assignments.csv')
print(f"Regime file shape: {regime_df.shape}")
print(f"Regime columns: {regime_df.columns.tolist()}")

# The issue is that regime_assignments has multiple entries per trading day
# We need to get just one regime per trading day
print(f"\nEntries per trading day in regime assignments (sample):")
sample_day = regime_df['TradingDay'].iloc[0]
sample_count = len(regime_df[regime_df['TradingDay'] == sample_day])
print(f"Trading day {sample_day} has {sample_count} entries")

# Create a deduplicated regime mapping - one regime per trading day
daily_regimes = regime_df.groupby('TradingDay')['Regime'].first().reset_index()
print(f"\nDaily regime mapping shape: {daily_regimes.shape}")
print(f"Sample daily regimes:")
print(daily_regimes.head())

# Merge forecast with daily regimes (one-to-one mapping)
merged_df = forecast_df.merge(
    daily_regimes, 
    left_on='trading_day', 
    right_on='TradingDay', 
    how='left'
)

print(f"\nMerged shape: {merged_df.shape}")
print(f"Null regimes after merge: {merged_df['Regime'].isnull().sum()}")

# Remove the duplicate TradingDay column
if 'TradingDay' in merged_df.columns:
    merged_df = merged_df.drop('TradingDay', axis=1)

# Reorder columns to put the regime column after 'actual'
columns = merged_df.columns.tolist()
actual_idx = columns.index('actual')

# Create new column order
new_columns = (columns[:actual_idx+1] + 
               ['Regime'] + 
               [col for col in columns[actual_idx+1:] if col != 'Regime'])

merged_df = merged_df[new_columns]

print(f"\nFinal columns: {merged_df.columns.tolist()}")
print(f"Final shape: {merged_df.shape}")

# Save the corrected file
output_file = 'forecasts/regime_forecast_hmm_weekly_with_actual_regimes_corrected.csv'
merged_df.to_csv(output_file, index=False)
print(f"\nSaved corrected file to: {output_file}")

# Show regime distribution in the forecast data
print("\nRegime distribution in forecast data:")
regime_counts = merged_df['Regime'].value_counts().sort_index()
total = len(merged_df)
for regime, count in regime_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count:,} ({percentage:.2f}%)")

# Verify no duplicates
print(f"\nVerifying uniqueness:")
print(f"Original forecast unique (day, ms): {forecast_df[['trading_day', 'ms_of_day']].drop_duplicates().shape[0]}")
print(f"Merged unique (day, ms): {merged_df[['trading_day', 'ms_of_day']].drop_duplicates().shape[0]}")
print(f"Should be equal: {forecast_df.shape[0] == merged_df.shape[0]}")

# Show sample of the corrected data
print(f"\nSample of corrected data:")
print(merged_df[['trading_day', 'ms_of_day', 'actual', 'Regime', 'upside_regime']].head(10))
