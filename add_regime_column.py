import pandas as pd

# Read the forecast file
forecast_df = pd.read_csv('forecasts/regime_forecast_hmm_weekly_20200106_20250809.csv')
print(f"Forecast file shape: {forecast_df.shape}")
print(f"Forecast columns: {forecast_df.columns.tolist()}")

# Read the regime assignments file
regime_df = pd.read_csv('regime_analysis/regime_assignments.csv')
print(f"Regime file shape: {regime_df.shape}")
print(f"Regime columns: {regime_df.columns.tolist()}")

# Check some sample values to understand the data
print("\nFirst few forecast rows:")
print(forecast_df[['trading_day', 'ms_of_day']].head())

print("\nFirst few regime rows:")
print(regime_df[['TradingDay']].head())

# Since we have trading_day in forecast and TradingDay in regime, let's merge on those
# First, let's check if they have compatible formats
print("\nUnique trading_day values in forecast (first 10):")
print(sorted(forecast_df['trading_day'].unique())[:10])

print("\nUnique TradingDay values in regime (first 10):")
print(sorted(regime_df['TradingDay'].unique())[:10])

# Merge on trading day
merged_df = forecast_df.merge(
    regime_df[['TradingDay', 'Regime']], 
    left_on='trading_day', 
    right_on='TradingDay', 
    how='left'
)

print(f"\nMerged shape: {merged_df.shape}")
print(f"Null regimes after merge: {merged_df['Regime'].isnull().sum()}")

# Reorder columns to put the regime column after 'actual'
columns = merged_df.columns.tolist()
actual_idx = columns.index('actual')

# Create new column order
new_columns = (columns[:actual_idx+1] + 
               ['Regime'] + 
               [col for col in columns[actual_idx+1:] if col not in ['Regime', 'TradingDay']])

merged_df = merged_df[new_columns]

print(f"\nFinal columns: {merged_df.columns.tolist()}")

# Save the updated file
output_file = 'forecasts/regime_forecast_hmm_weekly_with_actual_regimes.csv'
merged_df.to_csv(output_file, index=False)
print(f"\nSaved to: {output_file}")

# Show regime distribution in the forecast data
print("\nRegime distribution in forecast data:")
regime_counts = merged_df['Regime'].value_counts().sort_index()
total = len(merged_df)
for regime, count in regime_counts.items():
    percentage = (count / total) * 100
    print(f"Regime {regime}: {count} ({percentage:.2f}%)")
