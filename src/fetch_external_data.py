"""
Fetch External Market Data for Gold Prediction
Downloads: VIX, DXY (US Dollar Index), Oil (WTI), Interest Rates (US 10Y)
Merges with existing gold dataset.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("yfinance not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf
    HAS_YFINANCE = True

print("=" * 70)
print("FETCHING EXTERNAL MARKET DATA")
print("=" * 70)

# Load existing gold dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gold_path = os.path.join(base_dir, 'dataset', 'gold_geopolitical_dataset.csv')

print(f"\nLoading gold dataset from {gold_path}...")
gold_df = pd.read_csv(gold_path)
gold_df['date'] = pd.to_datetime(gold_df['date'])
print(f"Gold dataset: {len(gold_df)} rows, date range: {gold_df['date'].min()} to {gold_df['date'].max()}")

# Date range
start_date = gold_df['date'].min().strftime('%Y-%m-%d')
end_date = gold_df['date'].max().strftime('%Y-%m-%d')

# Tickers to fetch
tickers = {
    '^VIX': 'vix',           # CBOE Volatility Index
    'DX-Y.NYB': 'dxy',       # US Dollar Index  
    'CL=F': 'oil',           # WTI Crude Oil Futures
    '^TNX': 'us10y'          # US 10-Year Treasury Yield
}

print(f"\nFetching data from {start_date} to {end_date}...")

# Fetch each ticker
external_data = {}
for ticker, name in tickers.items():
    print(f"  Fetching {name} ({ticker})...", end=" ")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            external_data[name] = data['Close'].reset_index()
            external_data[name].columns = ['date', name]
            external_data[name]['date'] = pd.to_datetime(external_data[name]['date'])
            print(f"OK ({len(data)} rows)")
        else:
            print("No data")
    except Exception as e:
        print(f"Error: {e}")

# Merge all external data
print("\nMerging external data with gold dataset...")

merged_df = gold_df.copy()
for name, df in external_data.items():
    merged_df = merged_df.merge(df, on='date', how='left')
    print(f"  Added {name}: {merged_df[name].notna().sum()} values")

# Fill missing values with forward fill then backward fill
for name in external_data.keys():
    if name in merged_df.columns:
        merged_df[name] = merged_df[name].ffill().bfill()

# Create additional features from external data
print("\nCreating derived features...")

# VIX features
if 'vix' in merged_df.columns:
    merged_df['vix_ma7'] = merged_df['vix'].rolling(7).mean()
    merged_df['vix_ma30'] = merged_df['vix'].rolling(30).mean()
    merged_df['vix_change'] = merged_df['vix'].pct_change()
    merged_df['high_vix'] = (merged_df['vix'] > 25).astype(int)
    print("  VIX features created")

# DXY features  
if 'dxy' in merged_df.columns:
    merged_df['dxy_ma7'] = merged_df['dxy'].rolling(7).mean()
    merged_df['dxy_ma30'] = merged_df['dxy'].rolling(30).mean()
    merged_df['dxy_change'] = merged_df['dxy'].pct_change()
    merged_df['dxy_momentum'] = merged_df['dxy'] - merged_df['dxy'].shift(7)
    print("  DXY features created")

# Oil features
if 'oil' in merged_df.columns:
    merged_df['oil_ma7'] = merged_df['oil'].rolling(7).mean()
    merged_df['oil_ma30'] = merged_df['oil'].rolling(30).mean()
    merged_df['oil_change'] = merged_df['oil'].pct_change()
    merged_df['oil_volatility'] = merged_df['oil'].rolling(14).std()
    print("  Oil features created")

# Interest rate features
if 'us10y' in merged_df.columns:
    merged_df['us10y_ma7'] = merged_df['us10y'].rolling(7).mean()
    merged_df['us10y_ma30'] = merged_df['us10y'].rolling(30).mean()
    merged_df['us10y_change'] = merged_df['us10y'].pct_change()
    merged_df['rate_trend'] = merged_df['us10y'] - merged_df['us10y'].shift(30)
    print("  Interest rate features created")

# Cross-asset features
if 'vix' in merged_df.columns and 'dxy' in merged_df.columns:
    merged_df['fear_index'] = merged_df['vix'] * merged_df['dxy'] / 100
    print("  Fear index created")

if 'oil' in merged_df.columns and 'gold_close' in merged_df.columns:
    merged_df['gold_oil_ratio'] = merged_df['gold_close'] / (merged_df['oil'] + 0.01)
    print("  Gold/Oil ratio created")

# Fill any remaining NaN with 0
for col in merged_df.columns:
    if merged_df[col].isna().any():
        merged_df[col] = merged_df[col].fillna(0)

# Save enhanced dataset
output_path = os.path.join(base_dir, 'dataset', 'gold_enhanced_dataset.csv')
merged_df.to_csv(output_path, index=False)
print(f"\nEnhanced dataset saved to {output_path}")
print(f"Total rows: {len(merged_df)}")
print(f"Total columns: {len(merged_df.columns)}")

# Show new columns
new_cols = [c for c in merged_df.columns if c not in gold_df.columns]
print(f"\nNew columns added ({len(new_cols)}):")
for col in new_cols:
    print(f"  - {col}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
