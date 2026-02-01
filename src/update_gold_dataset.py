
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def update_gold_dataset():
    print("=" * 60)
    print("🌟 UPDATING GOLD DATASET (Appending New Data)")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset', 'gold_enhanced_dataset.csv')
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        return False
        
    # Load existing data
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    last_date = df['date'].iloc[-1]
    print(f"Last date in dataset: {last_date.date()}")
    
    today = datetime.now()
    if last_date.date() >= today.date():
        print("✅ Dataset is already up to date.")
        return True
        
    start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_date} to {end_date}...")
    
    # 1. Fetch Gold Price (GC=F)
    try:
        gold_ticker = yf.Ticker("GC=F")
        gold_data = gold_ticker.history(start=start_date, end=end_date)
        
        if gold_data.empty:
            print("⚠️ No new gold data found.")
            return True
            
        print(f"✓ Found {len(gold_data)} new days for Gold")
        
    except Exception as e:
        print(f"❌ Error fetching Gold: {e}")
        return False
        
    # Prepare new rows
    new_rows = pd.DataFrame()
    new_rows['date'] = gold_data.index.tz_localize(None) # Remove timezone
    
    # Map Gold columns
    # Existing columns: gold_open, gold_high, gold_low, gold_close
    new_rows['gold_open'] = gold_data['Open'].values
    new_rows['gold_high'] = gold_data['High'].values
    new_rows['gold_low'] = gold_data['Low'].values
    new_rows['gold_close'] = gold_data['Close'].values
    new_rows['price'] = gold_data['Close'].values # standard price column
    new_rows['Volume'] = gold_data['Volume'].values
    
    # 2. Fetch External Data
    tickers = {
        '^VIX': 'vix',
        'DX-Y.NYB': 'dxy',
        'CL=F': 'oil',
        '^TNX': 'us10y',
        'SI=F': 'silver_close' # Need silver too for GS Ratio
    }
    
    for ticker, col_name in tickers.items():
        try:
            print(f"  Fetching {col_name} ({ticker})...", end=" ")
            ext_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not ext_data.empty:
                # Align with new_rows dates
                # Reindex to handle missing days (holidays differ?)
                # Actually, usually safer to just merge on date
                temp = ext_data['Close'].reset_index()
                temp.columns = ['date', col_name]
                temp['date'] = pd.to_datetime(temp['date']).dt.tz_localize(None)
                
                new_rows = new_rows.merge(temp, on='date', how='left')
                print("OK")
            else:
                print("No Data")
        except Exception as e:
            print(f"Error: {e}")

    # 3. Fill missing external data (ffill)
    # Also need to carry over Geopolitical columns (GPR) from last known row
    # Get last row of existing df
    last_row = df.iloc[-1]
    
    # Columns that should be forward filled (static/missing)
    # GPR columns, etc.
    cols_to_ffill = [c for c in df.columns if c not in new_rows.columns]
    
    print("Forward filling static columns (GPR, etc.)...")
    for col in cols_to_ffill:
        new_rows[col] = last_row[col]
        
    # 4. Calculate Derived Columns (GS Ratio, etc) if needed
    # But GoldPredictor.create_features will recalculate dynamic features (lags, ma, etc.)
    # So we just need Raw Data.
    
    # Ensure all columns match order
    # Add new rows to df
    print(f"Appending {len(new_rows)} new rows...")
    combined = pd.concat([df, new_rows], ignore_index=True)
    
    # Sort and Deduplicate
    combined = combined.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    
    # Fill remaining NaNs (e.g. if VIX missing on Gold trading day)
    combined = combined.ffill()
    
    # Save
    combined.to_csv(data_path, index=False)
    print(f"✅ Successfully updated dataset to {combined['date'].iloc[-1].date()}")
    
    print(f"✅ Successfully updated dataset to {combined['date'].iloc[-1].date()}")
    
    return True

import traceback

if __name__ == "__main__":
    try:
        update_gold_dataset()
    except Exception:
        traceback.print_exc()
