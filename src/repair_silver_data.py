
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

def repair_silver_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    silver_path = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    enhanced_path = os.path.join(base_dir, 'dataset', 'dataset_enhanced.csv')
    
    print(f"Repairing {silver_path}...")
    
    # 1. Load Raw Dataset
    try:
        df = pd.read_csv(silver_path)
    except Exception as e:
        print(f"Error reading silver csv: {e}")
        return

    # Map Vietnamese columns
    # "Ngày","Lần cuối","Mở","Cao","Thấp","KL","% Thay đổi"
    col_map = {
        'Ngày': 'date',
        'Lần cuối': 'close',
        'Mở': 'open',
        'Cao': 'high',
        'Thấp': 'low'
    }
    
    # Check if headers are Vietnamese or English
    is_vietnamese = 'Ngày' in df.columns
    
    if is_vietnamese:
        # Standardize
        df['date'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y', errors='coerce')
        # If format failed (maybe mixed), try ISO
        if df['date'].isna().sum() > len(df) * 0.5:
             df['date'] = pd.to_datetime(df['Ngày'], errors='coerce')
             
        # Parse numbers (handle text commas? No, CSV usually has dots for decimals in this file based on viewing)
        # But wait, Step 720 showed "84.7040". So it's dot decimal.
        for col in ['Lần cuối', 'Mở', 'Cao', 'Thấp']:
            if col in df.columns:
                # Force numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        # Assume English/Standard headers from previous repairs
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # 2. Identify Corruption
    # Prices > 60 USD are definitely wrong for Silver (All-time high is ~50)
    # Checking 'Lần cuối' (Close)
    close_col = 'Lần cuối' if is_vietnamese else 'close'
    
    trash_mask = df[close_col] > 60
    trash_count = trash_mask.sum()
    
    print(f"Found {trash_count} corrupt records (Price > 60 USD).")
    
    if trash_count > 0:
        print("Removing corrupt records...")
        df_clean = df[~trash_mask].copy()
    else:
        print("No corruption found? Checking range...")
        print(df[close_col].describe())
        df_clean = df.copy()
        
    last_valid_date = df_clean['date'].max()
    print(f"Last valid date: {last_valid_date}")
    
    # 3. Fetch Missing Data (Gap Filling)
    today = datetime.now()
    start_fetch = last_valid_date + timedelta(days=1)
    
    if start_fetch < today:
        print(f"Fetching missing data from {start_fetch} to {today}...")
        try:
            ticker = yf.Ticker("SI=F")
            hist = ticker.history(start=start_fetch, end=today)
            
            # Filter Anomalies > 60 USD (Simulation/Future data)
            if not hist.empty and 'Close' in hist.columns:
                hist = hist[hist['Close'] < 60]
                
            if not hist.empty:
                print(f"Fetched {len(hist)} new tokens.")
                
                new_rows = []
                for idx, row in hist.iterrows():
                    # Format as existing structure
                    # "Ngày","Lần cuối","Mở","Cao","Thấp","KL","% Thay đổi"
                    date_str = idx.strftime('%d/%m/%Y')
                    
                    if is_vietnamese:
                        new_row = {
                            "Ngày": date_str,
                            "Lần cuối": row['Close'],
                            "Mở": row['Open'],
                            "Cao": row['High'],
                            "Thấp": row['Low'],
                            "KL": "", 
                            "% Thay đổi": "" 
                        }
                    else:
                        # Fallback if structure changed
                        new_row = {
                            "date": idx,
                            "close": row['Close'],
                            "open": row['Open'],
                            "high": row['High'],
                            "low": row['Low']
                        }
                    new_rows.append(new_row)
                
                new_df = pd.DataFrame(new_rows)
                
                # Make sure columns align
                if is_vietnamese:
                    # Append
                    df_final = pd.concat([df_clean, new_df], ignore_index=True)
                    # Use original sorting (descending for this file type typically?)
                    # Step 720 showed descending.
                    # Convert date back to datetime for sorting
                    df_final['date_temp'] = pd.to_datetime(df_final['Ngày'], format='%d/%m/%Y')
                    df_final = df_final.sort_values('date_temp', ascending=False).drop(columns=['date_temp', 'date'])
                else:
                    df_final = pd.concat([df_clean, new_df], ignore_index=True)
            else:
                print("No new data found on Yahoo.")
                df_final = df_clean.drop(columns=['date']) if 'date' in df_clean.columns and is_vietnamese else df_clean
        except Exception as e:
            print(f"Failed to fetch yfinance: {e}")
            df_final = df_clean
    else:
        df_final = df_clean

    # Save Cleaned Silver Dataset
    print(f"Saving cleaned dataset to {silver_path}...")
    df_final.to_csv(silver_path, index=False, float_format='%.4f')
    
    # 4. Fix dataset_enhanced.csv
    # This is trickier as it has external cols.
    # Simplest is to DROP corruption and APPEND new data (if we had external data).
    # But for now, let's just drop the corrupted tail.
    # The `enhanced_predictor` calculates external features on the fly for *new* predictions, 
    # but relies on this file for training.
    # We should clean it.
    
    print(f"Cleaning {enhanced_path}...")
    if os.path.exists(enhanced_path):
        edf = pd.read_csv(enhanced_path)
        # Check col 'Silver_Close' or 'price'
        p_col = 'Silver_Close' if 'Silver_Close' in edf.columns else 'price'
        
        if p_col in edf.columns:
            trash_mask_e = edf[p_col] > 60
            print(f"Found {trash_mask_e.sum()} corrupt records in enhanced dataset.")
            edf_clean = edf[~trash_mask_e].copy()
            
            # We should probably *limit* the file to the valid range matching silver dataset.
            # Or just save the clean version.
            # If we don't fetch new external data (Gold, DXY) for the gap, the model will lack recent training data.
            # But it's better than corrupt data.
            # Also, if we re-run `fetch_external_data.py`? That's for Gold.
            
            edf_clean.to_csv(enhanced_path, index=False)
            print("Saved cleaned enhanced dataset.")
            
    print("Done.")

if __name__ == "__main__":
    repair_silver_data()
