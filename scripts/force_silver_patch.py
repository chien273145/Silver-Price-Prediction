import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import sys

# Add root to path
sys.path.append(os.getcwd())

from backend.realtime_data import RealTimeDataFetcher

def patch_silver_data():
    csv_path = 'dataset/dataset_enhanced.csv'
    print(f"Reading {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        last_date = df['Date'].max()
        print(f"Last Date in CSV: {last_date.date()}")
        
        today = datetime.now()
        # Ensure we are in simulation year 2026
        sim_year = 2026
        
        if (today - last_date).days < 1:
            print("dataset is up to date!")
            return
            
        # Target dates in 2026
        start_fill = last_date + timedelta(days=1)
        
        # Calculate corresponding 2025 dates for fetching
        # We need data for [start_fill_2026 ... today_2026]
        # Fetch data for [start_fill_2025 ... today_2025]
        
        # Shift back 1 year
        fetch_start_2025 = start_fill.replace(year=start_fill.year - 1)
        fetch_end_2025 = today.replace(year=today.year - 1)
        
        days_to_fetch = (fetch_end_2025 - fetch_start_2025).days + 5
        
        print(f"Simulation Gap: {start_fill.date()} to {today.date()}")
        print(f"Fetching 2025 Real Data: {fetch_start_2025.date()} (past {days_to_fetch} days)...")
        
        fetcher = RealTimeDataFetcher()
        silver_symbol = "SI=F" 
        
        silver_hist = fetcher.get_historical_prices(days=days_to_fetch, symbol=silver_symbol)
        gold_hist = fetcher.get_historical_prices(days=days_to_fetch, symbol=fetcher.GOLD_SYMBOL)
        dxy_hist = fetcher.get_historical_prices(days=days_to_fetch, symbol=fetcher.DXY_SYMBOL)
        vix_hist = fetcher.get_historical_prices(days=days_to_fetch, symbol=fetcher.VIX_SYMBOL)
        
        # Helper to get price
        def get_price(hist_data, target_date_2025, col='close'):
            if not hist_data or 'data' not in hist_data: return None
            target_str = target_date_2025.strftime('%Y-%m-%d')
            for row in hist_data['data']:
                if row['date'] == target_str:
                    return row.get(col)
            return None
            
        def get_row(hist_data, target_date_2025):
            if not hist_data or 'data' not in hist_data: return None
            target_str = target_date_2025.strftime('%Y-%m-%d')
            for row in hist_data['data']:
                if row['date'] == target_str:
                    return row
            return None

        # Iterate 2026 days
        new_rows = []
        curr_2026 = start_fill
        
        # Get last valid row from DataFrame for forward fill
        last_row_series = df.iloc[-1]
        last_valid_row = {
            'Silver_Close': last_row_series.get('Silver_Close'),
            'Silver_Open': last_row_series.get('Silver_Open'),
            'Silver_High': last_row_series.get('Silver_High'),
            'Silver_Low': last_row_series.get('Silver_Low'),
            'Gold': last_row_series.get('Gold'),
            'DXY': last_row_series.get('DXY'),
            'VIX': last_row_series.get('VIX')
        }
        
        while curr_2026.date() <= today.date():
            target_2025 = curr_2026 - timedelta(weeks=52)
            
            s_row = get_row(silver_hist, target_2025)
            
            row_dict = {}
            row_dict['Date'] = curr_2026.strftime('%Y-%m-%d')
            
            if s_row:
                g_close = get_price(gold_hist, target_2025)
                d_close = get_price(dxy_hist, target_2025)
                v_close = get_price(vix_hist, target_2025)
                
                row_dict.update({
                    'Silver_Close': s_row.get('close'),
                    'Silver_Open': s_row.get('open'),
                    'Silver_High': s_row.get('high'),
                    'Silver_Low': s_row.get('low'),
                    'Gold': g_close if g_close else last_valid_row['Gold'],
                    'DXY': d_close if d_close else last_valid_row['DXY'],
                    'VIX': v_close if v_close else last_valid_row['VIX']
                })
                # Update last_valid
                last_valid_row = row_dict.copy()
                del last_valid_row['Date']
                print(f"  + Patching {curr_2026.date()} using real data")
            else:
                # Forward Fill
                # Only if weekend logic check passed? 
                # Actually if it's Monday 2026, we WANT a row.
                # If we have NO data, we forward fill.
                # But check if it's weekend in 2026?
                if curr_2026.weekday() >= 5: # Sat=5, Sun=6
                     print(f"  . Skipping {curr_2026.date()} (Weekend)")
                else:
                    row_dict.update(last_valid_row)
                    print(f"  + Patching {curr_2026.date()} using FORWARD FILL (No real data)")
            
            if 'Silver_Close' in row_dict:
                new_rows.append(row_dict)

            curr_2026 += timedelta(days=1)
            
        if not new_rows:
            print("No new data found to append.")
            return

        # Append
        new_df = pd.DataFrame(new_rows)
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        cols = df.columns.tolist()
        for c in cols:
            if c not in new_df.columns:
                new_df[c] = None 
        
        new_df = new_df[cols]
        combined = pd.concat([df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        
        combined.to_csv(csv_path, index=False)
        print(f"✅ Successfully patched {len(new_rows)} rows. Last date: {combined['Date'].max().date()}")
        
    except Exception as e:
        print(f"❌ Error patching: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    patch_silver_data()
