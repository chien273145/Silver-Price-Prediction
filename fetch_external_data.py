"""
Script để fetch external data từ Yahoo Finance.
Dữ liệu bao gồm: Gold Price, USD Index (DXY), VIX Index
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_external_data(start_date: str = "2008-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Fetch external market data từ Yahoo Finance.
    
    Args:
        start_date: Ngày bắt đầu (format: YYYY-MM-DD)
        end_date: Ngày kết thúc (mặc định là hôm nay)
    
    Returns:
        DataFrame với columns: Date, Gold, DXY, VIX
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Fetching external data from {start_date} to {end_date}...")
    
    # Define tickers
    tickers = {
        'GC=F': 'Gold',      # Gold Futures
        'DX-Y.NYB': 'DXY',   # US Dollar Index
        '^VIX': 'VIX'        # Volatility Index
    }
    
    all_data = {}
    
    for ticker, name in tickers.items():
        try:
            print(f"  Fetching {name} ({ticker})...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                # Lấy Close price
                if isinstance(data.columns, pd.MultiIndex):
                    all_data[name] = data['Close'][ticker]
                else:
                    all_data[name] = data['Close']
                print(f"    ✓ Got {len(all_data[name])} records")
            else:
                print(f"    ✗ No data for {name}")
                
        except Exception as e:
            print(f"    ✗ Error fetching {name}: {e}")
    
    # Combine all data
    if all_data:
        df = pd.DataFrame(all_data)
        df.index.name = 'Date'
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        # Forward fill missing values (weekends, holidays)
        df = df.ffill()
        
        print(f"\n✓ Combined dataset: {len(df)} records")
        return df
    else:
        print("✗ Failed to fetch any data")
        return pd.DataFrame()


def merge_with_silver_data(external_df: pd.DataFrame, silver_csv_path: str) -> pd.DataFrame:
    """
    Merge external data với silver price data.
    
    Args:
        external_df: DataFrame từ fetch_external_data()
        silver_csv_path: Path tới file CSV silver data
    
    Returns:
        DataFrame đã merge
    """
    print(f"\nMerging with silver data from {silver_csv_path}...")
    
    # Read silver data
    silver_df = pd.read_csv(silver_csv_path)
    
    # Parse date - handle Vietnamese format
    def parse_date(date_str):
        try:
            # Try DD/MM/YYYY format
            return pd.to_datetime(date_str, format='%d/%m/%Y').date()
        except:
            try:
                # Try YYYY-MM-DD format
                return pd.to_datetime(date_str).date()
            except:
                return None
    
    silver_df['Date'] = silver_df['Ngày'].apply(parse_date)
    
    # Rename columns
    silver_df = silver_df.rename(columns={
        'Lần cuối': 'Silver_Close',
        'Mở': 'Silver_Open',
        'Cao': 'Silver_High',
        'Thấp': 'Silver_Low',
        '% Thay đổi': 'Silver_Change'
    })
    
    # Select relevant columns
    silver_df = silver_df[['Date', 'Silver_Close', 'Silver_Open', 'Silver_High', 'Silver_Low']]
    
    # Convert to numeric
    for col in ['Silver_Close', 'Silver_Open', 'Silver_High', 'Silver_Low']:
        silver_df[col] = pd.to_numeric(silver_df[col], errors='coerce')
    
    # Merge
    merged_df = pd.merge(silver_df, external_df, on='Date', how='left')
    
    # Sort by date
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    # Forward fill missing external data
    for col in ['Gold', 'DXY', 'VIX']:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].ffill().bfill()
    
    # Drop rows with NaN
    merged_df = merged_df.dropna()
    
    print(f"✓ Merged dataset: {len(merged_df)} records")
    print(f"  Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    
    return merged_df


def save_enhanced_dataset(df: pd.DataFrame, output_path: str):
    """Lưu dataset đã enhance."""
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved enhanced dataset to {output_path}")


def main():
    """Main function."""
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    silver_csv = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    output_csv = os.path.join(base_dir, 'dataset', 'dataset_enhanced.csv')
    
    # Fetch external data
    external_df = fetch_external_data(start_date="2008-01-01")
    
    if external_df.empty:
        print("Failed to fetch external data. Exiting.")
        return
    
    # Merge with silver data
    merged_df = merge_with_silver_data(external_df, silver_csv)
    
    if merged_df.empty:
        print("Failed to merge data. Exiting.")
        return
    
    # Save
    save_enhanced_dataset(merged_df, output_csv)
    
    # Show sample
    print("\n" + "="*50)
    print("Sample of enhanced dataset:")
    print(merged_df.tail(10).to_string())
    
    print("\n" + "="*50)
    print("Statistics:")
    print(merged_df.describe())


if __name__ == "__main__":
    main()
