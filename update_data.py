"""
Script cập nhật dữ liệu giá bạc từ Yahoo Finance
Tải dữ liệu XAG/USD (Spot Silver) từ năm 2000 đến hiện tại
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def update_silver_data():
    """Tải dữ liệu giá bạc XAG/USD mới nhất."""
    
    print("=" * 60)
    print("🥈 CẬP NHẬT DỮ LIỆU GIÁ BẠC (XAG/USD)")
    print("=" * 60)
    
    # Đường dẫn file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'dataset', 'silver_price.csv')
    backup_path = os.path.join(base_dir, 'dataset', 'silver_price_backup.csv')
    
    # Backup file cũ
    if os.path.exists(data_path):
        import shutil
        shutil.copy(data_path, backup_path)
        print(f"✓ Đã backup file cũ: {backup_path}")
    
    # Tải dữ liệu từ Yahoo Finance
    print("\n📥 Đang tải dữ liệu XAG/USD từ Yahoo Finance...")
    
    # Thử nhiều symbol cho Spot Silver
    symbols_to_try = [
        ("XAGUSD=X", "XAG/USD Spot"),      # Spot Silver
        ("SI=F", "Silver Futures"),          # Silver Futures (backup)
    ]
    
    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    df = None
    used_symbol = None
    
    for symbol, name in symbols_to_try:
        try:
            print(f"   Thử tải {name} ({symbol})...")
            ticker = yf.Ticker(symbol)
            temp_df = ticker.history(start=start_date, end=end_date)
            
            if not temp_df.empty and len(temp_df) > 100:
                df = temp_df
                used_symbol = (symbol, name)
                print(f"   ✓ Thành công với {name}!")
                break
        except Exception as e:
            print(f"   ✗ Lỗi với {name}: {e}")
            continue
    
    if df is None or df.empty:
        print("❌ Không thể tải dữ liệu từ Yahoo Finance")
        print("\n💡 Gợi ý: Tải dữ liệu thủ công từ investing.com:")
        print("   1. Truy cập: https://www.investing.com/currencies/xag-usd-historical-data")
        print("   2. Chọn khoảng thời gian và tải xuống CSV")
        print("   3. Đổi tên file thành 'silver_price.csv' và đặt vào thư mục 'dataset/'")
        return False
    
    # Chuyển đổi format
    df = df.reset_index()
    df = df.rename(columns={
        'Date': 'date',
        'Close': 'price',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Volume': 'volume'
    })
    
    # Chỉ giữ các cột cần thiết
    df = df[['date', 'price']]
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Lưu file
    df.to_csv(data_path, index=False)
    
    print(f"\n✅ ĐÃ CẬP NHẬT THÀNH CÔNG!")
    print(f"   📁 File: {data_path}")
    print(f"   📊 Số records: {len(df):,}")
    print(f"   📅 Từ: {df['date'].iloc[0]}")
    print(f"   📅 Đến: {df['date'].iloc[-1]}")
    print(f"   💰 Giá mới nhất: ${df['price'].iloc[-1]:.2f}/oz")
    print(f"   📈 Nguồn: {used_symbol[1]} ({used_symbol[0]})")
    
    return True


def retrain_model():
    """Train lại model với dữ liệu mới."""
    print("\n" + "=" * 60)
    print("🔄 TRAIN LẠI MODEL VỚI DỮ LIỆU MỚI")
    print("=" * 60)
    
    import subprocess
    import sys
    
    # Chạy train.py
    python_path = sys.executable
    train_script = os.path.join(os.path.dirname(__file__), 'src', 'train.py')
    
    print(f"\n🚀 Đang train model...")
    print(f"   Python: {python_path}")
    print(f"   Script: {train_script}")
    print("\n" + "-" * 60)
    
    result = subprocess.run(
        [python_path, train_script, '--epochs', '50'],
        cwd=os.path.dirname(__file__)
    )
    
    return result.returncode == 0


if __name__ == "__main__":
    # Bước 1: Cập nhật dữ liệu
    success = update_silver_data()
    
    if success:
        print("\n")
        user_input = input("🔄 Bạn có muốn train lại model với dữ liệu mới? (y/n): ")
        
        if user_input.lower() == 'y':
            retrain_model()
            print("\n✅ Hoàn tất! Restart server để sử dụng model mới.")
        else:
            print("\n⚠️ Lưu ý: Model cũ sẽ không chính xác với dữ liệu mới.")
            print("   Chạy lệnh sau để train lại:")
            print("   python src/train.py")
    else:
        print("\n❌ Không thể cập nhật dữ liệu. Kiểm tra kết nối internet.")
