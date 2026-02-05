"""
Test to verify prediction dates for both Silver and Vietnam Gold.
"""
import pandas as pd
from datetime import datetime, timedelta

def get_future_trading_dates(last_date, num_days=7):
    dates = []
    current_date = pd.to_datetime(last_date)
    while len(dates) < num_days:
        current_date = current_date + timedelta(days=1)
        if current_date.weekday() < 5:  # Skip weekends
            dates.append(current_date)
    return dates

print("="*50)
print("SILVER MODEL")
print("="*50)
df_silver = pd.read_csv("dataset/dataset_enhanced.csv", parse_dates=['Date'])
last_date_silver = df_silver['Date'].iloc[-1]
print(f"Last date: {last_date_silver}")
future_silver = get_future_trading_dates(last_date_silver, 7)
print("Prediction dates:")
for i, d in enumerate(future_silver):
    print(f"  Day {i+1}: {d.strftime('%Y-%m-%d')}")

print("\n" + "="*50)
print("VIETNAM GOLD MODEL")
print("="*50)
df_gold = pd.read_csv("dataset/gold_price_sjc_complete.csv", parse_dates=['date'])
last_date_gold = df_gold['date'].iloc[-1]
print(f"Last date: {last_date_gold}")
future_gold = get_future_trading_dates(last_date_gold, 7)
print("Prediction dates:")
for i, d in enumerate(future_gold):
    print(f"  Day {i+1}: {d.strftime('%Y-%m-%d')}")

print(f"\nToday: {datetime.now().strftime('%Y-%m-%d')}")
