
import pandas as pd
try:
    df = pd.read_csv("dataset/dataset_enhanced.csv")
    print("Columns:", df.columns.tolist())
    print("Last Price:", df['price'].iloc[-1] if 'price' in df.columns else df['Silver_Close'].iloc[-1])
except Exception as e:
    print(e)
