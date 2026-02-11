
import pandas as pd
import os

try:
    df = pd.read_csv("dataset/dataset_enhanced.csv")
    print(f"Original shape: {df.shape}")
    
    # Filter numeric values only
    numeric_cols = ['Silver_Close', 'price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Filter < 45 USD (ATH ~50, Current ~30)
    # 58 was suspicious.
    if 'Silver_Close' in df.columns:
        df = df[df['Silver_Close'] < 45]
    elif 'price' in df.columns:
        df = df[df['price'] < 45]
        
    print(f"Cleaned shape: {df.shape}")
    if not df.empty:
        print(f"Last Price: {df['Silver_Close'].iloc[-1] if 'Silver_Close' in df.columns else df['price'].iloc[-1]}")
    
    # Save
    df.to_csv("dataset/dataset_enhanced.csv", index=False)
    print("Saved clean dataset.")
    
except Exception as e:
    print(e)
