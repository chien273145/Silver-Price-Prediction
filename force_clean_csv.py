
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
            
    # Filter < 60 USD
    if 'Silver_Close' in df.columns:
        df = df[df['Silver_Close'] < 60]
    elif 'price' in df.columns:
        df = df[df['price'] < 60]
        
    print(f"Cleaned shape: {df.shape}")
    
    # Save
    df.to_csv("dataset/dataset_enhanced.csv", index=False)
    print("Saved clean dataset.")
    
except Exception as e:
    print(e)
