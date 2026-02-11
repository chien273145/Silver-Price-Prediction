
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())
# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.enhanced_predictor import EnhancedPredictor

    p = EnhancedPredictor()
    p.load_data()
    
    # Manually compute minimal features
    df = p.data
    price = df['price']
    
    # Lag 1
    df['price_lag_1'] = price.shift(1)
    
    # MA 7
    df['price_ma7'] = price.rolling(window=7).mean()
    
    # Sentiment (dummy if missing)
    if 'sentiment_change' not in df.columns:
        df['sentiment_change'] = 0.0
    
    # Let's try to run add_technical_indicators if available.
    try:
        p.add_technical_indicators(df)
        print("add_technical_indicators ran.")
    except Exception as e:
        print(f"add_technical_indicators failed: {e}")
        
    # Drop NaN created by rolling/shift
    df.dropna(inplace=True)
    p.data = df
    
    print(f"Last Price for Prediction: {p.data['price'].iloc[-1]}")
    
    res = p.predict(in_vnd=False)
    print("Prediction (USD):", res['predictions'][0]['price'])
    
    res_vnd = p.predict(in_vnd=True)
    print("Prediction (VND):", res_vnd['predictions'][0]['price'])
    
except Exception as e:
    import traceback
    traceback.print_exc()
