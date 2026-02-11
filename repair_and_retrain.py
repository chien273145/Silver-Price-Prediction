"""
Repair datasets and retrain both Silver + Gold models.
Removes anomalous 2026 "future" prices from Yahoo Finance.
"""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# STEP 1: Analyze & Repair Silver Dataset
# ============================================================
def repair_silver():
    path = os.path.join(BASE_DIR, 'dataset', 'dataset_enhanced.csv')
    df = pd.read_csv(path)
    
    print("=" * 60)
    print("SILVER DATASET REPAIR")
    print("=" * 60)
    print(f"  Original rows: {len(df)}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Silver price range: ${df['Silver_Close'].min():.2f} - ${df['Silver_Close'].max():.2f}")
    
    # Find anomalous data: Silver > $50 USD is definitely fake
    anomalous = df[df['Silver_Close'] > 50]
    print(f"\n  Anomalous rows (Silver > $50): {len(anomalous)}")
    if len(anomalous) > 0:
        print(f"  Anomalous date range: {anomalous['Date'].min()} to {anomalous['Date'].max()}")
        print(f"  Anomalous price range: ${anomalous['Silver_Close'].min():.2f} - ${anomalous['Silver_Close'].max():.2f}")
    
    # Also check Gold column for anomalies (Gold > $3500 is fake)
    if 'Gold_Close' in df.columns:
        gold_anomalous = df[df['Gold_Close'] > 3500]
        print(f"  Anomalous Gold rows (> $3500): {len(gold_anomalous)}")
    
    # Remove anomalous rows
    clean = df[df['Silver_Close'] <= 50].copy()
    if 'Gold_Close' in clean.columns:
        clean = clean[clean['Gold_Close'] <= 3500].copy()
    
    print(f"\n  Clean rows: {len(clean)}")
    print(f"  Clean date range: {clean['Date'].min()} to {clean['Date'].max()}")
    print(f"  Clean Silver price: ${clean['Silver_Close'].min():.2f} - ${clean['Silver_Close'].max():.2f}")
    if 'Gold_Close' in clean.columns:
        print(f"  Clean Gold price: ${clean['Gold_Close'].min():.2f} - ${clean['Gold_Close'].max():.2f}")
    
    # Save
    clean.to_csv(path, index=False)
    print(f"\n  [SAVED] {path}")
    return len(anomalous)

# ============================================================
# STEP 2: Analyze & Repair Gold Dataset  
# ============================================================
def repair_gold():
    path = os.path.join(BASE_DIR, 'dataset', 'gold_enhanced_dataset.csv')
    if not os.path.exists(path):
        print("\n  [SKIP] gold_enhanced_dataset.csv not found")
        return 0
    
    df = pd.read_csv(path)
    print("\n" + "=" * 60)
    print("GOLD DATASET REPAIR")
    print("=" * 60)
    print(f"  Original rows: {len(df)}")
    
    price_col = 'gold_close' if 'gold_close' in df.columns else 'price'
    print(f"  Price column: {price_col}")
    print(f"  Price range: ${df[price_col].min():.2f} - ${df[price_col].max():.2f}")
    
    anomalous = df[df[price_col] > 3500]
    print(f"  Anomalous rows (> $3500): {len(anomalous)}")
    if len(anomalous) > 0:
        print(f"  Anomalous price range: ${anomalous[price_col].min():.2f} - ${anomalous[price_col].max():.2f}")
    
    clean = df[df[price_col] <= 3500].copy()
    clean.to_csv(path, index=False)
    print(f"  Clean rows: {len(clean)}")
    print(f"  Clean price range: ${clean[price_col].min():.2f} - ${clean[price_col].max():.2f}")
    print(f"  [SAVED] {path}")
    return len(anomalous)

# ============================================================
# STEP 3: Retrain Silver Model
# ============================================================
def retrain_silver():
    print("\n" + "=" * 60)
    print("RETRAINING SILVER MODEL (Ridge + XGB + LSTM)")
    print("=" * 60)
    
    sys.path.insert(0, BASE_DIR)
    from src.enhanced_predictor import EnhancedPredictor
    
    p = EnhancedPredictor()
    p.load_data()
    
    last_price = p.data['price'].iloc[-1]
    print(f"  Last price after cleaning: ${last_price:.2f}")
    
    if last_price > 50:
        print("  [ERROR] Price still > $50 after cleaning! Aborting.")
        return False
    
    # Train
    metrics = p.train()
    print(f"\n  Training metrics: {metrics}")
    
    # Save
    p.save_model()
    print("  [SAVED] Silver model saved!")
    return True

# ============================================================
# STEP 4: Retrain Gold Model
# ============================================================
def retrain_gold():
    print("\n" + "=" * 60)
    print("RETRAINING GOLD MODEL (Ridge + XGB + LSTM)")
    print("=" * 60)
    
    sys.path.insert(0, BASE_DIR)
    from src.gold_predictor import GoldPredictor
    
    g = GoldPredictor()
    g.load_data()
    g.create_features()
    
    last_price = g.data['price'].iloc[-1]
    print(f"  Last price after cleaning: ${last_price:.2f}")
    
    if last_price > 3500:
        print("  [ERROR] Price still > $3500 after cleaning! Aborting.")
        return False
    
    # Train
    metrics = g.train()
    print(f"\n  Training metrics: {metrics}")
    
    # Save
    g.save_model()
    print("  [SAVED] Gold model saved!")
    return True

# ============================================================
# STEP 5: Verify Predictions
# ============================================================
def verify():
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    sys.path.insert(0, BASE_DIR)
    
    # Silver
    try:
        from src.enhanced_predictor import EnhancedPredictor
        p = EnhancedPredictor()
        p.load_data()
        p.load_model()
        res = p.predict(in_vnd=True)
        
        pred = res['predictions'][0]
        print(f"\n  Silver Day 1: {pred['price']:,.0f} VND (${pred['price_usd']:.2f} USD)")
        print(f"  Target: ~3,100,000 VND (~$32 USD)")
        
        if pred['price_usd'] < 50:
            print("  [OK] Silver predictions look realistic!")
        else:
            print("  [WARNING] Silver still predicting > $50 USD")
    except Exception as e:
        print(f"  [ERROR] Silver verification failed: {e}")
    
    # Gold
    try:
        from src.gold_predictor import GoldPredictor
        g = GoldPredictor()
        g.load_data()
        g.create_features()
        g.load_model()
        res = g.predict(in_vnd=True)
        
        pred = res['predictions'][0]
        print(f"\n  Gold Day 1: {pred['price']:,.0f} VND (${pred['price_usd']:.2f} USD)")
        print(f"  Target: ~18,000,000 VND (~$2,800 USD)")
        
        if pred['price_usd'] < 3500:
            print("  [OK] Gold predictions look realistic!")
        else:
            print("  [WARNING] Gold still predicting > $3500 USD")
    except Exception as e:
        print(f"  [ERROR] Gold verification failed: {e}")


if __name__ == "__main__":
    print("Starting full data repair and model retraining...\n")
    
    # Step 1-2: Repair data
    silver_removed = repair_silver()
    gold_removed = repair_gold()
    
    print(f"\n  Total anomalous rows removed: Silver={silver_removed}, Gold={gold_removed}")
    
    if silver_removed == 0 and gold_removed == 0:
        print("  No anomalous data found. Datasets may already be clean.")
    
    # Step 3-4: Retrain
    silver_ok = retrain_silver()
    gold_ok = retrain_gold()
    
    # Step 5: Verify
    if silver_ok or gold_ok:
        verify()
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
