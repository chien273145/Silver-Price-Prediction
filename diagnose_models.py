"""
Diagnostic script for Silver and Gold prediction models.
Checks: data loading, features, model loading, prediction output.
"""
import sys, os
sys.path.append(os.getcwd())

def check_silver():
    print("=" * 60)
    print("SILVER MODEL DIAGNOSTIC")
    print("=" * 60)
    
    try:
        from src.enhanced_predictor import EnhancedPredictor
        p = EnhancedPredictor()
        
        # 1. Data loading
        print("\n[1] Data Loading...")
        p.load_data()
        n = len(p.data)
        last_date = p.data['Date'].iloc[-1]
        last_price = p.data['price'].iloc[-1]
        print(f"    Records: {n:,}")
        print(f"    Date range: {p.data['Date'].iloc[0]} -> {last_date}")
        print(f"    Last price: ${last_price:.2f} USD")
        
        # 2. Features
        print("\n[2] Features...")
        if p.feature_columns:
            print(f"    Feature columns: {len(p.feature_columns)}")
            print(f"    XGB columns: {len(p.xgb_feature_columns) if p.xgb_feature_columns else 'N/A'}")
        else:
            print("    [WARNING] No feature columns generated!")
        
        # 3. Model loading
        print("\n[3] Model Loading...")
        p.load_model()
        n_models = len(p.models) if p.models else 0
        has_xgb = bool(p.xgb_models)
        has_lstm = bool(p.lstm_model)
        print(f"    Ridge models: {n_models}")
        print(f"    XGBoost: {'Yes' if has_xgb else 'No'}")
        print(f"    LSTM: {'Yes' if has_lstm else 'No'}")
        
        # 4. Prediction (VND)
        print("\n[4] Prediction (VND)...")
        res = p.predict(in_vnd=True)
        print(f"    Currency: {res.get('currency', '?')}")
        print(f"    Unit: {res.get('unit', '?')}")
        print(f"    Exchange rate: {res.get('exchange_rate', '?')}")
        last_known = res.get('last_known', {})
        print(f"    Last known: {last_known.get('date', '?')} -> {last_known.get('price', 0):,.0f} VND")
        
        print("\n    7-Day Predictions:")
        for pred in res.get('predictions', []):
            print(f"      Day {pred['day']} ({pred['date']}): {pred['price']:,.0f} VND  (USD: ${pred['price_usd']:.2f})  Change: {pred['change']['percentage']:+.2f}%")
        
        print("\n    [OK] Silver model working!")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n    [ERROR] {e}")
        traceback.print_exc()
        return False


def check_gold():
    print("\n" + "=" * 60)
    print("GOLD MODEL DIAGNOSTIC")
    print("=" * 60)
    
    try:
        from src.gold_predictor import GoldPredictor
        g = GoldPredictor()
        
        # 1. Data loading
        print("\n[1] Data Loading...")
        g.load_data()
        g.create_features()
        n = len(g.data)
        last_date = g.data['date'].iloc[-1]
        last_price = g.data['price'].iloc[-1]
        print(f"    Records: {n:,}")
        print(f"    Date range: {g.data['date'].iloc[0]} -> {last_date}")
        print(f"    Last price: ${last_price:.2f} USD")
        
        # 2. Features
        print("\n[2] Features...")
        if g.feature_columns:
            print(f"    Feature columns: {len(g.feature_columns)}")
            print(f"    XGB columns: {len(g.xgb_feature_columns) if g.xgb_feature_columns else 'N/A'}")
        else:
            print("    [WARNING] No feature columns generated!")
        
        # 3. Model loading
        print("\n[3] Model Loading...")
        g.load_model()
        n_models = len(g.models) if g.models else 0
        has_xgb = bool(g.xgb_models)
        has_lstm = bool(g.lstm_model)
        print(f"    Ridge models: {n_models}")
        print(f"    XGBoost: {'Yes' if has_xgb else 'No'}")
        print(f"    LSTM: {'Yes' if has_lstm else 'No'}")
        
        # 4. Prediction (VND)
        print("\n[4] Prediction (VND)...")
        res = g.predict(in_vnd=True)
        print(f"    Currency: {res.get('currency', '?')}")
        print(f"    Unit: {res.get('unit', '?')}")
        last_known = res.get('last_known', {})
        print(f"    Last known: {last_known.get('date', '?')} -> {last_known.get('price', 0):,.0f} VND")
        
        print("\n    7-Day Predictions:")
        for pred in res.get('predictions', []):
            print(f"      Day {pred['day']} ({pred['date']}): {pred['price']:,.0f} VND  (USD: ${pred['price_usd']:.2f})  Change: {pred['change']['percentage']:+.2f}%")
        
        print("\n    [OK] Gold model working!")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n    [ERROR] {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    silver_ok = check_silver()
    gold_ok = check_gold()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Silver: {'OK' if silver_ok else 'FAILED'}")
    print(f"  Gold:   {'OK' if gold_ok else 'FAILED'}")
