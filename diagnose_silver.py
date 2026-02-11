"""Quick Silver-only diagnostic with full error output."""
import sys, os
sys.path.append(os.getcwd())

try:
    from src.enhanced_predictor import EnhancedPredictor
    p = EnhancedPredictor()
    p.load_data()
    
    n = len(p.data)
    last_price = p.data['price'].iloc[-1]
    last_date = p.data['Date'].iloc[-1]
    print(f"Data: {n} rows, last={last_date}, price=${last_price:.2f}")
    print(f"Features: {len(p.feature_columns) if p.feature_columns else 0}")
    
    # Try loading model
    p.load_model()
    n_models = len(p.models) if p.models else 0
    print(f"Models loaded: {n_models}")
    
    # Check feature mismatch
    model_features = p.feature_columns
    data_features = [c for c in p.data.columns]
    missing = [f for f in model_features if f not in data_features]
    if missing:
        print(f"\n[PROBLEM] Missing {len(missing)} features in data:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")
    else:
        print("All model features present in data.")
    
    # Try predict
    res = p.predict(in_vnd=True)
    print(f"\nPrediction OK!")
    for pred in res['predictions']:
        print(f"  Day {pred['day']} ({pred['date']}): {pred['price']:,.0f} VND (${pred['price_usd']:.2f})")

except Exception as e:
    import traceback
    print(f"\n[FAILED] {type(e).__name__}: {e}")
    traceback.print_exc()
