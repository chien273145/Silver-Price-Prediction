"""Check model metrics after retraining."""
import sys, os, joblib, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Silver
try:
    silver_path = os.path.join("models", "ridge_enhanced_models.pkl")
    data = joblib.load(silver_path)
    m = data.get("latest_metrics", {})
    print("=== SILVER MODEL ===")
    print(f"  Version:  {data.get('model_version', '?')}")
    print(f"  R2:       {m.get('avg_r2', 0):.4f}")
    print(f"  RMSE:     ${m.get('rmse', 0):.4f}")
    print(f"  MAE:      ${m.get('mae', 0):.4f}")
    print(f"  MAPE:     {m.get('mape', 0):.2f}%")
    print(f"  Ensemble: {m.get('ensemble', False)}")
    print(f"  Weights:  {data.get('ensemble_weights', {})}")
except Exception as e:
    print(f"Silver load error: {e}")

print()

# Gold
try:
    gold_path = os.path.join("models", "vietnam_gold_models.pkl")
    with open(gold_path, "rb") as f:
        data = pickle.load(f)
    m = data.get("metrics", {})
    r2s   = list(m.get("r2",   {}).values())
    mapes = list(m.get("mape", {}).values())
    rmses = list(m.get("rmse", {}).values())
    print("=== GOLD MODEL ===")
    print(f"  Version:  {data.get('model_version', '?')}")
    print(f"  Avg R2:   {np.mean(r2s):.4f}")
    print(f"  Avg MAPE: {np.mean(mapes):.2f}%")
    print(f"  Avg RMSE: {np.mean(rmses):,.0f} VND")
    print(f"  Weights:  {data.get('ensemble_weights', {})}")
    print(f"  USD/VND:  {data.get('usd_vnd_rate', '?')}")
    print(f"  Trained:  {data.get('trained_at', '?')}")
except Exception as e:
    print(f"Gold load error: {e}")

print()

# Quick prediction check
print("=== SILVER PREDICTION ===")
try:
    from src.enhanced_predictor import EnhancedPredictor
    p = EnhancedPredictor()
    p.load_model()
    result = p.predict(in_vnd=True)
    preds = result.get("predictions", [])
    for pred in preds[:3]:
        print(f"  {pred['date']}: {pred['predicted_price']:,.0f} VND/luong ({pred['change_percent']:+.2f}%)")
except Exception as e:
    print(f"Silver predict error: {e}")

print()
print("=== GOLD PREDICTION ===")
try:
    from src.vietnam_gold_predictor import VietnamGoldPredictor
    p = VietnamGoldPredictor()
    p.load_model()
    p.load_vietnam_data()
    p.load_world_data()
    p.merge_datasets()
    p.create_transfer_features()
    result = p.predict()
    for pred in result[:3]:
        print(f"  {pred['date']}: {pred['predicted_price']:,.0f} VND/luong ({pred['change_percent']:+.2f}%)")
except Exception as e:
    print(f"Gold predict error: {e}")
