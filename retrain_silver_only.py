"""Quick Silver retrain + save test."""
import sys, os
sys.path.insert(0, os.getcwd())

from src.enhanced_predictor import EnhancedPredictor

p = EnhancedPredictor()
p.load_data()

last_price = p.data['price'].iloc[-1]
print(f"Last price: ${last_price:.2f}")
print(f"Total rows: {len(p.data)}")

if last_price > 50:
    print("[ERROR] Dataset still has inflated prices!")
else:
    print("[OK] Dataset looks clean. Training...")
    try:
        metrics = p.train()
        print(f"Metrics: {metrics}")
        print("Saving model...")
        p.save_model()
        print("[SAVED] Silver model saved!")
        
        # Verify file timestamp
        import datetime
        model_path = os.path.join(p.model_dir, 'ridge_enhanced_models.pkl')
        mtime = os.path.getmtime(model_path)
        print(f"Model file updated at: {datetime.datetime.fromtimestamp(mtime)}")
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()
