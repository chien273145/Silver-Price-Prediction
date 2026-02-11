
from src.enhanced_predictor import EnhancedPredictor
from src.gold_predictor import GoldPredictor
import json

def retrain():
    print("="*50)
    print("RETRAINING SILVER MODEL")
    print("="*50)
    
    silver = EnhancedPredictor()
    try:
        metrics = silver.train()
        print("\nSilver Metrics:")
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"Silver training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("RETRAINING GOLD MODEL")
    print("="*50)
    
    gold = GoldPredictor()
    try:
        metrics = gold.train()
        print("\nGold Metrics:")
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"Gold training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    retrain()
