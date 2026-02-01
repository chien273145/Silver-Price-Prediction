"""Evaluate Gold Price Prediction Model"""
import sys
sys.path.insert(0, '.')
from src.gold_predictor import GoldPredictor
import numpy as np

print("=" * 60)
print("GOLD MODEL EVALUATION")
print("=" * 60)

p = GoldPredictor()
p.load_data()
p.create_features()
p.load_model()

# Display metrics
metrics = p.metrics
print("\n--- Training Metrics (per prediction day) ---")
print(f"{'Day':<6} {'R²':<12} {'MAPE (%)':<12}")
print("-" * 30)

for day in range(1, 8):
    r2 = metrics['r2'].get(day, 0)
    mape = metrics['mape'].get(day, 0)
    print(f"{day:<6} {r2:<12.4f} {mape:<12.2f}")

avg_r2 = np.mean(list(metrics['r2'].values()))
avg_mape = np.mean(list(metrics['mape'].values()))

print("-" * 30)
print(f"{'AVG':<6} {avg_r2:<12.4f} {avg_mape:<12.2f}")

print("\n--- Model Info ---")
info = p.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Features: {info['features']}")
print(f"PCA components: {info['pca_components']}")

# Rating
print("\n--- Model Quality Rating ---")
if avg_r2 >= 0.95 and avg_mape <= 5:
    print("✅ EXCELLENT - Model is highly accurate")
elif avg_r2 >= 0.9 and avg_mape <= 10:
    print("✅ GOOD - Model performs well")
elif avg_r2 >= 0.8:
    print("⚠️ ACCEPTABLE - Model is usable but could be improved")
else:
    print("❌ POOR - Model needs improvement")
