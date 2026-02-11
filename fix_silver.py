"""Clean Silver data, retrain, and save — all in one."""
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd

# Step 1: Clean dataset
path = os.path.join('dataset', 'dataset_enhanced.csv')
df = pd.read_csv(path)
print(f"Before: {len(df)} rows, max Silver=${df['Silver_Close'].max():.2f}")

# Remove Silver > $50 (2026 fake data)
clean = df[df['Silver_Close'] <= 50].copy()
# Also remove Gold > $3500 if present
if 'Gold_Close' in clean.columns:
    clean = clean[clean['Gold_Close'] <= 3500].copy()

clean.to_csv(path, index=False)
print(f"After:  {len(clean)} rows, max Silver=${clean['Silver_Close'].max():.2f}")
print(f"  Last date: {clean['Date'].iloc[-1]}, Last price: ${clean['Silver_Close'].iloc[-1]:.2f}")

# Step 2: Retrain Silver
from src.enhanced_predictor import EnhancedPredictor

p = EnhancedPredictor()
p.load_data()
print(f"\nLoaded {len(p.data)} rows, last price=${p.data['price'].iloc[-1]:.2f}")

metrics = p.train()
print(f"\nMetrics: {metrics}")

# Step 3: Save
p.save_model()
print("\n[SAVED] Silver model saved!")

# Verify
import datetime
model_path = os.path.join(p.model_dir, 'ridge_enhanced_models.pkl')
mtime = os.path.getmtime(model_path)
print(f"Model file updated: {datetime.datetime.fromtimestamp(mtime)}")
