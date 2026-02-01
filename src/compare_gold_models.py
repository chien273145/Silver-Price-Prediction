"""
Gold Price Model Comparison
Compare multiple ML models to find the best predictor.
"""
import sys
sys.path.insert(0, '.')
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

print("=" * 70)
print("🥇 GOLD PRICE MODEL COMPARISON")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"XGBoost available: {HAS_XGB}")
print(f"LightGBM available: {HAS_LGBM}")

# Load data
# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'gold_geopolitical_dataset.csv')
print(f"\nLoading data from {data_path}...")
data = pd.read_csv(data_path)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)
data['price'] = data['gold_close']
print(f"✓ Loaded {len(data)} rows")

# Define features
price_features = ['gold_open', 'gold_high', 'gold_low', 'gold_close']
silver_features = ['silver_close', 'gs_ratio', 'gs_ratio_ma7', 'gs_ratio_ma30']
gpr_features = [
    'gpr_level', 'gpr_war', 'gpr_terrorism', 'gpr_economic', 
    'gpr_political', 'gpr_nuclear', 'gpr_trade', 'gpr_sanctions',
    'gpr_ma7', 'gpr_ma30', 'gpr_ma90', 'gpr_momentum', 'gpr_volatility', 'gpr_spike'
]
pandemic_features = [
    'pandemic_severity', 'pandemic_phase_encoded', 
    'is_pandemic', 'is_lockdown', 'is_post_pandemic'
]
technical_features = [
    'gold_lag1', 'gold_lag7', 'gold_lag14', 'gold_lag30',
    'gold_ma7', 'gold_ma14', 'gold_ma30', 'gold_ma60',
    'gold_ema7', 'gold_ema14', 'gold_ema30',
    'gold_rsi', 'gold_macd', 'gold_bb_pct', 'gold_volatility', 'gold_return'
]
time_features = ['day_of_week', 'month', 'quarter', 'year']
risk_features = ['composite_risk', 'risk_regime_encoded', 'event_count']

all_features = (price_features + silver_features + gpr_features + 
                pandemic_features + technical_features + time_features + risk_features)
feature_columns = [col for col in all_features if col in data.columns]
print(f"✓ Using {len(feature_columns)} features")

# Handle missing values
data[feature_columns] = data[feature_columns].ffill().fillna(0)

# Prepare data
X = data[feature_columns].values
# Target: 7-day ahead prediction
data['target_7'] = data['price'].shift(-7)
valid_mask = ~data['target_7'].isna()
X = X[valid_mask]
y = data['target_7'].values[valid_mask]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"✓ PCA: {X_scaled.shape[1]} -> {X_pca.shape[1]} components")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, shuffle=False)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Define models
models = {
    'Ridge': Ridge(alpha=1.0),
    'BayesianRidge': BayesianRidge(),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, epsilon=0.1),
}

if HAS_XGB:
    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0)
if HAS_LGBM:
    models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)

# Train and evaluate
print("\n" + "=" * 70)
print(f"TRAINING & EVALUATING {len(models)} MODELS (7-day prediction)")
print("=" * 70)

results = []

for name, model in models.items():
    print(f"\n🔄 Training {name}...", end=" ", flush=True)
    start = datetime.now()
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Metrics
        r2 = model.score(X_test, y_test)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        
        train_time = (datetime.now() - start).total_seconds()
        
        results.append({
            'Model': name,
            'R²': r2,
            'MAPE (%)': mape,
            'RMSE ($)': rmse,
            'MAE ($)': mae,
            'Time (s)': train_time
        })
        
        print(f"✓ R²={r2:.4f}, MAPE={mape:.2f}%, Time={train_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Summary table
print("\n" + "=" * 70)
print("📊 RESULTS SUMMARY (sorted by R²)")
print("=" * 70)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('R²', ascending=False).reset_index(drop=True)

print(f"\n{'Rank':<6} {'Model':<18} {'R²':<10} {'MAPE(%)':<10} {'RMSE($)':<12} {'MAE($)':<12}")
print("-" * 70)
for i, row in df_results.iterrows():
    rank = i + 1
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
    print(f"{medal} {rank:<4} {row['Model']:<18} {row['R²']:<10.4f} {row['MAPE (%)']:<10.2f} {row['RMSE ($)']:<12.2f} {row['MAE ($)']:<12.2f}")

# Best model
best = df_results.iloc[0]
print("\n" + "=" * 70)
print(f"🏆 BEST MODEL: {best['Model']}")
print(f"   R² Score: {best['R²']:.4f}")
print(f"   MAPE: {best['MAPE (%)']:.2f}%")
print(f"   RMSE: ${best['RMSE ($)']:.2f}")
print(f"   MAE: ${best['MAE ($)']:.2f}")
print("=" * 70)

# Save results
# Save results
results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gold_model_comparison.csv')
df_results.to_csv(results_path, index=False)
print(f"\n✓ Results saved to {results_path}")
