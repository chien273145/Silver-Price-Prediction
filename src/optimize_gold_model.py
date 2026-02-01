"""
Gold Model Optimization - Test All Methods
"""
import sys
sys.path.insert(0, '.')
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🥇 GOLD MODEL OPTIMIZATION - Testing All Methods")
print("=" * 70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'gold_geopolitical_dataset.csv')
print(f"\nLoading data...")
data = pd.read_csv(data_path)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)
data['price'] = data['gold_close']

# Base features
base_features = [
    'gold_open', 'gold_high', 'gold_low', 'gold_close',
    'silver_close', 'gs_ratio', 'gs_ratio_ma7', 'gs_ratio_ma30',
    'gpr_level', 'gpr_war', 'gpr_terrorism', 'gpr_economic', 
    'gpr_political', 'gpr_nuclear', 'gpr_trade', 'gpr_sanctions',
    'gpr_ma7', 'gpr_ma30', 'gpr_ma90', 'gpr_momentum', 'gpr_volatility', 'gpr_spike',
    'pandemic_severity', 'pandemic_phase_encoded', 
    'is_pandemic', 'is_lockdown', 'is_post_pandemic',
    'gold_lag1', 'gold_lag7', 'gold_lag14', 'gold_lag30',
    'gold_ma7', 'gold_ma14', 'gold_ma30', 'gold_ma60',
    'gold_ema7', 'gold_ema14', 'gold_ema30',
    'gold_rsi', 'gold_macd', 'gold_bb_pct', 'gold_volatility', 'gold_return',
    'day_of_week', 'month', 'quarter', 'year',
    'composite_risk', 'risk_regime_encoded', 'event_count'
]
feature_columns = [col for col in base_features if col in data.columns]

# Handle missing values
data[feature_columns] = data[feature_columns].ffill().fillna(0)

# Prepare data
X = data[feature_columns].values
data['target_7'] = data['price'].shift(-7)
valid_mask = ~data['target_7'].isna()
X = X[valid_mask]
y = data['target_7'].values[valid_mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"✓ Loaded {len(data)} rows, {len(feature_columns)} features")
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")

# Results storage
results = []

def evaluate(name, y_pred, y_true, train_time):
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    results.append({'Method': name, 'R²': r2, 'MAPE (%)': mape, 'RMSE ($)': rmse, 'MAE ($)': mae, 'Time (s)': train_time})
    print(f"   R²={r2:.4f}, MAPE={mape:.2f}%")
    return r2

# ============================================================
# 0. BASELINE - Current Ridge (for comparison)
# ============================================================
print("\n" + "=" * 70)
print("📊 0. BASELINE - Current Ridge Model")
print("=" * 70)

start = datetime.now()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

baseline_model = Ridge(alpha=1.0)
baseline_model.fit(X_train_pca, y_train)
baseline_pred = baseline_model.predict(X_test_pca)
baseline_r2 = evaluate("Baseline Ridge", baseline_pred, y_test, (datetime.now() - start).total_seconds())

# ============================================================
# 1. HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 70)
print("🔧 1. HYPERPARAMETER TUNING (GridSearch for Ridge alpha)")
print("=" * 70)

start = datetime.now()
param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_pca, y_train)
best_alpha = grid_search.best_params_['alpha']
print(f"   Best alpha: {best_alpha}")

tuned_model = Ridge(alpha=best_alpha)
tuned_model.fit(X_train_pca, y_train)
tuned_pred = tuned_model.predict(X_test_pca)
tuned_r2 = evaluate("Tuned Ridge", tuned_pred, y_test, (datetime.now() - start).total_seconds())

# ============================================================
# 2. FEATURE ENGINEERING (More lags and interactions)
# ============================================================
print("\n" + "=" * 70)
print("🔧 2. FEATURE ENGINEERING (More lags + Momentum features)")
print("=" * 70)

start = datetime.now()

# Create extended features
data_ext = data.copy()

# Add more lag features
for lag in [2, 3, 5, 10, 21, 60, 90]:
    if f'gold_lag{lag}' not in data_ext.columns:
        data_ext[f'gold_lag{lag}'] = data_ext['price'].shift(lag)

# Add momentum features
data_ext['momentum_3'] = data_ext['price'] - data_ext['price'].shift(3)
data_ext['momentum_7'] = data_ext['price'] - data_ext['price'].shift(7)
data_ext['momentum_14'] = data_ext['price'] - data_ext['price'].shift(14)
data_ext['momentum_30'] = data_ext['price'] - data_ext['price'].shift(30)

# Add rate of change
data_ext['roc_7'] = (data_ext['price'] - data_ext['price'].shift(7)) / data_ext['price'].shift(7)
data_ext['roc_14'] = (data_ext['price'] - data_ext['price'].shift(14)) / data_ext['price'].shift(14)
data_ext['roc_30'] = (data_ext['price'] - data_ext['price'].shift(30)) / data_ext['price'].shift(30)

# Add rolling stats
data_ext['rolling_std_7'] = data_ext['price'].rolling(7).std()
data_ext['rolling_std_14'] = data_ext['price'].rolling(14).std()
data_ext['rolling_std_30'] = data_ext['price'].rolling(30).std()
data_ext['rolling_min_7'] = data_ext['price'].rolling(7).min()
data_ext['rolling_max_7'] = data_ext['price'].rolling(7).max()

# Extended feature list
ext_features = feature_columns + [
    'gold_lag2', 'gold_lag3', 'gold_lag5', 'gold_lag10', 'gold_lag21', 'gold_lag60', 'gold_lag90',
    'momentum_3', 'momentum_7', 'momentum_14', 'momentum_30',
    'roc_7', 'roc_14', 'roc_30',
    'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
    'rolling_min_7', 'rolling_max_7'
]
ext_features = [f for f in ext_features if f in data_ext.columns]
print(f"   Extended features: {len(ext_features)}")

# Fill missing
data_ext[ext_features] = data_ext[ext_features].ffill().fillna(0)

# Prepare data
X_ext = data_ext[ext_features].values
data_ext['target_7'] = data_ext['price'].shift(-7)
valid_mask_ext = ~data_ext['target_7'].isna()
X_ext = X_ext[valid_mask_ext]
y_ext = data_ext['target_7'].values[valid_mask_ext]

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_ext, y_ext, test_size=0.2, shuffle=False)

scaler_ext = StandardScaler()
X_train_ext_scaled = scaler_ext.fit_transform(X_train_ext)
X_test_ext_scaled = scaler_ext.transform(X_test_ext)

pca_ext = PCA(n_components=0.95)
X_train_ext_pca = pca_ext.fit_transform(X_train_ext_scaled)
X_test_ext_pca = pca_ext.transform(X_test_ext_scaled)
print(f"   PCA components: {X_train_ext_pca.shape[1]}")

fe_model = Ridge(alpha=best_alpha)
fe_model.fit(X_train_ext_pca, y_train_ext)
fe_pred = fe_model.predict(X_test_ext_pca)
fe_r2 = evaluate("Feature Engineering", fe_pred, y_test_ext, (datetime.now() - start).total_seconds())

# ============================================================
# 3. POLYNOMIAL FEATURES
# ============================================================
print("\n" + "=" * 70)
print("🔧 3. POLYNOMIAL FEATURES (Degree 2)")
print("=" * 70)

start = datetime.now()

# Use smaller feature set for polynomial (to avoid memory issues)
poly_features = ['gold_close', 'gold_lag1', 'gold_lag7', 'gold_ma7', 'gold_ma14', 
                 'gpr_level', 'gold_rsi', 'gold_volatility', 'composite_risk']
poly_features = [f for f in poly_features if f in data.columns]

X_poly_base = data[poly_features].values[valid_mask]
X_train_poly, X_test_poly, _, _ = train_test_split(X_poly_base, y, test_size=0.2, shuffle=False)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2 = poly.fit_transform(X_train_poly)
X_test_poly2 = poly.transform(X_test_poly)
print(f"   Polynomial features: {X_train_poly2.shape[1]}")

scaler_poly = StandardScaler()
X_train_poly2_scaled = scaler_poly.fit_transform(X_train_poly2)
X_test_poly2_scaled = scaler_poly.transform(X_test_poly2)

poly_model = Ridge(alpha=10.0)  # Higher alpha for regularization with many features
poly_model.fit(X_train_poly2_scaled, y_train)
poly_pred = poly_model.predict(X_test_poly2_scaled)
poly_r2 = evaluate("Polynomial Features", poly_pred, y_test, (datetime.now() - start).total_seconds())

# ============================================================
# 4. STACKING ENSEMBLE
# ============================================================
print("\n" + "=" * 70)
print("🔧 4. STACKING ENSEMBLE (Ridge + Lasso + BayesianRidge)")
print("=" * 70)

start = datetime.now()

estimators = [
    ('ridge', Ridge(alpha=best_alpha)),
    ('lasso', Lasso(alpha=0.1)),
    ('bayesian', BayesianRidge())
]
stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)
stacking_model.fit(X_train_pca, y_train)
stacking_pred = stacking_model.predict(X_test_pca)
stack_r2 = evaluate("Stacking Ensemble", stacking_pred, y_test, (datetime.now() - start).total_seconds())

# ============================================================
# 5. TIME SERIES CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("🔧 5. TIME SERIES CROSS-VALIDATION (5-Fold)")
print("=" * 70)

start = datetime.now()

tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_idx, val_idx in tscv.split(X_train_pca):
    X_cv_train, X_cv_val = X_train_pca[train_idx], X_train_pca[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
    
    cv_model = Ridge(alpha=best_alpha)
    cv_model.fit(X_cv_train, y_cv_train)
    cv_scores.append(cv_model.score(X_cv_val, y_cv_val))

print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean CV R²: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Final model with best alpha from TS-CV
tscv_model = Ridge(alpha=best_alpha)
tscv_model.fit(X_train_pca, y_train)
tscv_pred = tscv_model.predict(X_test_pca)
tscv_r2 = evaluate("TS Cross-Validation", tscv_pred, y_test, (datetime.now() - start).total_seconds())

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("📊 OPTIMIZATION RESULTS SUMMARY")
print("=" * 70)

df_results = pd.DataFrame(results)
df_results['Improvement'] = ((df_results['R²'] - baseline_r2) / baseline_r2 * 100).round(2)
df_results = df_results.sort_values('R²', ascending=False).reset_index(drop=True)

print(f"\n{'Rank':<6} {'Method':<25} {'R²':<10} {'MAPE(%)':<10} {'Improve%':<10}")
print("-" * 70)
for i, row in df_results.iterrows():
    rank = i + 1
    medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
    improve = f"+{row['Improvement']:.2f}%" if row['Improvement'] > 0 else f"{row['Improvement']:.2f}%"
    print(f"{medal} {rank:<4} {row['Method']:<25} {row['R²']:<10.4f} {row['MAPE (%)']:<10.2f} {improve:<10}")

# Best method
best = df_results.iloc[0]
print("\n" + "=" * 70)
print(f"🏆 BEST METHOD: {best['Method']}")
print(f"   R² Score: {best['R²']:.4f}")
print(f"   MAPE: {best['MAPE (%)']:.2f}%")
print(f"   Improvement over baseline: {best['Improvement']:+.2f}%")
print("=" * 70)

# Save results
# Save results
results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gold_optimization_results.csv')
df_results.to_csv(results_path, index=False)
print(f"\n✓ Results saved to {results_path}")
