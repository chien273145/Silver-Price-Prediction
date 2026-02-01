# -*- coding: utf-8 -*-
"""
Overfitting Check for Gold Model
Tests: Train vs Test gap, Walk-forward validation, Learning curves
"""
import sys
sys.path.insert(0, '.')
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OVERFITTING CHECK FOR GOLD MODEL")
print("=" * 70)

# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'gold_geopolitical_dataset.csv')
data = pd.read_csv(data_path)
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date').reset_index(drop=True)
data['price'] = data['gold_close']

# Features used in polynomial model
poly_features = ['gold_close', 'gold_lag1', 'gold_lag7', 'gold_ma7', 'gold_ma14', 
                 'gpr_level', 'gold_rsi', 'gold_volatility', 'composite_risk']
poly_features = [f for f in poly_features if f in data.columns]

# Handle missing
data[poly_features] = data[poly_features].ffill().fillna(0)

# Prepare data
X = data[poly_features].values
data['target_7'] = data['price'].shift(-7)
valid_mask = ~data['target_7'].isna()
X = X[valid_mask]
y = data['target_7'].values[valid_mask]

print(f"Data: {len(X)} samples, {len(poly_features)} base features")

# ============================================================
# TEST 1: TRAIN vs TEST SCORE COMPARISON
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: TRAIN vs TEST SCORE (Overfitting Gap)")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

model = Ridge(alpha=10.0)
model.fit(X_train_scaled, y_train)

train_r2 = model.score(X_train_scaled, y_train)
test_r2 = model.score(X_test_scaled, y_test)
gap = train_r2 - test_r2

print(f"   Train R2: {train_r2:.4f}")
print(f"   Test R2:  {test_r2:.4f}")
print(f"   Gap:      {gap:.4f}")

if gap > 0.05:
    print("   [!] WARNING: Large gap indicates OVERFITTING!")
else:
    print("   [OK] Gap is acceptable")

# ============================================================
# TEST 2: WALK-FORWARD VALIDATION (Time Series proper)
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: WALK-FORWARD VALIDATION (5 folds)")
print("=" * 70)

n_samples = len(X)
fold_size = n_samples // 6

wf_train_scores = []
wf_test_scores = []

for i in range(5):
    train_end = fold_size * (i + 2)
    test_start = train_end
    test_end = test_start + fold_size
    
    if test_end > n_samples:
        break
    
    X_wf_train = X[:train_end]
    y_wf_train = y[:train_end]
    X_wf_test = X[test_start:test_end]
    y_wf_test = y[test_start:test_end]
    
    poly_wf = PolynomialFeatures(degree=2, include_bias=False)
    X_wf_train_poly = poly_wf.fit_transform(X_wf_train)
    X_wf_test_poly = poly_wf.transform(X_wf_test)
    
    scaler_wf = StandardScaler()
    X_wf_train_scaled = scaler_wf.fit_transform(X_wf_train_poly)
    X_wf_test_scaled = scaler_wf.transform(X_wf_test_poly)
    
    model_wf = Ridge(alpha=10.0)
    model_wf.fit(X_wf_train_scaled, y_wf_train)
    
    train_score = model_wf.score(X_wf_train_scaled, y_wf_train)
    test_score = model_wf.score(X_wf_test_scaled, y_wf_test)
    
    wf_train_scores.append(train_score)
    wf_test_scores.append(test_score)
    
    print(f"   Fold {i+1}: Train R2={train_score:.4f}, Test R2={test_score:.4f}, Gap={train_score - test_score:.4f}")

avg_wf_gap = np.mean(np.array(wf_train_scores) - np.array(wf_test_scores))
avg_wf_test = np.mean(wf_test_scores)
print(f"\n   Average Walk-Forward Gap: {avg_wf_gap:.4f}")
print(f"   Average Test R2: {avg_wf_test:.4f}")

if avg_wf_gap > 0.1:
    print("   [!] WARNING: Significant overfitting detected!")
elif avg_wf_gap > 0.05:
    print("   [!] CAUTION: Moderate overfitting present")
else:
    print("   [OK] Model generalizes well")

# ============================================================
# TEST 3: DIFFERENT ALPHA VALUES (Regularization check)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: REGULARIZATION SENSITIVITY (Alpha sweep)")
print("=" * 70)

alphas = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
best_alpha = None
best_gap = float('inf')
best_test_r2 = 0

print(f"   {'Alpha':<10} {'Train R2':<12} {'Test R2':<12} {'Gap':<10} {'Status'}")
print("   " + "-" * 55)

for alpha in alphas:
    model_reg = Ridge(alpha=alpha)
    model_reg.fit(X_train_scaled, y_train)
    
    train_r2_reg = model_reg.score(X_train_scaled, y_train)
    test_r2_reg = model_reg.score(X_test_scaled, y_test)
    gap_reg = train_r2_reg - test_r2_reg
    
    status = "[OK]" if gap_reg < 0.03 else ("[!]" if gap_reg < 0.05 else "[X]")
    print(f"   {alpha:<10} {train_r2_reg:<12.4f} {test_r2_reg:<12.4f} {gap_reg:<10.4f} {status}")
    
    if gap_reg < best_gap and test_r2_reg > 0.80:
        best_gap = gap_reg
        best_alpha = alpha
        best_test_r2 = test_r2_reg

print(f"\n   BEST Alpha for low overfitting: {best_alpha}")
print(f"   Test R2: {best_test_r2:.4f}, Gap: {best_gap:.4f}")

# ============================================================
# TEST 4: COMPARE WITH SIMPLER MODEL (No polynomial)
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: SIMPLER MODEL COMPARISON (No polynomial)")
print("=" * 70)

scaler_simple = StandardScaler()
X_train_simple = scaler_simple.fit_transform(X_train)
X_test_simple = scaler_simple.transform(X_test)

model_simple = Ridge(alpha=1.0)
model_simple.fit(X_train_simple, y_train)

train_r2_simple = model_simple.score(X_train_simple, y_train)
test_r2_simple = model_simple.score(X_test_simple, y_test)
gap_simple = train_r2_simple - test_r2_simple

print(f"   Simple model (no poly): Train R2={train_r2_simple:.4f}, Test R2={test_r2_simple:.4f}, Gap={gap_simple:.4f}")
print(f"   Polynomial model:       Train R2={train_r2:.4f}, Test R2={test_r2:.4f}, Gap={gap:.4f}")

# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

if avg_wf_gap > 0.1 or gap > 0.1:
    print("""
   [!] OVERFITTING DETECTED!
   
   Recommendations:
   1. Use higher regularization (alpha >= 100)
   2. Reduce polynomial degree to 1 (linear only)
   3. Use Feature Engineering method instead (R2=0.91, less overfitting)
   4. Consider simpler model with R2 ~0.90
""")
    recommended = "Feature Engineering (R2=0.91)"
else:
    print("""
   [OK] Model appears to generalize well
   
   However, for production use:
   1. Monitor prediction accuracy over time
   2. Retrain periodically with new data  
   3. Use walk-forward validation for hyperparameter tuning
""")
    recommended = f"Polynomial Features with alpha={best_alpha} (R2={best_test_r2:.2f})"

print(f"   RECOMMENDED: {recommended}")
print("=" * 70)
