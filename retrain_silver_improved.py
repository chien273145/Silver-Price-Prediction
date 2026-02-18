"""
retrain_silver_improved.py
==========================
Retrain Silver model directly with improved hyperparameters.
Uses EnhancedPredictor but overrides XGBoost params directly in the source.
"""
import os, sys, time
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print(f"[{datetime.now():%H:%M:%S}] Starting Silver retrain with improved hyperparameters...")

from src.enhanced_predictor import EnhancedPredictor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print(f"[{datetime.now():%H:%M:%S}] XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print(f"[{datetime.now():%H:%M:%S}] XGBoost NOT available - Ridge only")

# ─── Load data ────────────────────────────────────────────────────────────────
predictor = EnhancedPredictor(sequence_length=60, prediction_days=7)
predictor.lstm_sequence_length = 45
predictor.use_lstm = False  # Skip LSTM (no TF on this machine)

print(f"[{datetime.now():%H:%M:%S}] Loading data...")
predictor.load_data()
print(f"[{datetime.now():%H:%M:%S}] Data loaded: {len(predictor.data)} rows")

# ─── Prepare features ─────────────────────────────────────────────────────────
X = predictor.data[predictor.feature_columns].values
X_xgb_raw = predictor.data[predictor.xgb_feature_columns].values if predictor.xgb_feature_columns else X

# Targets (7 days ahead)
y_targets = []
for day in range(1, predictor.prediction_days + 1):
    target = predictor.data["price"].shift(-day).values
    y_targets.append(target)

valid_mask = ~np.isnan(y_targets[-1])
X = X[valid_mask]
X_xgb_raw = X_xgb_raw[valid_mask]
y_targets = [y[valid_mask] for y in y_targets]
base_prices = predictor.data["price"].values[valid_mask]

# Return targets for XGBoost
xgb_return_targets = []
for day in range(predictor.prediction_days):
    returns = (y_targets[day] - base_prices) / (base_prices + 1e-10)
    xgb_return_targets.append(returns)

# ─── Scale ────────────────────────────────────────────────────────────────────
predictor.scaler = StandardScaler()
X_scaled = predictor.scaler.fit_transform(X)

predictor.xgb_scaler = StandardScaler()
X_xgb_scaled = predictor.xgb_scaler.fit_transform(X_xgb_raw)

# PCA for Ridge
print(f"[{datetime.now():%H:%M:%S}] Applying PCA...")
predictor.pca = PCA(n_components=0.95, svd_solver="full")
X_for_ridge = predictor.pca.fit_transform(X_scaled)
n_comp = predictor.pca.n_components_
print(f"  Reduced to {n_comp} PCA components")

# Scale targets
y_all = np.column_stack(y_targets)
predictor.target_scaler = MinMaxScaler()
y_scaled = predictor.target_scaler.fit_transform(y_all)

# Train/test split
test_size = 0.2
split_idx = int(len(X_for_ridge) * (1 - test_size))
X_train_r, X_test_r = X_for_ridge[:split_idx], X_for_ridge[split_idx:]
X_train_xgb, X_test_xgb = X_xgb_scaled[:split_idx], X_xgb_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
base_prices_test = base_prices[split_idx:]

# ─── Train Ridge ──────────────────────────────────────────────────────────────
print(f"[{datetime.now():%H:%M:%S}] Training Ridge models...")
predictor.models = []
for day in range(predictor.prediction_days):
    model = Ridge(alpha=1.0)
    model.fit(X_train_r, y_train[:, day])
    predictor.models.append(model)

y_pred_ridge_scaled = np.column_stack([m.predict(X_test_r) for m in predictor.models])
y_pred_ridge_prices = predictor.target_scaler.inverse_transform(y_pred_ridge_scaled)
y_test_prices = predictor.target_scaler.inverse_transform(y_test)

# ─── Train XGBoost ────────────────────────────────────────────────────────────
predictor.xgb_models = []
has_xgb = False

if XGBOOST_AVAILABLE:
    print(f"[{datetime.now():%H:%M:%S}] Training XGBoost models (n_estimators=1200)...")
    xgb_price_preds = np.zeros_like(y_pred_ridge_prices)

    for day in range(predictor.prediction_days):
        y_xgb_train = xgb_return_targets[day][:split_idx]
        y_xgb_test  = xgb_return_targets[day][split_idx:]

        xgb_model = XGBRegressor(
            n_estimators=1200,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.5,
            gamma=0.05,
            max_delta_step=1,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            early_stopping_rounds=40,
        )
        xgb_model.fit(
            X_train_xgb, y_xgb_train,
            eval_set=[(X_test_xgb, y_xgb_test)],
            verbose=False,
        )
        predictor.xgb_models.append(xgb_model)

        xgb_return_pred = xgb_model.predict(X_test_xgb)
        xgb_price_preds[:, day] = base_prices_test * (1 + xgb_return_pred)

        ridge_mape = np.mean(np.abs((y_test_prices[:, day] - y_pred_ridge_prices[:, day]) / y_test_prices[:, day])) * 100
        xgb_mape   = np.mean(np.abs((y_test_prices[:, day] - xgb_price_preds[:, day])    / y_test_prices[:, day])) * 100
        print(f"  Day {day+1}: Ridge MAPE={ridge_mape:.2f}%, XGB MAPE={xgb_mape:.2f}%")

    has_xgb = True

# ─── Optimize ensemble weights ────────────────────────────────────────────────
if has_xgb:
    print(f"[{datetime.now():%H:%M:%S}] Optimizing ensemble weights...")
    best_w, best_mape = (1.0, 0.0, 0.0), float("inf")
    step = 0.1
    for w_r in np.arange(0, 1.0 + step / 2, step):
        for w_x in np.arange(0, 1.0 - w_r + step / 2, step):
            w_l = round(1.0 - w_r - w_x, 2)
            if w_l < -0.01:
                continue
            w_l = max(w_l, 0.0)
            y_blend = w_r * y_pred_ridge_prices + w_x * xgb_price_preds
            candidate_mape = np.mean(np.abs((y_test_prices - y_blend) / (y_test_prices + 1e-10))) * 100
            if candidate_mape < best_mape:
                best_mape = candidate_mape
                best_w = (round(w_r, 2), round(w_x, 2), round(w_l, 2))

    predictor.ensemble_weights = {"ridge": best_w[0], "xgboost": best_w[1], "lstm": best_w[2]}
    w_r, w_x, w_l = best_w
    y_pred_prices = w_r * y_pred_ridge_prices + w_x * xgb_price_preds
    print(f"  Optimal weights: Ridge={w_r}, XGB={w_x}, LSTM={w_l}")
else:
    predictor.ensemble_weights = {"ridge": 1.0, "xgboost": 0.0, "lstm": 0.0}
    y_pred_prices = y_pred_ridge_prices

# ─── Final metrics ────────────────────────────────────────────────────────────
rmse = np.sqrt(np.mean((y_test_prices - y_pred_prices) ** 2))
mae  = np.mean(np.abs(y_test_prices - y_pred_prices))
mape = np.mean(np.abs((y_test_prices - y_pred_prices) / (y_test_prices + 1e-10))) * 100
ss_res = np.sum((y_test_prices - y_pred_prices) ** 2)
ss_tot = np.sum((y_test_prices - np.mean(y_test_prices)) ** 2)
avg_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

print(f"\n[{datetime.now():%H:%M:%S}] === SILVER FINAL METRICS ===")
print(f"  R2:   {avg_r2:.4f}")
print(f"  RMSE: ${rmse:.4f}")
print(f"  MAE:  ${mae:.4f}")
print(f"  MAPE: {mape:.2f}%")

predictor.latest_metrics = {
    "avg_r2": avg_r2, "rmse": rmse, "mae": mae, "mape": mape,
    "ensemble": has_xgb,
}
predictor.use_ensemble = has_xgb
predictor.use_lstm = False

# ─── Save ─────────────────────────────────────────────────────────────────────
predictor.save_model()
print(f"[{datetime.now():%H:%M:%S}] Silver model saved!")

# ─── Quick prediction test ────────────────────────────────────────────────────
print(f"\n[{datetime.now():%H:%M:%S}] Testing prediction...")
result = predictor.predict(in_vnd=True)
preds = result.get("predictions", [])
for p in preds[:3]:
    print(f"  {p['date']}: {p['predicted_price']:,.0f} VND/luong ({p['change_percent']:+.2f}%)")
print("Done!")
