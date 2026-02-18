"""
update_data.py
==============
Script cập nhật dataset hàng ngày và retrain mô hình.
Chạy bởi GitHub Actions mỗi ngày lúc 01:00 UTC (08:00 SA Việt Nam).

Usage:
    python update_data.py           # Update data only
    python update_data.py --retrain # Update data + retrain models
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA UPDATE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def update_silver_dataset() -> bool:
    """
    Fetch latest Silver, Gold, DXY, VIX data and append to dataset_enhanced.csv.
    Returns True if new data was added.
    """
    path = os.path.join(DATASET_DIR, "dataset_enhanced.csv")
    log(f"[Silver] Reading {path}...")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    today = datetime.now()
    gap = (today - last_date).days

    log(f"[Silver] Last date: {last_date.date()}, gap: {gap} days")

    if gap <= 1:
        log("[Silver] Already up to date. Skipping.")
        return False

    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"[Silver] Fetching {fetch_start} to {fetch_end}...")

    try:
        tickers = yf.download(
            ["SI=F", "GC=F", "DX-Y.NYB", "^VIX"],
            start=fetch_start,
            end=fetch_end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        log(f"[Silver] yfinance error: {e}")
        return False

    if tickers.empty:
        log("[Silver] No data returned from Yahoo Finance.")
        return False

    # Handle multi-level columns
    if isinstance(tickers.columns, pd.MultiIndex):
        close = tickers["Close"]
        high  = tickers["High"]
        low   = tickers["Low"]
        open_ = tickers["Open"]
    else:
        close = tickers[["Close"]]
        high  = tickers[["High"]]
        low   = tickers[["Low"]]
        open_ = tickers[["Open"]]

    new_rows = []
    for date_idx in close.index:
        row_date = pd.Timestamp(date_idx)
        if row_date <= last_date:
            continue

        def get(df_col, ticker):
            try:
                v = df_col[ticker].get(date_idx, np.nan)
                return float(v) if not pd.isna(v) else np.nan
            except Exception:
                return np.nan

        silver_c = get(close, "SI=F")
        silver_h = get(high,  "SI=F")
        silver_l = get(low,   "SI=F")
        silver_o = get(open_, "SI=F")
        gold_c   = get(close, "GC=F")
        dxy_c    = get(close, "DX-Y.NYB")
        vix_c    = get(close, "^VIX")

        # Skip invalid data
        if np.isnan(silver_c) or np.isnan(gold_c):
            continue
        if not (5 < silver_c < 100):
            log(f"[Silver] Skipping {row_date.date()}: silver={silver_c} out of range")
            continue
        if not (1000 < gold_c < 4000):
            log(f"[Silver] Skipping {row_date.date()}: gold={gold_c} out of range")
            continue

        # Use last known values as fallback
        last_dxy = float(df["DXY"].iloc[-1]) if "DXY" in df.columns else 104.0
        last_vix = float(df["VIX"].iloc[-1]) if "VIX" in df.columns else 20.0

        new_rows.append({
            "Date":         str(row_date.date()),
            "Silver_Close": silver_c,
            "Silver_Open":  silver_o if not np.isnan(silver_o) else silver_c,
            "Silver_High":  silver_h if not np.isnan(silver_h) else silver_c,
            "Silver_Low":   silver_l if not np.isnan(silver_l) else silver_c,
            "Gold":         gold_c,
            "DXY":          dxy_c if not np.isnan(dxy_c) else last_dxy,
            "VIX":          vix_c if not np.isnan(vix_c) else last_vix,
            "price":        silver_c,
            "date":         str(row_date.date()),
            "high":         silver_h if not np.isnan(silver_h) else silver_c,
            "low":          silver_l if not np.isnan(silver_l) else silver_c,
        })

    if not new_rows:
        log("[Silver] No valid new rows found.")
        return False

    new_df = pd.DataFrame(new_rows)
    # Align columns
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df.reindex(columns=df.columns)

    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(path, index=False)
    log(f"[Silver] Added {len(new_rows)} rows. Total: {len(df)}")
    return True


def update_gold_world_dataset() -> bool:
    """
    Fetch latest Gold world data and append to gold_geopolitical_dataset.csv.
    Returns True if new data was added.
    """
    path = os.path.join(DATASET_DIR, "gold_geopolitical_dataset.csv")
    log(f"[Gold] Reading {path}...")

    df = pd.read_csv(path, parse_dates=["date"])
    last_date = df["date"].max()
    today = datetime.now()
    gap = (today - last_date).days

    log(f"[Gold] Last date: {last_date.date()}, gap: {gap} days")

    if gap <= 1:
        log("[Gold] Already up to date. Skipping.")
        return False

    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"[Gold] Fetching {fetch_start} to {fetch_end}...")

    try:
        tickers = yf.download(
            ["GC=F", "SI=F", "DX-Y.NYB", "^VIX"],
            start=fetch_start,
            end=fetch_end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        log(f"[Gold] yfinance error: {e}")
        return False

    if tickers.empty:
        log("[Gold] No data returned from Yahoo Finance.")
        return False

    if isinstance(tickers.columns, pd.MultiIndex):
        close = tickers["Close"]
        high  = tickers["High"]
        low   = tickers["Low"]
        open_ = tickers["Open"]
    else:
        close = tickers[["Close"]]
        high  = tickers[["High"]]
        low   = tickers[["Low"]]
        open_ = tickers[["Open"]]

    new_rows = []
    last_row = df.iloc[-1]

    for date_idx in close.index:
        row_date = pd.Timestamp(date_idx)
        if row_date <= last_date:
            continue

        def get(df_col, ticker):
            try:
                v = df_col[ticker].get(date_idx, np.nan)
                return float(v) if not pd.isna(v) else np.nan
            except Exception:
                return np.nan

        gold_c   = get(close, "GC=F")
        gold_h   = get(high,  "GC=F")
        gold_l   = get(low,   "GC=F")
        gold_o   = get(open_, "GC=F")
        silver_c = get(close, "SI=F")

        if np.isnan(gold_c) or not (1000 < gold_c < 4000):
            log(f"[Gold] Skipping {row_date.date()}: gold={gold_c}")
            continue

        prev_gold = float(last_row.get("gold_close", gold_c))
        gold_return = (gold_c - prev_gold) / (prev_gold + 1e-10)
        gs_ratio = gold_c / (silver_c + 1e-10) if not np.isnan(silver_c) else float(last_row.get("gs_ratio", 80))

        row_data = {col: last_row.get(col, np.nan) for col in df.columns}
        row_data.update({
            "date":         str(row_date.date()),
            "gold_close":   gold_c,
            "gold_open":    gold_o if not np.isnan(gold_o) else gold_c,
            "gold_high":    gold_h if not np.isnan(gold_h) else gold_c,
            "gold_low":     gold_l if not np.isnan(gold_l) else gold_c,
            "gold_return":  gold_return,
            "silver_close": silver_c if not np.isnan(silver_c) else float(last_row.get("silver_close", 32)),
            "gs_ratio":     gs_ratio,
            "gold_ma7":     gold_c,
            "gold_ma14":    gold_c,
            "gold_ma30":    gold_c,
        })
        new_rows.append(row_data)
        last_row = pd.Series(row_data)

    if not new_rows:
        log("[Gold] No valid new rows found.")
        return False

    new_df = pd.DataFrame(new_rows).reindex(columns=df.columns)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(path, index=False)
    log(f"[Gold] Added {len(new_rows)} rows. Total: {len(df)}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# RETRAIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def retrain_silver():
    log("[Silver] Starting retrain...")
    t0 = time.time()
    try:
        from src.enhanced_predictor import EnhancedPredictor
        from xgboost import XGBRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.decomposition import PCA
        import numpy as np

        predictor = EnhancedPredictor(sequence_length=60, prediction_days=7)
        predictor.lstm_sequence_length = 45
        predictor.use_lstm = False

        predictor.load_data()
        log(f"[Silver] Data loaded: {len(predictor.data)} rows")

        X = predictor.data[predictor.feature_columns].values
        X_xgb_raw = predictor.data[predictor.xgb_feature_columns].values if predictor.xgb_feature_columns else X

        y_targets = []
        for day in range(1, predictor.prediction_days + 1):
            y_targets.append(predictor.data["price"].shift(-day).values)

        valid_mask = ~np.isnan(y_targets[-1])
        X = X[valid_mask]
        X_xgb_raw = X_xgb_raw[valid_mask]
        y_targets = [y[valid_mask] for y in y_targets]
        base_prices = predictor.data["price"].values[valid_mask]

        xgb_return_targets = [(y_targets[d] - base_prices) / (base_prices + 1e-10) for d in range(predictor.prediction_days)]

        predictor.scaler = StandardScaler()
        X_scaled = predictor.scaler.fit_transform(X)

        predictor.xgb_scaler = StandardScaler()
        X_xgb_scaled = predictor.xgb_scaler.fit_transform(X_xgb_raw)

        predictor.pca = PCA(n_components=0.95, svd_solver="full")
        X_for_ridge = predictor.pca.fit_transform(X_scaled)

        y_all = np.column_stack(y_targets)
        predictor.target_scaler = MinMaxScaler()
        y_scaled = predictor.target_scaler.fit_transform(y_all)

        split_idx = int(len(X_for_ridge) * 0.8)
        X_train_r, X_test_r = X_for_ridge[:split_idx], X_for_ridge[split_idx:]
        X_train_xgb, X_test_xgb = X_xgb_scaled[:split_idx], X_xgb_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        base_prices_test = base_prices[split_idx:]

        # Ridge
        predictor.models = []
        for day in range(predictor.prediction_days):
            m = Ridge(alpha=1.0)
            m.fit(X_train_r, y_train[:, day])
            predictor.models.append(m)

        y_pred_ridge_scaled = np.column_stack([m.predict(X_test_r) for m in predictor.models])
        y_pred_ridge_prices = predictor.target_scaler.inverse_transform(y_pred_ridge_scaled)
        y_test_prices = predictor.target_scaler.inverse_transform(y_test)

        # XGBoost
        predictor.xgb_models = []
        xgb_price_preds = np.zeros_like(y_pred_ridge_prices)
        for day in range(predictor.prediction_days):
            xgb = XGBRegressor(
                n_estimators=1200, max_depth=4, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.5, gamma=0.05, max_delta_step=1,
                random_state=42, n_jobs=-1, tree_method="hist", verbosity=0,
                early_stopping_rounds=40,
            )
            xgb.fit(X_train_xgb, xgb_return_targets[day][:split_idx],
                    eval_set=[(X_test_xgb, xgb_return_targets[day][split_idx:])],
                    verbose=False)
            predictor.xgb_models.append(xgb)
            xgb_price_preds[:, day] = base_prices_test * (1 + xgb.predict(X_test_xgb))

        # Optimize weights
        best_w, best_mape = (1.0, 0.0, 0.0), float("inf")
        for w_r in np.arange(0, 1.01, 0.1):
            for w_x in np.arange(0, 1.01 - w_r, 0.1):
                w_l = max(round(1.0 - w_r - w_x, 2), 0.0)
                blend = w_r * y_pred_ridge_prices + w_x * xgb_price_preds
                mape = np.mean(np.abs((y_test_prices - blend) / (y_test_prices + 1e-10))) * 100
                if mape < best_mape:
                    best_mape = mape
                    best_w = (round(w_r, 2), round(w_x, 2), w_l)

        predictor.ensemble_weights = {"ridge": best_w[0], "xgboost": best_w[1], "lstm": best_w[2]}
        y_pred = best_w[0] * y_pred_ridge_prices + best_w[1] * xgb_price_preds

        rmse = float(np.sqrt(np.mean((y_test_prices - y_pred) ** 2)))
        mae  = float(np.mean(np.abs(y_test_prices - y_pred)))
        mape = float(np.mean(np.abs((y_test_prices - y_pred) / (y_test_prices + 1e-10))) * 100)
        ss_res = np.sum((y_test_prices - y_pred) ** 2)
        ss_tot = np.sum((y_test_prices - np.mean(y_test_prices)) ** 2)
        avg_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        predictor.latest_metrics = {"avg_r2": avg_r2, "rmse": rmse, "mae": mae, "mape": mape, "ensemble": True}
        predictor.use_ensemble = True
        predictor.use_lstm = False
        predictor.save_model()

        elapsed = time.time() - t0
        log(f"[Silver] Done in {elapsed:.0f}s | R2={avg_r2:.4f} | MAPE={mape:.2f}% | Weights={predictor.ensemble_weights}")
        return True

    except Exception as e:
        log(f"[Silver] Retrain FAILED: {e}")
        traceback.print_exc()
        return False


def retrain_gold():
    log("[Gold] Starting retrain...")
    t0 = time.time()
    try:
        from src.vietnam_gold_predictor import VietnamGoldPredictor
        import xgboost as xgb

        predictor = VietnamGoldPredictor(prediction_days=7)
        predictor.usd_vnd_rate = 25900
        predictor.lstm_sequence_length = 45

        predictor.load_vietnam_data()
        predictor.load_world_data()
        predictor.merge_datasets()
        predictor.create_transfer_features()

        # Patch XGBoost params
        _orig_init = xgb.XGBRegressor.__init__
        def _patched(self, **kwargs):
            kwargs.setdefault("n_estimators", 1200)
            kwargs.setdefault("max_depth", 4)
            kwargs.setdefault("learning_rate", 0.025)
            kwargs.setdefault("subsample", 0.8)
            kwargs.setdefault("colsample_bytree", 0.7)
            kwargs.setdefault("min_child_weight", 3)
            kwargs.setdefault("reg_alpha", 0.1)
            kwargs.setdefault("reg_lambda", 1.5)
            kwargs.setdefault("gamma", 0.05)
            kwargs.setdefault("max_delta_step", 1)
            kwargs.setdefault("random_state", 42)
            kwargs.setdefault("n_jobs", -1)
            kwargs.setdefault("tree_method", "hist")
            kwargs.setdefault("verbosity", 0)
            _orig_init(self, **kwargs)
        xgb.XGBRegressor.__init__ = _patched

        try:
            metrics = predictor.train(test_size=0.2, alpha=1.0)
        finally:
            xgb.XGBRegressor.__init__ = _orig_init

        predictor.save_model()
        elapsed = time.time() - t0

        r2s   = list(metrics.get("r2",   {}).values())
        mapes = list(metrics.get("mape", {}).values())
        log(f"[Gold] Done in {elapsed:.0f}s | Avg R2={np.mean(r2s):.4f} | Avg MAPE={np.mean(mapes):.2f}%")
        return True

    except Exception as e:
        log(f"[Gold] Retrain FAILED: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update datasets and optionally retrain models")
    parser.add_argument("--retrain", action="store_true", help="Retrain models after updating data")
    parser.add_argument("--silver-only", action="store_true", help="Only process silver")
    parser.add_argument("--gold-only",   action="store_true", help="Only process gold")
    args = parser.parse_args()

    log("=" * 60)
    log(f"Starting data update | retrain={args.retrain}")
    log("=" * 60)

    do_silver = not args.gold_only
    do_gold   = not args.silver_only

    silver_updated = False
    gold_updated   = False

    if do_silver:
        silver_updated = update_silver_dataset()

    if do_gold:
        gold_updated = update_gold_world_dataset()

    if args.retrain:
        if do_silver and silver_updated:
            retrain_silver()
        elif do_silver:
            log("[Silver] No new data, skipping retrain.")

        if do_gold and gold_updated:
            retrain_gold()
        elif do_gold:
            log("[Gold] No new data, skipping retrain.")

    log("=" * 60)
    log("All done!")
    log("=" * 60)
