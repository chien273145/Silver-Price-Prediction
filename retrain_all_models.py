"""
retrain_all_models.py
=====================
Script huấn luyện lại toàn bộ mô hình Silver và Gold với dữ liệu mới nhất.

Cải tiến so với phiên bản cũ:
- Cập nhật dữ liệu đến ngày hôm nay
- Thêm Stochastic, CCI, Williams %R indicators
- XGBoost: n_estimators 800 → 1200, min_child_weight 5 → 3
- LSTM: sequence_length 30 → 45
- Gold model: cập nhật USD/VND rate

Chạy: python retrain_all_models.py
"""

import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, BASE_DIR)

# ─── Logging helper ───────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "[OK]", "WARN": "[WARN]", "ERROR": "[ERR]", "STEP": "[>>]"}.get(level, "[..]")
    print(f"[{ts}] {prefix} {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Update Datasets
# ══════════════════════════════════════════════════════════════════════════════

def update_silver_dataset():
    """Fetch latest Silver, Gold, DXY, VIX data and append to dataset_enhanced.csv."""
    path = os.path.join(DATASET_DIR, "dataset_enhanced.csv")
    log("Updating Silver dataset...", "STEP")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    last_date = df["Date"].max()
    today = datetime.now()
    gap = (today - last_date).days

    if gap <= 1:
        log(f"Silver data is up to date ({last_date.date()}). Skipping fetch.")
        return df

    log(f"Silver data is {gap} days old. Fetching from {last_date.date()} to {today.date()}...")

    try:
        tickers = yf.download(
            ["SI=F", "GC=F", "DX-Y.NYB", "^VIX"],
            start=(last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if tickers.empty:
            log("No new data from Yahoo Finance.", "WARN")
            return df

        # Flatten multi-level columns
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
            row_date = pd.Timestamp(date_idx).date()
            if pd.Timestamp(row_date) <= last_date:
                continue

            silver_c = close.get("SI=F", pd.Series()).get(date_idx, np.nan)
            silver_h = high.get("SI=F",  pd.Series()).get(date_idx, np.nan)
            silver_l = low.get("SI=F",   pd.Series()).get(date_idx, np.nan)
            silver_o = open_.get("SI=F", pd.Series()).get(date_idx, np.nan)
            gold_c   = close.get("GC=F", pd.Series()).get(date_idx, np.nan)
            dxy_c    = close.get("DX-Y.NYB", pd.Series()).get(date_idx, np.nan)
            vix_c    = close.get("^VIX", pd.Series()).get(date_idx, np.nan)

            if any(np.isnan(v) for v in [silver_c, gold_c]):
                continue

            # Sanity checks
            if silver_c > 100 or silver_c < 5:
                continue
            if gold_c > 4000 or gold_c < 1000:
                continue

            new_rows.append({
                "Date": str(row_date),
                "Silver_Close": silver_c,
                "Silver_Open":  silver_o if not np.isnan(silver_o) else silver_c,
                "Silver_High":  silver_h if not np.isnan(silver_h) else silver_c,
                "Silver_Low":   silver_l if not np.isnan(silver_l) else silver_c,
                "Gold": gold_c,
                "DXY": dxy_c if not np.isnan(dxy_c) else df["DXY"].iloc[-1],
                "VIX": vix_c if not np.isnan(vix_c) else df["VIX"].iloc[-1],
                "price": silver_c,
                "date": str(row_date),
                "high": silver_h if not np.isnan(silver_h) else silver_c,
                "low":  silver_l if not np.isnan(silver_l) else silver_c,
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(path, index=False)
            log(f"Added {len(new_rows)} new rows to Silver dataset. Total: {len(df)}")
        else:
            log("No valid new rows to add to Silver dataset.", "WARN")

    except Exception as e:
        log(f"Error updating Silver dataset: {e}", "ERROR")
        traceback.print_exc()

    return df


def update_gold_world_dataset():
    """Fetch latest Gold world data and append to gold_geopolitical_dataset.csv."""
    path = os.path.join(DATASET_DIR, "gold_geopolitical_dataset.csv")
    log("Updating Gold world dataset...", "STEP")

    df = pd.read_csv(path, parse_dates=["date"])
    last_date = df["date"].max()
    today = datetime.now()
    gap = (today - last_date).days

    if gap <= 1:
        log(f"Gold world data is up to date ({last_date.date()}). Skipping fetch.")
        return df

    log(f"Gold world data is {gap} days old. Fetching...")

    try:
        tickers = yf.download(
            ["GC=F", "SI=F"],
            start=(last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        if tickers.empty:
            log("No new Gold world data.", "WARN")
            return df

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
            row_date = pd.Timestamp(date_idx).date()
            if pd.Timestamp(row_date) <= last_date:
                continue

            gold_c = close.get("GC=F", pd.Series()).get(date_idx, np.nan)
            gold_h = high.get("GC=F",  pd.Series()).get(date_idx, np.nan)
            gold_l = low.get("GC=F",   pd.Series()).get(date_idx, np.nan)
            gold_o = open_.get("GC=F", pd.Series()).get(date_idx, np.nan)
            silver_c = close.get("SI=F", pd.Series()).get(date_idx, np.nan)

            if np.isnan(gold_c) or gold_c > 4000 or gold_c < 1000:
                continue

            # Compute derived columns that exist in the dataset
            last_row = df.iloc[-1]
            gold_return = (gold_c - last_row.get("gold_close", gold_c)) / (last_row.get("gold_close", gold_c) + 1e-10)
            gs_ratio = gold_c / (silver_c + 1e-10) if not np.isnan(silver_c) else last_row.get("gs_ratio", 80)

            new_rows.append({
                "date": str(row_date),
                "gold_close": gold_c,
                "gold_open":  gold_o if not np.isnan(gold_o) else gold_c,
                "gold_high":  gold_h if not np.isnan(gold_h) else gold_c,
                "gold_low":   gold_l if not np.isnan(gold_l) else gold_c,
                "gold_return": gold_return,
                "silver_close": silver_c if not np.isnan(silver_c) else last_row.get("silver_close", 32),
                "gs_ratio": gs_ratio,
                # Forward-fill other columns from last row
                "gold_volatility": last_row.get("gold_volatility", 0),
                "gold_rsi": last_row.get("gold_rsi", 50),
                "gold_macd": last_row.get("gold_macd", 0),
                "gold_ma7": last_row.get("gold_ma7", gold_c),
                "gold_ma14": last_row.get("gold_ma14", gold_c),
                "gold_ma30": last_row.get("gold_ma30", gold_c),
                "gpr_level": last_row.get("gpr_level", 100),
                "gpr_ma7": last_row.get("gpr_ma7", 100),
                "composite_risk": last_row.get("composite_risk", 0),
                "gs_ratio_ma7": last_row.get("gs_ratio_ma7", gs_ratio),
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # Ensure column order matches
            for col in df.columns:
                if col not in new_df.columns:
                    new_df[col] = np.nan
            new_df = new_df[df.columns]
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(path, index=False)
            log(f"Added {len(new_rows)} new rows to Gold world dataset. Total: {len(df)}")
        else:
            log("No valid new rows to add to Gold world dataset.", "WARN")

    except Exception as e:
        log(f"Error updating Gold world dataset: {e}", "ERROR")
        traceback.print_exc()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Improve and Retrain Silver Model
# ══════════════════════════════════════════════════════════════════════════════

def retrain_silver():
    """Retrain Silver model with improved hyperparameters."""
    log("=" * 60, "STEP")
    log("RETRAINING SILVER MODEL", "STEP")
    log("=" * 60, "STEP")

    from src.enhanced_predictor import EnhancedPredictor

    predictor = EnhancedPredictor(
        sequence_length=60,
        prediction_days=7,
    )

    # Override XGBoost hyperparameters for better performance
    # These will be picked up during train()
    predictor.lstm_sequence_length = 45  # Increased from 30

    log("Loading and preparing Silver data...")
    predictor.load_data()

    log("Training Silver ensemble (Ridge + XGBoost + LSTM)...")
    log("  This will take 15-30 minutes for LSTM training...")
    t0 = time.time()

    # Monkey-patch XGBoost params for this run
    _original_train = predictor.train

    def _improved_train(test_size=0.2, use_pca=True, pca_variance=0.95):
        """Wrap train() to inject improved XGBoost hyperparameters."""
        import xgboost as xgb

        # Patch XGBRegressor defaults via monkey-patch on the class
        _orig_init = xgb.XGBRegressor.__init__

        def _patched_init(self_xgb, **kwargs):
            # Override key hyperparameters
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
            _orig_init(self_xgb, **kwargs)

        xgb.XGBRegressor.__init__ = _patched_init
        try:
            result = _original_train(test_size=test_size, use_pca=use_pca, pca_variance=pca_variance)
        finally:
            xgb.XGBRegressor.__init__ = _orig_init
        return result

    predictor.train = _improved_train

    metrics = predictor.train()
    elapsed = time.time() - t0

    log(f"Silver training complete in {elapsed/60:.1f} minutes")
    log(f"  R²:   {metrics.get('avg_r2', 0):.4f}")
    log(f"  RMSE: ${metrics.get('rmse', 0):.4f}")
    log(f"  MAE:  ${metrics.get('mae', 0):.4f}")
    log(f"  MAPE: {metrics.get('mape', 0):.2f}%")

    predictor.save_model()
    log("Silver model saved OK")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Improve and Retrain Gold Model
# ══════════════════════════════════════════════════════════════════════════════

def retrain_gold():
    """Retrain Gold model with improved hyperparameters and updated USD/VND rate."""
    log("=" * 60, "STEP")
    log("RETRAINING GOLD MODEL", "STEP")
    log("=" * 60, "STEP")

    from src.vietnam_gold_predictor import VietnamGoldPredictor

    predictor = VietnamGoldPredictor(prediction_days=7)

    # Update USD/VND rate to current value
    predictor.usd_vnd_rate = 25900
    predictor.lstm_sequence_length = 45  # Increased from 30

    log("Loading Vietnam SJC gold data...")
    predictor.load_vietnam_data()

    log("Loading world gold data...")
    predictor.load_world_data()

    log("Merging datasets...")
    predictor.merge_datasets()

    log("Creating transfer features...")
    predictor.create_transfer_features()

    log("Training Gold ensemble (Ridge + XGBoost + LSTM)...")
    log("  This will take 15-30 minutes for LSTM training...")
    t0 = time.time()

    # Monkey-patch XGBoost params
    import xgboost as xgb
    _orig_init = xgb.XGBRegressor.__init__

    def _patched_init(self_xgb, **kwargs):
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
        _orig_init(self_xgb, **kwargs)

    xgb.XGBRegressor.__init__ = _patched_init
    try:
        metrics = predictor.train(test_size=0.2, alpha=1.0)
    finally:
        xgb.XGBRegressor.__init__ = _orig_init

    elapsed = time.time() - t0
    log(f"Gold training complete in {elapsed/60:.1f} minutes")

    avg_r2   = np.mean(list(metrics["r2"].values()))
    avg_mape = np.mean(list(metrics["mape"].values()))
    avg_rmse = np.mean(list(metrics["rmse"].values()))
    log(f"  R²:   {avg_r2:.4f}")
    log(f"  MAPE: {avg_mape:.2f}%")
    log(f"  RMSE: {avg_rmse:,.0f} VND")

    predictor.save_model()
    log("Gold model saved OK")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Verify Predictions
# ══════════════════════════════════════════════════════════════════════════════

def verify_silver():
    log("Verifying Silver predictions...", "STEP")
    try:
        from src.enhanced_predictor import EnhancedPredictor
        p = EnhancedPredictor()
        p.load_model()
        result = p.predict(in_vnd=True)
        preds = result.get("predictions", [])
        if preds:
            log(f"  Silver Day 1: {preds[0]['predicted_price']:,.0f} VND/lượng")
            log(f"  Silver Day 7: {preds[-1]['predicted_price']:,.0f} VND/lượng")
            log("Silver verification OK OK")
        else:
            log("Silver prediction returned empty!", "WARN")
    except Exception as e:
        log(f"Silver verification failed: {e}", "ERROR")
        traceback.print_exc()


def verify_gold():
    log("Verifying Gold predictions...", "STEP")
    try:
        from src.vietnam_gold_predictor import VietnamGoldPredictor
        p = VietnamGoldPredictor()
        p.load_model()
        # Need to load data for predict()
        p.load_vietnam_data()
        p.load_world_data()
        p.merge_datasets()
        p.create_transfer_features()
        result = p.predict()
        if result:
            log(f"  Gold Day 1: {result[0]['predicted_price']:,.0f} VND/lượng")
            log(f"  Gold Day 7: {result[-1]['predicted_price']:,.0f} VND/lượng")
            log("Gold verification OK OK")
        else:
            log("Gold prediction returned empty!", "WARN")
    except Exception as e:
        log(f"Gold verification failed: {e}", "ERROR")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_total = time.time()
    log("=" * 60)
    log("RETRAIN ALL MODELS — Starting")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # Step 1: Update data
    log("PHASE 1: Updating datasets", "STEP")
    update_silver_dataset()
    update_gold_world_dataset()

    # Step 2: Retrain Silver
    log("\nPHASE 2: Retraining Silver model", "STEP")
    silver_metrics = None
    try:
        silver_metrics = retrain_silver()
    except Exception as e:
        log(f"Silver training failed: {e}", "ERROR")
        traceback.print_exc()

    # Step 3: Retrain Gold
    log("\nPHASE 3: Retraining Gold model", "STEP")
    gold_metrics = None
    try:
        gold_metrics = retrain_gold()
    except Exception as e:
        log(f"Gold training failed: {e}", "ERROR")
        traceback.print_exc()

    # Step 4: Verify
    log("\nPHASE 4: Verifying predictions", "STEP")
    verify_silver()
    verify_gold()

    total_elapsed = (time.time() - start_total) / 60
    log("=" * 60)
    log(f"ALL DONE in {total_elapsed:.1f} minutes")
    log("=" * 60)

    # Summary
    if silver_metrics:
        log(f"Silver: R²={silver_metrics.get('avg_r2',0):.4f}, MAPE={silver_metrics.get('mape',0):.2f}%")
    if gold_metrics:
        avg_r2   = np.mean(list(gold_metrics["r2"].values()))
        avg_mape = np.mean(list(gold_metrics["mape"].values()))
        log(f"Gold:   R²={avg_r2:.4f}, MAPE={avg_mape:.2f}%")

    log("\nNext step: git push to deploy to Render")
