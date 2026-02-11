"""
Gold Price Predictor - Ridge Regression with Extended Features
Dự đoán giá vàng 7 ngày sử dụng Feature Engineering + Ridge Regression.
Optimized: R² = 0.91, MAPE = 6.22%
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# LSTM ensemble (optional - graceful fallback)
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_USD_VND_RATE = 25000


class GoldPredictor:
    """Gold price predictor using Extended Feature Engineering + Ridge Regression."""
    
    def __init__(
        self,
        model_dir: str = None,
        data_path: str = None,
        prediction_days: int = 7
    ):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(self.base_dir, 'models')
        self.data_path = data_path or os.path.join(
            self.base_dir, 'dataset', 'gold_enhanced_dataset.csv'
        )
        
        self.prediction_days = prediction_days
        
        self.data = None
        self.models = None
        self.scaler = None
        self.pca = None
        self.feature_columns = None
        self.metrics = None

        # Ensemble
        self.xgb_models = None
        self.xgb_scaler = None
        self.xgb_feature_columns = []
        self.ensemble_weights = {'ridge': 0.3, 'xgboost': 0.4, 'lstm': 0.3}
        self.use_ensemble = XGBOOST_AVAILABLE

        # LSTM config
        self.lstm_model = None
        self.lstm_sequence_length = 30
        self.use_lstm = LSTM_AVAILABLE

        self.usd_vnd_rate = DEFAULT_USD_VND_RATE
        self.troy_ounce_to_luong = 1.20565
        self.troy_ounce_to_luong = 1.20565
        self.vietnam_premium = 1.125 # Adjusted to match market (172M vs 189M)

        # Sentiment analyzer (Phase 2)
        self._sentiment_analyzer = None
        try:
            from src.finbert_sentiment import FinBERTSentiment
            self._sentiment_analyzer = FinBERTSentiment(use_cache_only=True)
        except Exception:
            pass
        
    def load_data(self):
        """Load gold geopolitical dataset."""
        print(f"Loading dataset from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        if 'gold_close' in self.data.columns:
            self.data['price'] = self.data['gold_close']
            
        # KEEP DATA IN USD
        # We model World Gold Price (USD/oz) directly.
        # Conversion to VND happens only at prediction output.
        
        # Ensure numeric
        self.data['price'] = pd.to_numeric(self.data['price'], errors='coerce')
        self.data = self.data.dropna(subset=['price'])
        
        
        print(f"Loaded {len(self.data)} rows. Price range: {self.data['price'].min():,.0f} - {self.data['price'].max():,.0f}")
        print("Last 5 rows dates:")
        print(self.data['date'].tail().astype(str).tolist())
        
    def create_features(self):
        """Create extended features with momentum and volatility indicators."""
        print("Creating extended features...")
        
        # Base features from dataset
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
        
        # Create additional lag features
        for lag in [2, 3, 5, 10, 21, 60, 90]:
            col_name = f'gold_lag{lag}'
            if col_name not in self.data.columns:
                self.data[col_name] = self.data['price'].shift(lag)
                base_features.append(col_name)
        
        # Create momentum features
        self.data['momentum_3'] = self.data['price'] - self.data['price'].shift(3)
        self.data['momentum_7'] = self.data['price'] - self.data['price'].shift(7)
        self.data['momentum_14'] = self.data['price'] - self.data['price'].shift(14)
        self.data['momentum_30'] = self.data['price'] - self.data['price'].shift(30)
        
        # Rate of change
        self.data['roc_7'] = (self.data['price'] - self.data['price'].shift(7)) / (self.data['price'].shift(7) + 1e-10)
        self.data['roc_14'] = (self.data['price'] - self.data['price'].shift(14)) / (self.data['price'].shift(14) + 1e-10)
        self.data['roc_30'] = (self.data['price'] - self.data['price'].shift(30)) / (self.data['price'].shift(30) + 1e-10)
        
        # Rolling statistics
        self.data['rolling_std_7'] = self.data['price'].rolling(7).std()
        self.data['rolling_std_14'] = self.data['price'].rolling(14).std()
        self.data['rolling_std_30'] = self.data['price'].rolling(30).std()
        self.data['rolling_min_7'] = self.data['price'].rolling(7).min()
        self.data['rolling_max_7'] = self.data['price'].rolling(7).max()
        
        # ======= ADVANCED FEATURES (Phase 1 improvements) =======
        # Mean-reversion signals
        vol_20 = self.data['rolling_std_14'].replace(0, 1e-10)
        self.data['mean_reversion_14'] = (self.data['price'] - self.data['gold_ma14']) / vol_20
        self.data['mean_reversion_30'] = (self.data['price'] - self.data['gold_ma30']) / (self.data['rolling_std_30'] + 1e-10)
        
        # Rolling cross-correlation with DXY and VIX
        if 'dxy' in self.data.columns:
            self.data['corr_gold_dxy_20'] = self.data['price'].rolling(20).corr(self.data['dxy'])
        if 'vix' in self.data.columns:
            self.data['corr_gold_vix_20'] = self.data['price'].rolling(20).corr(self.data['vix'])
        if 'silver_close' in self.data.columns:
            self.data['corr_gold_silver_20'] = self.data['price'].rolling(20).corr(self.data['silver_close'])
        
        # Price regime indicator (trending vs ranging)
        if 'gold_ma14' in self.data.columns:
            price_range_14 = self.data['gold_ma14'].rolling(14).apply(
                lambda x: (x.max() - x.min()) / (x.mean() + 1e-10) if len(x) > 0 else 0, raw=True
            )
            self.data['regime_trending'] = (price_range_14 > price_range_14.rolling(60).mean()).astype(int)
        
        # Keltner channel position
        if 'gold_ema14' in self.data.columns and 'gold_volatility' in self.data.columns:
            kc_range = 2.0 * self.data['gold_volatility']
            self.data['kc_position'] = (self.data['price'] - self.data['gold_ema14']) / (kc_range + 1e-10)
        
        # Extended features list (including VIX, DXY, Oil, US10Y)
        market_features = [
            # VIX features
            'vix', 'vix_ma7', 'vix_ma30', 'vix_change', 'high_vix',
            # DXY features  
            'dxy', 'dxy_ma7', 'dxy_ma30', 'dxy_change', 'dxy_momentum',
            # Oil features
            'oil', 'oil_ma7', 'oil_ma30', 'oil_change', 'oil_volatility',
            # Interest rate features
            'us10y', 'us10y_ma7', 'us10y_ma30', 'us10y_change', 'rate_trend',
            # Cross-asset features
            'fear_index', 'gold_oil_ratio'
        ]
        
        advanced_features = [
            'mean_reversion_14', 'mean_reversion_30',
            'corr_gold_dxy_20', 'corr_gold_vix_20', 'corr_gold_silver_20',
            'regime_trending', 'kc_position'
        ]
        
        extended_features = base_features + [
            'momentum_3', 'momentum_7', 'momentum_14', 'momentum_30',
            'roc_7', 'roc_14', 'roc_30',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
            'rolling_min_7', 'rolling_max_7'
        ] + market_features + advanced_features
        
        # ======= SENTIMENT FEATURES (Phase 2) =======
        if self._sentiment_analyzer and self._sentiment_analyzer._cache:
            dates = self.data['date'].dt.strftime('%Y-%m-%d').tolist()
            scores = self._sentiment_analyzer.get_sentiment_features(dates)
            self.data['sentiment_score'] = scores
            self.data['sentiment_ma3'] = self.data['sentiment_score'].rolling(3, min_periods=1).mean()
            self.data['sentiment_ma7'] = self.data['sentiment_score'].rolling(7, min_periods=1).mean()
            self.data['sentiment_change'] = self.data['sentiment_score'].diff().fillna(0)
            extended_features += ['sentiment_score', 'sentiment_ma3', 'sentiment_ma7', 'sentiment_change']
        
        self.feature_columns = [col for col in extended_features if col in self.data.columns]
        print(f"Selected {len(self.feature_columns)} extended features (incl. VIX, DXY, Oil, US10Y)")

        # XGBoost stationary features (exclude raw prices & non-stationary)
        _xgb_exclude = {
            'gold_open', 'gold_high', 'gold_low', 'gold_close', 'silver_close',
            'gold_lag1', 'gold_lag2', 'gold_lag3', 'gold_lag5', 'gold_lag7',
            'gold_lag10', 'gold_lag14', 'gold_lag21', 'gold_lag30', 'gold_lag60', 'gold_lag90',
            'gold_ma7', 'gold_ma14', 'gold_ma30', 'gold_ma60',
            'gold_ema7', 'gold_ema14', 'gold_ema30',
            'dxy', 'vix', 'oil', 'us10y',
            'dxy_ma7', 'dxy_ma30', 'vix_ma7', 'vix_ma30',
            'oil_ma7', 'oil_ma30', 'us10y_ma7', 'us10y_ma30',
            'momentum_3', 'momentum_7', 'momentum_14', 'momentum_30',
            'rolling_min_7', 'rolling_max_7', 'year',
        }
        self.xgb_feature_columns = [f for f in self.feature_columns if f not in _xgb_exclude]
        print(f"  XGBoost stationary features: {len(self.xgb_feature_columns)}/{len(self.feature_columns)}")

        # Handle missing values
        self.data[self.feature_columns] = self.data[self.feature_columns].ffill().fillna(0)
        
    def train(self, test_size: float = 0.2, use_pca: bool = True, pca_variance: float = 0.95, alpha: float = 1.0):
        """Train Ridge + XGBoost Ensemble models with extended features."""
        print(f"Training Ridge+XGBoost ensemble (alpha={alpha})...")

        # Prepare data
        # Ridge features (all)
        X = self.data[self.feature_columns].values
        # XGBoost features (stationary only)
        X_xgb_raw = self.data[self.xgb_feature_columns].values if self.xgb_feature_columns else X

        # Create absolute price targets for Ridge
        targets = {}
        for day in range(1, self.prediction_days + 1):
            target_col = f'target_day_{day}'
            self.data[target_col] = self.data['price'].shift(-day)
            targets[day] = self.data[target_col].values

        # Remove rows with NaN targets
        valid_mask = ~np.isnan(targets[self.prediction_days])
        X = X[valid_mask]
        X_xgb_raw = X_xgb_raw[valid_mask]
        for day in targets:
            targets[day] = targets[day][valid_mask]

        # Create RETURN targets for XGBoost
        base_prices = self.data['price'].values[valid_mask]
        xgb_targets = {}
        for day in range(1, self.prediction_days + 1):
            xgb_targets[day] = (targets[day] - base_prices) / (base_prices + 1e-10)

        # Scale Ridge features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Scale XGBoost features (separate scaler)
        self.xgb_scaler = StandardScaler()
        X_xgb_scaled = self.xgb_scaler.fit_transform(X_xgb_raw)

        # Apply PCA (for Ridge only)
        if use_pca:
            self.pca = PCA(n_components=pca_variance)
            X_for_ridge = self.pca.fit_transform(X_scaled)
            print(f"PCA: {X_scaled.shape[1]} -> {X_for_ridge.shape[1]} components (Ridge)")
        else:
            X_for_ridge = X_scaled

        # Split once (same indices for both)
        split_idx = int(len(X_for_ridge) * (1 - test_size))
        X_train_r, X_test_r = X_for_ridge[:split_idx], X_for_ridge[split_idx:]
        X_train_xgb, X_test_xgb = X_xgb_scaled[:split_idx], X_xgb_scaled[split_idx:]
        base_prices_test = base_prices[split_idx:]

        # Train models
        self.models = {}
        self.xgb_models = {}
        self.metrics = {'r2': {}, 'mape': {}}

        all_ridge_preds = {}
        all_xgb_preds = {}  # Stored as PRICES (converted from returns)
        all_y_tests = {}

        for day in range(1, self.prediction_days + 1):
            y = targets[day]
            y_train, y_test = y[:split_idx], y[split_idx:]
            all_y_tests[day] = y_test

            # Ridge: predict absolute price
            model = Ridge(alpha=alpha)
            model.fit(X_train_r, y_train)
            y_pred_r = model.predict(X_test_r)
            self.models[day] = model
            all_ridge_preds[day] = y_pred_r

            # XGBoost: stationary features → predict returns
            if self.use_ensemble:
                y_xgb = xgb_targets[day]
                y_xgb_train, y_xgb_test = y_xgb[:split_idx], y_xgb[split_idx:]

                xgb_model = XGBRegressor(
                    n_estimators=800, max_depth=3, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.7,
                    min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.5,
                    gamma=0.1,
                    random_state=42, n_jobs=1, tree_method='hist', verbosity=0,
                    early_stopping_rounds=30
                )
                xgb_model.fit(X_train_xgb, y_xgb_train,
                              eval_set=[(X_test_xgb, y_xgb_test)], verbose=False)
                self.xgb_models[day] = xgb_model

                # Convert XGBoost return predictions → absolute prices
                xgb_return_pred = xgb_model.predict(X_test_xgb)
                xgb_price_pred = base_prices_test * (1 + xgb_return_pred)
                all_xgb_preds[day] = xgb_price_pred

                ridge_mape = np.mean(np.abs((y_test - y_pred_r) / y_test)) * 100
                xgb_mape = np.mean(np.abs((y_test - xgb_price_pred) / y_test)) * 100
                print(f"  Day {day}: Ridge MAPE={ridge_mape:.2f}%, XGB MAPE={xgb_mape:.2f}%")
            else:
                mape = np.mean(np.abs((y_test - y_pred_r) / y_test)) * 100
                print(f"  Day {day}: Ridge MAPE={mape:.2f}%")

        # --- LSTM: sequence → predict returns ---
        all_lstm_preds = {}
        if self.use_lstm and split_idx > self.lstm_sequence_length + 10:
            print("\n  Training LSTM (seq_len={})...".format(self.lstm_sequence_length))
            seq_len = self.lstm_sequence_length

            def _make_sequences(X_data, base_p, tgts, start, end):
                seqs, rets = [], []
                for i in range(start, end):
                    seqs.append(X_data[i - seq_len:i])
                    future_returns = [(tgts[d][i] - base_p[i]) / (base_p[i] + 1e-10)
                                      for d in range(1, self.prediction_days + 1)]
                    rets.append(future_returns)
                return np.array(seqs, dtype=np.float32), np.array(rets, dtype=np.float32)

            X_seq_train, y_seq_train = _make_sequences(
                X_scaled, base_prices, targets, seq_len, split_idx)
            X_seq_test, y_seq_test = _make_sequences(
                X_scaled, base_prices, targets, split_idx, len(X_scaled))

            print(f"    LSTM train: {len(X_seq_train)}, test: {len(X_seq_test)}")

            lstm_model = Sequential([
                tf.keras.layers.Input(shape=(seq_len, X_scaled.shape[1])),
                Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
                Bidirectional(LSTM(32, dropout=0.2)),
                Dense(32, activation='relu'),
                Dropout(0.15),
                Dense(16, activation='relu'),
                Dropout(0.1),
                Dense(self.prediction_days)
            ])
            lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='huber')
            lstm_model.fit(
                X_seq_train, y_seq_train,
                epochs=100, batch_size=16,
                validation_data=(X_seq_test, y_seq_test),
                callbacks=[EarlyStopping(patience=15, restore_best_weights=True)],
                verbose=0
            )
            self.lstm_model = lstm_model

            lstm_return_preds = lstm_model.predict(X_seq_test, verbose=0)
            for day in range(1, self.prediction_days + 1):
                all_lstm_preds[day] = base_prices_test * (1 + lstm_return_preds[:, day - 1])
                lstm_mape = np.mean(np.abs((all_y_tests[day] - all_lstm_preds[day]) / all_y_tests[day])) * 100
                print(f"    Day {day}: LSTM MAPE={lstm_mape:.2f}%")

        # --- Optimize ensemble weights (3D grid: Ridge + XGB + LSTM) ---
        has_xgb = self.use_ensemble and bool(self.xgb_models)
        has_lstm = self.use_lstm and bool(all_lstm_preds)

        if has_xgb or has_lstm:
            best_w, best_mape = (1.0, 0.0, 0.0), float('inf')
            step = 0.1
            for w_r in np.arange(0, 1.0 + step / 2, step):
                for w_x in np.arange(0, 1.0 - w_r + step / 2, step):
                    w_l = round(1.0 - w_r - w_x, 2)
                    if w_l < -0.01:
                        continue
                    w_l = max(w_l, 0.0)
                    if not has_xgb and w_x > 0.01:
                        continue
                    if not has_lstm and w_l > 0.01:
                        continue
                    total_mape = 0
                    for day in range(1, self.prediction_days + 1):
                        blended = w_r * all_ridge_preds[day]
                        if has_xgb:
                            blended = blended + w_x * all_xgb_preds[day]
                        if has_lstm and day in all_lstm_preds:
                            blended = blended + w_l * all_lstm_preds[day]
                        total_mape += np.mean(np.abs((all_y_tests[day] - blended) / all_y_tests[day]))
                    avg_mape = (total_mape / self.prediction_days) * 100
                    if avg_mape < best_mape:
                        best_mape = avg_mape
                        best_w = (round(w_r, 2), round(w_x, 2), round(w_l, 2))

            self.ensemble_weights = {'ridge': best_w[0], 'xgboost': best_w[1], 'lstm': best_w[2]}
            print(f"\n  Optimal weights: Ridge={best_w[0]}, XGB={best_w[1]}, LSTM={best_w[2]}")

        # --- Evaluate with optimized weights ---
        w_r = self.ensemble_weights.get('ridge', 1.0)
        w_x = self.ensemble_weights.get('xgboost', 0.0)
        w_l = self.ensemble_weights.get('lstm', 0.0)

        for day in range(1, self.prediction_days + 1):
            y_test = all_y_tests[day]
            y_pred = w_r * all_ridge_preds[day]
            if has_xgb and day in all_xgb_preds:
                y_pred = y_pred + w_x * all_xgb_preds[day]
            if has_lstm and day in all_lstm_preds:
                y_pred = y_pred + w_l * all_lstm_preds[day]

            r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            self.metrics['r2'][day] = r2
            self.metrics['mape'][day] = mape

        avg_r2 = np.mean(list(self.metrics['r2'].values()))
        avg_mape = np.mean(list(self.metrics['mape'].values()))
        mode = f"R={w_r}+X={w_x}+L={w_l}" if (has_xgb or has_lstm) else "Ridge"
        print(f"{mode} Average R2={avg_r2:.4f}, MAPE={avg_mape:.2f}%")
        
    def save_model(self):
        """Save trained model (Ridge + XGBoost + LSTM ensemble)."""
        os.makedirs(self.model_dir, exist_ok=True)

        if self.lstm_model:
            version_num = 4
        elif self.xgb_models:
            version_num = 3
        else:
            version_num = 1

        model_data = {
            'models': self.models,
            'xgb_models': self.xgb_models if self.use_ensemble else None,
            'ensemble_weights': self.ensemble_weights,
            'scaler': self.scaler,
            'xgb_scaler': self.xgb_scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'xgb_feature_columns': self.xgb_feature_columns,
            'metrics': self.metrics,
            'prediction_days': self.prediction_days,
            # LSTM fields (v4)
            'lstm_weights': self.lstm_model.get_weights() if self.lstm_model else None,
            'lstm_n_features': len(self.feature_columns),
            'lstm_sequence_length': self.lstm_sequence_length,
            'model_version': version_num
        }

        path = os.path.join(self.model_dir, 'gold_ridge_models.pkl')
        joblib.dump(model_data, path)
        version_labels = {4: "v4 R+X+L", 3: "v3 R+X", 1: "v1 Ridge"}
        print(f"Model saved to {path} [{version_labels.get(version_num, 'v?')}]")
        
    def load_model(self):
        """Load trained model."""
        path = os.path.join(self.model_dir, 'gold_ridge_models.pkl')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.pca = model_data.get('pca')
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data.get('metrics', {})
        self.prediction_days = model_data.get('prediction_days', 7)

        # Load XGBoost ensemble (backward compatible with v1/v2)
        self.xgb_models = model_data.get('xgb_models', None)
        self.ensemble_weights = model_data.get('ensemble_weights', {'ridge': 0.4, 'xgboost': 0.6})
        self.xgb_feature_columns = model_data.get('xgb_feature_columns', self.feature_columns)
        self.xgb_scaler = model_data.get('xgb_scaler', self.scaler)
        self.use_ensemble = bool(self.xgb_models and XGBOOST_AVAILABLE)

        # Load LSTM (v4+)
        self.lstm_sequence_length = model_data.get('lstm_sequence_length', 30)
        lstm_weights = model_data.get('lstm_weights', None)
        if lstm_weights is not None and LSTM_AVAILABLE:
            try:
                n_features = model_data.get('lstm_n_features', len(self.feature_columns))
                self.lstm_model = Sequential([
                    tf.keras.layers.Input(shape=(self.lstm_sequence_length, n_features)),
                    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
                    Bidirectional(LSTM(32, dropout=0.2)),
                    Dense(32, activation='relu'),
                    Dropout(0.15),
                    Dense(16, activation='relu'),
                    Dropout(0.1),
                    Dense(self.prediction_days)
                ])
                self.lstm_model.set_weights(lstm_weights)
            except Exception as e:
                print(f"  Warning: Could not load LSTM: {e}")
                self.lstm_model = None
        self.use_lstm = LSTM_AVAILABLE and self.lstm_model is not None

        version = model_data.get('model_version', 1)
        components = ["Ridge"]
        if self.use_ensemble:
            components.append("XGB")
        if self.use_lstm:
            components.append("LSTM")
        print(f"Loaded gold model (v{version}, {'+'.join(components)}, {len(self.models)} day predictions)")
        
    def predict(self, in_vnd: bool = True) -> Dict:
        """Make predictions for the next 7 days."""
        if self.models is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get latest data point
        latest = self.data.iloc[-1]
        last_price_usd = latest['price']

        # Ridge features (all → scale → PCA)
        latest_features = latest[self.feature_columns].values.reshape(1, -1)
        if self.scaler:
            scaled_full = self.scaler.transform(latest_features)
            scaled_for_ridge = self.pca.transform(scaled_full) if self.pca else scaled_full
        else:
            scaled_full = latest_features
            scaled_for_ridge = latest_features

        # XGBoost features (stationary → separate scaler)
        latest_xgb = self.data[self.xgb_feature_columns].iloc[-1:].values
        latest_xgb_scaled = self.xgb_scaler.transform(latest_xgb) if self.xgb_scaler else latest_xgb

        # LSTM: get last seq_len scaled features as sequence
        lstm_returns = None
        if self.use_lstm and self.lstm_model is not None:
            seq_len = self.lstm_sequence_length
            if len(self.data) >= seq_len:
                lstm_input = self.data[self.feature_columns].iloc[-seq_len:].values
                lstm_input_scaled = self.scaler.transform(lstm_input).astype(np.float32)
                lstm_returns = self.lstm_model.predict(
                    lstm_input_scaled.reshape(1, seq_len, -1), verbose=0)[0]

        # Active weights (normalize if a component is missing)
        w_r = self.ensemble_weights.get('ridge', 1.0)
        w_x = self.ensemble_weights.get('xgboost', 0.0) if (self.use_ensemble and self.xgb_models) else 0
        w_l = self.ensemble_weights.get('lstm', 0.0) if lstm_returns is not None else 0
        total_w = w_r + w_x + w_l
        if total_w > 0:
            w_r, w_x, w_l = w_r / total_w, w_x / total_w, w_l / total_w

        # Predict (Result is USD/oz)
        predictions_usd_raw = []
        for day in range(1, self.prediction_days + 1):
            ridge_pred = self.models[day].predict(scaled_for_ridge)[0]
            pred = w_r * ridge_pred

            if w_x > 0 and day in self.xgb_models:
                xgb_return = self.xgb_models[day].predict(latest_xgb_scaled)[0]
                xgb_pred = last_price_usd * (1 + xgb_return)
                pred += w_x * xgb_pred

            if w_l > 0 and lstm_returns is not None:
                lstm_pred = last_price_usd * (1 + lstm_returns[day - 1])
                pred += w_l * lstm_pred

            predictions_usd_raw.append(pred)
            
        # Construct response
        last_date = self.data['date'].iloc[-1]
        
        response_list = []
        
        # Helper vars
        conversion_factor = self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
        
        # Last known
        last_price_usd = latest['price']
        last_price_vnd = last_price_usd * conversion_factor
        
        final_last_known = last_price_vnd if in_vnd else last_price_usd

        prev_price = final_last_known
        
        current_date = last_date
        
        for i, pred_usd in enumerate(predictions_usd_raw):
            # Skip weekends
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            
            # Calculate values
            val_usd = float(pred_usd)
            val_vnd = val_usd * conversion_factor
            
            # Select target
            target_price = val_vnd if in_vnd else val_usd
            
            # Change
            change_abs = target_price - prev_price
            change_pct = (change_abs / prev_price) * 100
            
            response_list.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day': i + 1,
                'price': round(target_price, 2),
                'price_usd': round(val_usd, 2),
                'change': {
                    'absolute': round(change_abs, 2),
                    'percentage': round(change_pct, 2)
                }
            })
            
            prev_price = target_price
            
        # Summary
        prices = [p['price'] for p in response_list]
        
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'currency': 'VND' if in_vnd else 'USD',
            'unit': 'VND/luong' if in_vnd else 'USD/oz',
            'exchange_rate': self.usd_vnd_rate,
            'last_known': {
                'date': last_date.strftime('%Y-%m-%d'),
                'price': round(final_last_known, 2)
            },
            'predictions': response_list,
            'summary': {
                'min': round(min(prices), 2),
                'max': round(max(prices), 2),
                'avg': round(sum(prices) / len(prices), 2),
                'change_percent': round((prices[-1] - final_last_known) / final_last_known * 100, 2)
            }
        }
    
    def _convert_to_vnd_single(self, price_usd: float) -> float:
        """Convert USD/oz to VND/luong."""
        return price_usd * self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
    
    def _get_future_trading_dates(self, last_date, num_days: int) -> List:
        """Get future trading dates (skip weekends)."""
        dates = []
        current = last_date
        while len(dates) < num_days:
            current = current + timedelta(days=1)
            if current.weekday() < 5:
                dates.append(current)
        return dates
    
    def get_historical_data(self, days: int = 30, in_vnd: bool = True) -> Dict:
        """Get historical price data."""
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        df = self.data.tail(days).copy()
        
        data_list = []
        for _, row in df.iterrows():
            price = row['price']
            
            # Data is already in USD (normalized in load_data)
            price_usd = price
            
            if in_vnd:
                # Convert USD to VND
                factor = self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
                price = price_usd * factor
            else:
                price = price_usd
            
            data_list.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'price': round(price, 2)
            })
        
        return {
            'success': True,
            'currency': 'VND' if in_vnd else 'USD',
            'unit': 'VND/luong' if in_vnd else 'USD/oz',
            'exchange_rate': self.usd_vnd_rate if in_vnd else None,
            'data': data_list
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': 'Ridge+XGB+LSTM Ensemble' if self.use_lstm else ('Ridge+XGB Ensemble' if self.use_ensemble else 'Ridge Regression'),
            'asset': 'Gold (XAU)',
            'features': len(self.feature_columns) if self.feature_columns else 0,
            'pca_components': self.pca.n_components_ if self.pca else None,
            'metrics': self.metrics,
            'prediction_days': self.prediction_days
        }

    def get_market_drivers(self) -> Dict:
        """Get latest market drivers."""
        if self.data is None:
            return {}
            
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Helper
        def get_val(col): return float(latest.get(col, 0))
        def get_change(col):
            if col not in latest or col not in prev: return 0.0
            val_now = latest[col]
            val_prev = prev[col]
            if val_prev == 0: return 0.0
            return float((val_now - val_prev) / val_prev * 100)
            
        # Features often used: dxy, oil, vix, us10y
        drivers = {
            'dxy': {'value': get_val('dxy'), 'change': get_change('dxy')},
            'vix': {'value': get_val('vix'), 'change': get_change('vix')},
            'oil': {'value': get_val('oil'), 'change': get_change('oil')},
            'us10y': {'value': get_val('us10y'), 'change': get_change('us10y')},
            'rsi': {'value': get_val('gold_rsi'), 'change': get_change('gold_rsi')},
        }
        
        factors = []
        
        # Logic
        if drivers['dxy']['change'] < -0.1:
            factors.append(f"Đồng USD suy yếu ({drivers['dxy']['change']:.2f}%) hỗ trợ giá Vàng")
        elif drivers['dxy']['change'] > 0.1:
            factors.append(f"Đồng USD mạnh lên ({drivers['dxy']['change']:.2f}%) kìm hãm giá Vàng")
            
        if drivers['us10y']['change'] > 1.0:
            factors.append(f"Lợi suất trái phiếu tăng ({drivers['us10y']['change']:.2f}%) gây áp lực giảm")
            
        if drivers['vix']['change'] > 3.0:
            factors.append(f"Tâm lý bất ổn (VIX +{drivers['vix']['change']:.2f}%) kích thích mua Vàng trú ẩn")
            
        if drivers['rsi']['value'] > 75:
            factors.append("RSI báo hiệu thị trường Quá Mua, rủi ro điều chỉnh")
        elif drivers['rsi']['value'] < 25:
             factors.append("RSI báo hiệu thị trường Quá Bán, cơ hội hồi phục")
             
        return {
            'factors': factors, 
            'raw': drivers,
            'ai_explanation': self._generate_ai_explanation(drivers, factors)
        }
    
    def _generate_ai_explanation(self, drivers: dict, factors: list) -> dict:
        """Generate AI explanation of market conditions."""
        explanation = {
            'summary': '',
            'key_factors': [],
            'outlook': '',
            'confidence': 'Medium'
        }
        
        # Analyze key market conditions
        vix_val = drivers.get('vix', {}).get('value', 20)
        dxy_change = drivers.get('dxy', {}).get('change', 0)
        us10y_change = drivers.get('us10y', {}).get('change', 0)
        rsi_val = drivers.get('rsi', {}).get('value', 50)
        
        # Generate summary
        if vix_val > 30:
            explanation['summary'] = "Thị trường đang trong trạng thái 'sợ hãi' với VIX cao"
        elif vix_val > 25:
            explanation['summary'] = "Thị trường có sự bất ổn gia tăng"
        elif dxy_change < -1.0:
            explanation['summary'] = "Đồng USD suy yếu mạnh, hỗ trợ kim loại quý"
        elif dxy_change > 1.0:
            explanation['summary'] = "Đồng USD mạnh lên, gây áp lực lên kim loại quý"
        else:
            explanation['summary'] = "Thị trường đang ở trạng thái tương đối cân bằng"
        
        # Key factors with impact levels
        if vix_val > 25:
            explanation['key_factors'].append({
                'factor': 'VIX cao',
                'impact': 'Tích cực', 
                'reason': 'Nhà đầu tư tìm đến kim loại quý như nơi trú ẩn an toàn',
                'value': f"{vix_val:.1f}"
            })
        
        if dxy_change < -0.5:
            explanation['key_factors'].append({
                'factor': 'USD suy yếu',
                'impact': 'Tích cực',
                'reason': 'Kim loại quý trở nên rẻ hơn đối với người mua bằng ngoại tệ khác',
                'value': f"{dxy_change:.2f}%"
            })
        elif dxy_change > 0.5:
            explanation['key_factors'].append({
                'factor': 'USD mạnh lên', 
                'impact': 'Tiêu cực',
                'reason': 'Kim loại quý trở nên đắt hơn đối với người mua bằng ngoại tệ khác',
                'value': f"{dxy_change:.2f}%"
            })
        
        if us10y_change > 1.0:
            explanation['key_factors'].append({
                'factor': 'Lãi suất tăng',
                'impact': 'Tiêu cực',
                'reason': 'Chi phí cơ hội giữ kim loại quý tăng lên',
                'value': f"+{us10y_change:.2f}%"
            })
        
        if rsi_val > 75:
            explanation['key_factors'].append({
                'factor': 'RSI quá mua',
                'impact': 'Tiêu cực', 
                'reason': 'Có thể sắp xảy ra điều chỉnh giá',
                'value': f"{rsi_val:.0f}"
            })
        elif rsi_val < 25:
            explanation['key_factors'].append({
                'factor': 'RSI quá bán',
                'impact': 'Tích cực',
                'reason': 'Có thể sắp xảy ra hồi phục giá', 
                'value': f"{rsi_val:.0f}"
            })
        
        # Generate outlook
        positive_factors = sum(1 for f in explanation['key_factors'] if f['impact'] == 'Tích cực')
        negative_factors = sum(1 for f in explanation['key_factors'] if f['impact'] == 'Tiêu cực')
        
        if positive_factors > negative_factors:
            explanation['outlook'] = "Xu hướng tăng giá có khả năng cao trong ngắn hạn"
            explanation['confidence'] = 'High' if positive_factors >= 3 else 'Medium'
        elif negative_factors > positive_factors:
            explanation['outlook'] = "Xu hướng giảm giá có thể xảy ra trong ngắn hạn"
            explanation['confidence'] = 'High' if negative_factors >= 3 else 'Medium'
        else:
            explanation['outlook'] = "Thị trường có thể đi ngang trong ngắn hạn"
            explanation['confidence'] = 'Medium'
        
        return explanation

    def get_yesterday_accuracy(self) -> Dict:
        """Simulate yesterday prediction."""
        if self.data is None or len(self.data) < 10:
            return None
            
        # Ridge features at t-1
        features_yesterday = self.data[self.feature_columns].iloc[-2].values.reshape(1, -1)
        if self.scaler:
            feat_full = self.scaler.transform(features_yesterday)
            feat_for_ridge = self.pca.transform(feat_full) if self.pca else feat_full
        else:
            feat_full = features_yesterday
            feat_for_ridge = features_yesterday

        # Predict Day 1 with ensemble
        base_price = self.data['price'].iloc[-2]
        ridge_pred = self.models[1].predict(feat_for_ridge)[0]

        # Active weights
        w_r = self.ensemble_weights.get('ridge', 1.0)
        w_x = self.ensemble_weights.get('xgboost', 0.0) if (self.use_ensemble and self.xgb_models) else 0
        w_l = self.ensemble_weights.get('lstm', 0.0) if (self.use_lstm and self.lstm_model) else 0
        total_w = w_r + w_x + w_l
        if total_w > 0:
            w_r, w_x, w_l = w_r / total_w, w_x / total_w, w_l / total_w

        pred_usd = w_r * ridge_pred

        if w_x > 0 and 1 in self.xgb_models:
            xgb_features = self.data[self.xgb_feature_columns].iloc[-2:-1].values
            xgb_scaled = self.xgb_scaler.transform(xgb_features)
            xgb_return = self.xgb_models[1].predict(xgb_scaled)[0]
            xgb_pred = base_price * (1 + xgb_return)
            pred_usd += w_x * xgb_pred

        if w_l > 0 and self.lstm_model is not None:
            seq_len = self.lstm_sequence_length
            end_idx = len(self.data) - 1
            if end_idx >= seq_len:
                lstm_input = self.data[self.feature_columns].iloc[end_idx - seq_len:end_idx].values
                lstm_input_scaled = self.scaler.transform(lstm_input).astype(np.float32)
                lstm_returns = self.lstm_model.predict(
                    lstm_input_scaled.reshape(1, seq_len, -1), verbose=0)[0]
                pred_usd += w_l * (base_price * (1 + lstm_returns[0]))
        
        # Actual
        actual_usd = self.data['price'].iloc[-1] # price is already USD/oz
        
        diff = abs(pred_usd - actual_usd)
        error_pct = (diff / actual_usd) * 100
        
        return {
            'predicted_usd': float(pred_usd),
            'actual_usd': float(actual_usd),
            'diff_usd': float(diff),
            'accuracy': float(100 - error_pct),
            'date': self.data['date'].iloc[-1].strftime('%Y-%m-%d')
        }
    
    def set_exchange_rate(self, rate: float):
        """Set USD to VND exchange rate."""
        self.usd_vnd_rate = rate


def main():
    """Train gold model with Extended Features."""
    print("=" * 60)
    print("GOLD PRICE PREDICTOR - Extended Features Training")
    print("=" * 60)
    
    predictor = GoldPredictor()
    predictor.load_data()
    predictor.create_features()
    predictor.train(test_size=0.2, use_pca=True, alpha=1.0)
    predictor.save_model()
    
    print("\n--- Testing Prediction ---")
    result = predictor.predict(in_vnd=True)
    print(f"Last known: {result['last_known']}")
    print("7-day forecast:")
    for p in result['predictions']:
        print(f"  Day {p['day']}: {p['price']:,.0f} VND/luong")
    print(f"Summary: {result['summary']}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
