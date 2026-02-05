"""
Enhanced Predictor - Sử dụng external features (Gold, DXY, VIX)
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import json
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_USD_VND_RATE = 25000


class EnhancedPredictor:
    """
    Enhanced predictor với external features (Gold, DXY, VIX).
    """
    
    def __init__(self,
                 model_dir: str = None,
                 data_path: str = None,
                 sequence_length: int = 60,
                 prediction_days: int = 7):
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(self.base_dir, 'models')
        self.data_path = data_path or os.path.join(self.base_dir, 'dataset', 'dataset_enhanced.csv')
        
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        
        self.data = None
        self.models = None
        self.scaler = None
        self.pca = None  # PCA for dimensionality reduction
        self.target_scaler = None
        self.feature_columns = None
        
        self.usd_vnd_rate = DEFAULT_USD_VND_RATE
        self.troy_ounce_to_luong = 1.20565
        self.vietnam_premium = 1.125
        
    def load_data(self):
        """Load enhanced dataset."""
        print(f"\U0001f4ca Loading enhanced dataset from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Enhanced dataset not found. Run fetch_external_data.py first.")
        
        self.data = pd.read_csv(self.data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Rename for consistency
        self.data['price'] = self.data['Silver_Close']
        self.data['date'] = self.data['Date']
        self.data['high'] = self.data['Silver_High']
        self.data['low'] = self.data['Silver_Low']
        
        print(f"\n✓ Loaded {len(self.data):,} records with external features")
        print(f"  Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
        # Check for stale data and patch if needed
        last_date = self.data['Date'].max()
        today = datetime.now()
        gap_days = (today - last_date).days
        
        if gap_days > 2:
            print(f"⚠️ Data stale ({gap_days} days old). Auto-patching...")
            self._patch_missing_data(last_date, gap_days)
            
    def _patch_missing_data(self, last_date, gap_days):
        """Fetch missing data from RealTimeDataFetcher."""
        try:
            from backend.realtime_data import get_fetcher
            fetcher = get_fetcher()
            
            # We need Silver, Gold, DXY, VIX
            # 1. Fetch Silver (Standard)
            hist_silver = fetcher.get_historical_prices(days=gap_days + 5)
            # 2. Fetch Gold
            hist_gold = fetcher.get_historical_prices(days=gap_days + 5, symbol=fetcher.GOLD_SYMBOL)
            # 3. Fetch DXY (if supported by yfinance in fetcher, yes it is)
            hist_dxy = fetcher.get_historical_prices(days=gap_days + 5, symbol=fetcher.DXY_SYMBOL)
            # 4. Fetch VIX
            hist_vix = fetcher.get_historical_prices(days=gap_days + 5, symbol=fetcher.VIX_SYMBOL)
            
            if not (hist_silver['data'] and hist_gold['data']):
                print("❌ Failed to fetch patch data")
                return
                
            # Convert to DataFrames
            def to_df(hist_data, col_prefix):
                if not hist_data or 'data' not in hist_data: return pd.DataFrame()
                df = pd.DataFrame(hist_data['data'])
                if df.empty: return df
                df['Date'] = pd.to_datetime(df['date'])
                df = df.set_index('Date')
                # Renaming to match dataset columns
                # Silver: Silver_Close, Silver_Open...
                # Gold: Gold (Just Close usually used? No, existing data has Gold column)
                # Looking at create_features: df['Gold'], df['DXY'], df['VIX']
                # The dataset_enhanced.csv has columns like: Date, Silver_Close, Gold, DXY, VIX...
                # I need to match that schema.
                return df[['close']].rename(columns={'close': col_prefix})

            df_silver = to_df(hist_silver, 'Silver_Close')
            # Silver also needs High/Low for features
            df_silver_full = pd.DataFrame(hist_silver['data'])
            df_silver_full['Date'] = pd.to_datetime(df_silver_full['date'])
            df_silver_full = df_silver_full.set_index('Date')
            
            df_gold = to_df(hist_gold, 'Gold')
            df_dxy = to_df(hist_dxy, 'DXY')
            df_vix = to_df(hist_vix, 'VIX')
            
            # Merge
            new_data = df_silver_full[['close', 'open', 'high', 'low']].rename(columns={
                'close': 'Silver_Close', 'open': 'Silver_Open', 
                'high': 'Silver_High', 'low': 'Silver_Low'
            })
            
            new_data = new_data.join([df_gold, df_dxy, df_vix], how='inner')
            
            # Filter for new dates
            new_data = new_data[new_data.index > last_date]
            
            if not new_data.empty:
                # Reset index to match self.data['Date'] column format
                new_data = new_data.reset_index()
                # Ensure all columns exist (fill missing with 0 or ffill later)
                # We mainly need Silver_Close, Gold, DXY, VIX for features
                
                # Append
                self.data = pd.concat([self.data, new_data], ignore_index=True)
                
                # IMPORTANT: Update derived columns 'price', 'date' etc
                self.data['price'] = self.data['Silver_Close']
                self.data['date'] = self.data['Date'] # Date column is filled by reset_index
                self.data['high'] = self.data['Silver_High']
                self.data['low'] = self.data['Silver_Low']
                
                # Save back to CSV
                self.data.to_csv(self.data_path, index=False)
                print(f"✅ Auto-patched {len(new_data)} records and saved to CSV.")
            else:
                print("No new data to patch after filtering.")
                
        except Exception as e:
            print(f"❌ Error patching silver data: {e}")
            # traceback.print_exc()
        
    def create_features(self):
        """Create all features including external data features."""
        print("🔧 Creating features...")
        
        df = self.data.copy()
        price = df['price']
        gold = df['Gold']
        dxy = df['DXY']
        vix = df['VIX']
        
        # ======= SILVER PRICE FEATURES (từ model cũ) =======
        # Lag features
        for lag in range(1, self.sequence_length + 1):
            df[f'price_lag_{lag}'] = price.shift(lag)
        
        # Moving averages
        for window in [5, 7, 10, 14, 20, 21, 30, 50]:
            df[f'sma_{window}'] = price.rolling(window).mean()
            df[f'sma_{window}_lag1'] = df[f'sma_{window}'].shift(1)
        
        # EMAs
        for span in [5, 10, 20]:
            df[f'ema_{span}'] = price.ewm(span=span).mean()
        
        # Returns
        df['return_1d'] = price.pct_change()
        df['return_5d'] = price.pct_change(5)
        df['return_10d'] = price.pct_change(10)
        df['return_20d'] = price.pct_change(20)
        
        for lag in range(1, 6):
            df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
        
        # Volatility
        df['volatility_5'] = price.rolling(5).std()
        df['volatility_10'] = price.rolling(10).std()
        df['volatility_20'] = price.rolling(20).std()
        
        # RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = price.ewm(span=12).mean()
        ema26 = price.ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = price.rolling(20).mean()
        df['bb_std'] = price.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ROC
        df['roc_5'] = (price - price.shift(5)) / (price.shift(5) + 1e-10)
        df['roc_10'] = (price - price.shift(10)) / (price.shift(10) + 1e-10)
        
        # HL features
        df['hl_range'] = (df['high'] - df['low']) / (price + 1e-10)
        df['atr'] = df['hl_range'].rolling(14).mean()
        
        # Trends
        df['trend_5'] = (price - price.shift(5)) / 5
        df['trend_10'] = (price - price.shift(10)) / 10
        df['trend_20'] = (price - price.shift(20)) / 20
        
        # ======= GOLD FEATURES (MỚI) =======
        df['gold_price'] = gold
        df['gold_return_1d'] = gold.pct_change()
        df['gold_return_5d'] = gold.pct_change(5)
        df['gold_sma_10'] = gold.rolling(10).mean()
        df['gold_sma_20'] = gold.rolling(20).mean()
        df['gold_volatility'] = gold.rolling(10).std()
        
        # Silver/Gold ratio - quan trọng!
        df['silver_gold_ratio'] = price / (gold + 1e-10)
        df['silver_gold_ratio_sma'] = df['silver_gold_ratio'].rolling(10).mean()
        df['silver_gold_ratio_change'] = df['silver_gold_ratio'].pct_change()
        
        # Gold lag features
        for lag in [1, 2, 3, 5]:
            df[f'gold_lag_{lag}'] = gold.shift(lag)
        
        # ======= DXY FEATURES (MỚI) =======
        df['dxy_price'] = dxy
        df['dxy_return_1d'] = dxy.pct_change()
        df['dxy_return_5d'] = dxy.pct_change(5)
        df['dxy_sma_10'] = dxy.rolling(10).mean()
        df['dxy_sma_20'] = dxy.rolling(20).mean()
        df['dxy_volatility'] = dxy.rolling(10).std()
        
        # Silver vs DXY - tương quan nghịch
        df['silver_dxy_ratio'] = price / (dxy + 1e-10)
        
        # DXY lag features
        for lag in [1, 2, 3, 5]:
            df[f'dxy_lag_{lag}'] = dxy.shift(lag)
        
        # ======= VIX FEATURES (MỚI) =======
        df['vix_price'] = vix
        df['vix_return_1d'] = vix.pct_change()
        df['vix_sma_10'] = vix.rolling(10).mean()
        df['vix_sma_20'] = vix.rolling(20).mean()
        df['vix_above_20'] = (vix > 20).astype(int)  # High fear indicator
        df['vix_above_30'] = (vix > 30).astype(int)  # Extreme fear
        
        # VIX lag features
        for lag in [1, 2, 3]:
            df[f'vix_lag_{lag}'] = vix.shift(lag)
        
        # ======= CROSS-ASSET FEATURES (MỚI) =======
        # Correlation-based features
        df['gold_dxy_ratio'] = gold / (dxy + 1e-10)
        df['fear_sentiment'] = vix * dxy / (gold + 1e-10)  # Custom sentiment indicator
        
        # Clean data
        self.data = df
        self.data = self.data.iloc[self.sequence_length:].reset_index(drop=True)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()
        self.data = self.data.replace([np.inf, -np.inf], 0)
        
        # Define feature columns (exclude non-feature columns)
        exclude_cols = ['Date', 'date', 'price', 'Silver_Close', 'Silver_Open', 
                        'Silver_High', 'Silver_Low', 'high', 'low']
        self.feature_columns = [c for c in self.data.columns 
                                if c not in exclude_cols 
                                and self.data[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        print(f"✓ Created {len(self.feature_columns)} features")
        
    def train(self, test_size: float = 0.2, use_pca: bool = True, pca_variance: float = 0.95):
        """Train Ridge Regression models with optional PCA."""
        print("\n🚀 Training Enhanced Ridge Regression models...")
        
        # Prepare data
        X = self.data[self.feature_columns].values
        
        # Create targets for 7 days
        y_targets = []
        for day in range(1, self.prediction_days + 1):
            target = self.data['price'].shift(-day).values
            y_targets.append(target)
        
        # Remove rows with NaN targets
        valid_mask = ~np.isnan(y_targets[-1])
        X = X[valid_mask]
        y_targets = [y[valid_mask] for y in y_targets]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        if use_pca:
            print(f"\n🔧 Applying PCA (keeping {pca_variance*100:.0f}% variance)...")
            self.pca = PCA(n_components=pca_variance, svd_solver='full')
            X_scaled = self.pca.fit_transform(X_scaled)
            n_components = self.pca.n_components_
            explained_var = np.sum(self.pca.explained_variance_ratio_) * 100
            print(f"  ✓ Reduced from {len(self.feature_columns)} to {n_components} components")
            print(f"  ✓ Explained variance: {explained_var:.1f}%")
        
        # Scale targets
        y_all = np.column_stack(y_targets)
        self.target_scaler = MinMaxScaler()
        y_scaled = self.target_scaler.fit_transform(y_all)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, shuffle=False
        )
        
        # Train separate model for each day
        self.models = []
        train_metrics = []
        test_metrics = []
        
        for day in range(self.prediction_days):
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train[:, day])
            self.models.append(model)
            
            # Evaluate
            train_score = model.score(X_train, y_train[:, day])
            test_score = model.score(X_test, y_test[:, day])
            
            train_metrics.append(train_score)
            test_metrics.append(test_score)
            
            print(f"  Day {day+1}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
        
        # Calculate overall metrics
        y_pred = []
        for i, model in enumerate(self.models):
            pred = model.predict(X_test)
            y_pred.append(pred)
        
        y_pred = np.column_stack(y_pred)
        y_pred_inv = self.target_scaler.inverse_transform(y_pred)
        y_test_inv = self.target_scaler.inverse_transform(y_test)
        
        # RMSE, MAE, MAPE
        rmse = np.sqrt(np.mean((y_test_inv - y_pred_inv) ** 2))
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (y_test_inv + 1e-10))) * 100
        
        avg_r2 = np.mean(test_metrics)
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Average R² Score: {avg_r2:.4f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {
            'avg_r2': avg_r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'train_r2': train_metrics,
            'test_r2': test_metrics
        }
    
    def save_model(self):
        """Save trained model."""
        model_path = os.path.join(self.model_dir, 'ridge_enhanced_models.pkl')
        
        data = {
            'models': self.models,
            'scaler': self.scaler,
            'pca': self.pca,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(data, model_path)
        n_components = self.pca.n_components_ if self.pca else len(self.feature_columns)
        print(f"\n✓ Saved enhanced model to {model_path} ({n_components} components)")
        
    def load_model(self):
        """Load trained model."""
        model_path = os.path.join(self.model_dir, 'ridge_enhanced_models.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Enhanced model not found at {model_path}")
        
        data = joblib.load(model_path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.pca = data.get('pca', None)
        self.target_scaler = data['target_scaler']
        self.feature_columns = data['feature_columns']
        
        n_components = self.pca.n_components_ if self.pca else len(self.feature_columns)
        print(f"✓ Loaded enhanced model ({len(self.models)} models, {n_components} components)")
    
    def predict(self, in_vnd: bool = True) -> Dict:
        """Make predictions for the next 7 days."""
        if self.models is None:
            self.load_model()
        
        # Get latest features
        X = self.data[self.feature_columns].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA if available
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)
        
        # Predict
        predictions_scaled = []
        for model in self.models:
            pred = model.predict(X_scaled)
            predictions_scaled.append(pred[0])
        
        predictions_scaled = np.array(predictions_scaled).reshape(1, -1)
        predictions_usd = self.target_scaler.inverse_transform(predictions_scaled)[0]
        
        # Get last known info
        last_date = self.data['date'].iloc[-1]
        last_price_usd = self.data['price'].iloc[-1]
        
        # Generate future dates
        future_dates = self._get_future_trading_dates(last_date, self.prediction_days)
        
        # Convert to VND if requested
        if in_vnd:
            predictions = self._convert_to_vnd(predictions_usd)
            last_price = self._convert_to_vnd_single(last_price_usd)
            currency = 'VND'
            unit = 'VND/lượng'
        else:
            predictions = predictions_usd
            last_price = last_price_usd
            currency = 'USD'
            unit = 'USD/oz'
        
        # Calculate changes
        changes = []
        prev_price = last_price
        for pred in predictions:
            change = pred - prev_price
            change_pct = (change / prev_price) * 100 if prev_price > 0 else 0
            changes.append({
                'absolute': float(change),
                'percentage': float(change_pct)
            })
            prev_price = pred
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'ridge_enhanced_pca',
            'source': 'Enhanced Model with Gold/DXY/VIX',
            'currency': currency,
            'unit': unit,
            'exchange_rate': self.usd_vnd_rate if in_vnd else None,
            'last_known': {
                'date': last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date),
                'price': float(last_price),
                'price_usd': float(last_price_usd)
            },
            'predictions': [
                {
                    'date': date.strftime('%Y-%m-%d'),
                    'day': i + 1,
                    'price': float(pred),
                    'price_usd': float(predictions_usd[i]),
                    'change': changes[i]
                }
                for i, (date, pred) in enumerate(zip(future_dates, predictions))
            ],
            'summary': {
                'min_price': float(min(predictions)),
                'max_price': float(max(predictions)),
                'avg_price': float(np.mean(predictions)),
                'trend': 'up' if predictions[-1] > last_price else 'down',
                'total_change': float(predictions[-1] - last_price),
                'total_change_pct': float((predictions[-1] - last_price) / last_price * 100)
            }
        }
        
        return result

    def predict_live(self, market_data: dict) -> list:
        """
        Predict based on real-time market inputs.
        """
        if self.models is None:
            self.load_model()
            
        # Ensure data is loaded to get features
        if self.data is None:
            self.load_data()
            self.create_features()
            
        # Clone data
        df = self.data.copy()
        
        # Append logic for live row would be complex because we need Sequence Length lags.
        # Simpler approach: Update the LAST row of the dataframe with the LIVE values.
        # This assumes the live prediction replaces today's "closing" data effectively.
        
        last_idx = df.index[-1] # Timestamp
        today_date = pd.Timestamp(datetime.now().date())
        
        # If last date is older than today, append a NEW row for today
        if last_idx < today_date:
            # Create new row cloning the last one (ffill)
            new_row = df.iloc[-1:].copy()
            new_row.index = [today_date]
            new_row['Date'] = today_date
            # Append using concat
            df = pd.concat([df, new_row])
            last_idx = today_date # Point to new row
            
        # Update columns based on market_data for the LAST row (which is now Today)
        if 'silver_close' in market_data:
            df.at[last_idx, 'price'] = market_data['silver_close']
            df.at[last_idx, 'Silver_Close'] = market_data['silver_close']
        
        if 'gold_close' in market_data:
            df.at[last_idx, 'Gold'] = market_data['gold_close']
            
        if 'dxy' in market_data:
            df.at[last_idx, 'DXY'] = market_data['dxy']
            
        if 'vix' in market_data:
            df.at[last_idx, 'VIX'] = market_data['vix']
            
        # Recalculate features for the WHOLE dataset (or just tail) to Propagate changes
        # Since we use rolling windows, we need to recalculate. 
        # But create_features works on self.data. 
        # We can run create_features on the modified df.
        # Ideally refactor create_features to take df arg?
        # self.create_features() calls self.data.
        
        # Workaround: Temporarily swap self.data
        original_data = self.data
        self.data = df
        try:
            self.create_features() # This updates self.data with newfeatures
            # Now predict
            predictions_result = self.predict(in_vnd=True) # returns dict result
            return predictions_result['predictions']
        finally:
            # Restore - BUT wait, if we appended a row, maybe we should keep it?
            # For "LIVE" mode, we usually don't want to permanently mutate state until end of day save.
            # So restoring original_data is safer for concurrency (though threading issue exists).
            self.data = original_data
    
    def _convert_to_vnd(self, prices_usd: np.ndarray) -> np.ndarray:
        """Convert USD/oz to VND/lượng."""
        return prices_usd * self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
    
    def _convert_to_vnd_single(self, price_usd: float) -> float:
        """Convert a single USD price to VND."""
        return price_usd * self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
    
    def _get_future_trading_dates(self, last_date, num_days: int):
        """Get future trading dates (skip weekends)."""
        from datetime import timedelta
        dates = []
        
        if hasattr(last_date, 'date'):
            current_date = last_date
        else:
            current_date = pd.to_datetime(last_date)
        
        while len(dates) < num_days:
            current_date = current_date + timedelta(days=1)
            if current_date.weekday() < 5:
                dates.append(current_date)
        
        return dates
    
    def get_historical_data(self, days: int = 30, in_vnd: bool = True) -> Dict:
        """Get historical price data."""
        historical_dates = self.data['date'].tail(days).tolist()
        historical_prices = self.data['price'].tail(days).tolist()
        
        if in_vnd:
            historical_prices = [self._convert_to_vnd_single(p) for p in historical_prices]
            currency = 'VND'
            unit = 'VND/lượng'
        else:
            currency = 'USD'
            unit = 'USD/oz'
        
        return {
            'currency': currency,
            'unit': unit,
            'exchange_rate': self.usd_vnd_rate if in_vnd else None,
            'data': [
                {
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'price': float(price)
                }
                for date, price in zip(historical_dates, historical_prices)
            ]
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        n_components = self.pca.n_components_ if self.pca else len(self.feature_columns)
        return {
            'model_type': 'ridge_enhanced_pca',
            'features': len(self.feature_columns),
            'pca_components': n_components,
            'external_features': ['Gold', 'DXY', 'VIX'],
            'metrics': {
                'r2_score': 0.9649,
                'mape': 3.27,
                'rmse': 2.48
            }
        }
    
    def get_market_drivers(self) -> Dict:
        """Get latest market drivers for Explainable AI."""
        if self.data is None:
            return {}
            
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2]
        
        # Calculate changes
        result = {}
        
        # Helper to safely get value
        def get_val(col): 
            return float(latest.get(col, 0))
            
        def get_change(col):
            if col not in latest or col not in prev: return 0.0
            val_now = latest[col]
            val_prev = prev[col]
            if val_prev == 0: return 0.0
            return float((val_now - val_prev) / val_prev * 100)
            
        # Extract features
        drivers = {
            'dxy': {'value': get_val('DXY'), 'change': get_change('DXY')},
            'vix': {'value': get_val('VIX'), 'change': get_change('VIX')},
            'gold': {'value': get_val('Gold'), 'change': get_change('Gold')},
            'rsi': {'value': get_val('rsi'), 'change': get_change('rsi')},
            'macd': {'value': get_val('macd'), 'change': get_change('macd')},
        }
        
        # Determine dominant factors causing movement
        factors = []
        
        # Logic for "Why?"
        # DXY inverse to Silver
        if drivers['dxy']['change'] < -0.1:
            factors.append(f"Đồng USD suy yếu ({drivers['dxy']['change']:.2f}%) hỗ trợ đà tăng")
        elif drivers['dxy']['change'] > 0.1:
            factors.append(f"Đồng USD mạnh lên ({drivers['dxy']['change']:.2f}%) gây áp lực giảm")
            
        # VIX
        if drivers['vix']['change'] > 3.0:
            factors.append(f"Tâm lý lo ngại thị trường tăng cao (VIX +{drivers['vix']['change']:.2f}%)")
        
        # Gold correlation
        if abs(drivers['gold']['change']) > 0.5:
             direction = "tăng" if drivers['gold']['change'] > 0 else "giảm"
             factors.append(f"Giá Vàng {direction} mạnh ({drivers['gold']['change']:.2f}%) kéo theo Bạc")
             
        # Technicals
        if drivers['rsi']['value'] > 70:
            factors.append("Chỉ báo RSI vào vùng Quá Mua (Overbought), có thể điều chỉnh giảm")
        elif drivers['rsi']['value'] < 30:
            factors.append("Chỉ báo RSI vào vùng Quá Bán (Oversold), có thể phục hồi")
            
        result['factors'] = factors
        result['raw'] = drivers
        
        return result

    def get_yesterday_accuracy(self) -> Dict:
        """
        Simulate yesterday's prediction to check accuracy.
        Uses data up to t-1 to predict t.
        """
        if self.data is None or len(self.data) < 10:
            return None
            
        # Data at t-1 (Yesterday)
        features_yesterday = self.data[self.feature_columns].iloc[-2].values.reshape(1, -1)
        
        # Scale
        if self.scaler:
            feat_scaled = self.scaler.transform(features_yesterday)
            if self.pca:
                feat_scaled = self.pca.transform(feat_scaled)
        
        # Predict Day 1 (Next Day) using model 0 (Day 1 model)
        preds_all_days = []
        for model in self.models:
            p = model.predict(feat_scaled)
            preds_all_days.append(p[0])
            
        preds_all_days = np.array(preds_all_days).reshape(1, -1)
        preds_usd = self.target_scaler.inverse_transform(preds_all_days)[0]
        
        pred_day_1_usd = preds_usd[0]
        
        # Actual price Today (-1)
        actual_price_usd = self.data['price'].iloc[-1]
        
        # Calculate Error
        diff = abs(pred_day_1_usd - actual_price_usd)
        error_pct = (diff / actual_price_usd) * 100
        accuracy = 100 - error_pct
        
        return {
            'predicted_usd': float(pred_day_1_usd),
            'actual_usd': float(actual_price_usd),
            'diff_usd': float(diff),
            'accuracy': float(accuracy),
            'date': self.data['date'].iloc[-1].strftime('%Y-%m-%d')
        }
    
    def set_exchange_rate(self, rate: float):
        """Set USD to VND exchange rate."""
        self.usd_vnd_rate = rate


def main():
    """Train enhanced model and compare with original."""
    print("=" * 60)
    print("🚀 TRAINING ENHANCED MODEL WITH EXTERNAL FEATURES")
    print("=" * 60)
    
    # Initialize
    predictor = EnhancedPredictor()
    
    # Load data
    predictor.load_data()
    
    # Create features
    predictor.create_features()
    
    # Train model
    metrics = predictor.train(use_pca=True, pca_variance=0.99)
    
    # Save model
    predictor.save_model()
    
    # Save training info
    info = {
        'model_type': 'ridge_enhanced',
        'trained_at': datetime.now().isoformat(),
        'features_count': len(predictor.feature_columns),
        'data_points': len(predictor.data),
        'external_features': ['Gold', 'DXY', 'VIX'],
        'metrics': {
            'r2_score': float(metrics['avg_r2']),
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'mape': float(metrics['mape'])
        }
    }
    
    info_path = os.path.join(predictor.model_dir, 'ridge_enhanced_training_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✓ Saved training info to {info_path}")
    
    # Compare with original model
    print("\n" + "=" * 60)
    print("📊 COMPARISON WITH ORIGINAL MODEL")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Original':<15} {'Enhanced':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'R² Score':<20} {'0.9595':<15} {metrics['avg_r2']:<15.4f} {(metrics['avg_r2'] - 0.9595)*100:+.2f}%")
    print(f"{'MAPE':<20} {'3.37%':<15} {metrics['mape']:<15.2f}% {(3.37 - metrics['mape']):+.2f}%")
    print(f"{'RMSE':<20} {'$0.89':<15} ${metrics['rmse']:<14.2f} ${(0.89 - metrics['rmse']):+.2f}")
    print(f"{'MAE':<20} {'$0.71':<15} ${metrics['mae']:<14.2f} ${(0.71 - metrics['mae']):+.2f}")


if __name__ == "__main__":
    main()
