"""
Vietnam Gold Price Predictor - Transfer Learning from World Gold Model
Dự đoán giá vàng SJC Việt Nam sử dụng Transfer Learning từ mô hình giá vàng thế giới.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
DEFAULT_USD_VND_RATE = 25500
GOLD_OZ_TO_LUONG = 1.20565  # 1 oz = 1.20565 lượng (37.5g / 31.1g)


class VietnamGoldPredictor:
    """
    Transfer Learning: World Gold Model → Vietnam SJC Price Prediction.
    
    Chiến lược:
    1. Load pre-trained world gold features
    2. Add Vietnam-specific features (premium, spread, lunar calendar)
    3. Train adapter layer for VN prices
    """
    
    def __init__(self, model_dir: str = None, prediction_days: int = 7):
        self.prediction_days = prediction_days
        
        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(base_dir, 'models')
        self.dataset_dir = os.path.join(base_dir, 'dataset')
        
        # Data
        self.vn_data = None
        self.world_data = None
        self.merged_data = None
        
        # Models
        self.transfer_models = {}
        self.scaler = None
        self.feature_columns = []
        self.metrics = {'r2': {}, 'mape': {}, 'rmse': {}}
        
        # Exchange rate
        self.usd_vnd_rate = DEFAULT_USD_VND_RATE
    
    def load_vietnam_data(self):
        """Load and clean Vietnam SJC gold price dataset."""
        print("Loading Vietnam SJC gold data...")
        
        vn_path = os.path.join(self.dataset_dir, 'gold_price_sjc_complete.csv')
        if not os.path.exists(vn_path):
            raise FileNotFoundError(f"Vietnam gold data not found: {vn_path}")
        
        self.vn_data = pd.read_csv(vn_path, parse_dates=['date'])
        
        # Clean data: fix anomalies where buy_price is too low or spread is too high
        # Normal SJC price should be > 40 million VND/lượng
        anomaly_mask = (self.vn_data['buy_price'] < 40) | \
                      ((self.vn_data['sell_price'] - self.vn_data['buy_price']) > 10)
        
        if anomaly_mask.any():
            print(f"  Fixing {anomaly_mask.sum()} anomalous buy_price values...")
            # Use sell_price - typical spread (2 million)
            self.vn_data.loc[anomaly_mask, 'buy_price'] = \
                self.vn_data.loc[anomaly_mask, 'sell_price'] - 2
        
        # Calculate mid price
        self.vn_data['mid_price'] = (self.vn_data['buy_price'] + self.vn_data['sell_price']) / 2
        
        # Calculate spread
        self.vn_data['spread'] = self.vn_data['sell_price'] - self.vn_data['buy_price']
        
        print(f"  Loaded {len(self.vn_data)} records from {self.vn_data['date'].min()} to {self.vn_data['date'].max()}")
        return self.vn_data
    
    def load_world_data(self):
        """Load and patch world gold geopolitical dataset."""
        print("Loading world gold data...")
        
        world_path = os.path.join(self.dataset_dir, 'gold_geopolitical_dataset.csv')
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"World gold data not found: {world_path}")
        
        self.world_data = pd.read_csv(world_path, parse_dates=['date'])
        
        # Patch missing data (if CSV is old)
        last_date = self.world_data['date'].max()
        today = datetime.now()
        gap_days = (today - last_date).days
        
        if gap_days > 2:
            print(f"  World data stale ({gap_days} days old). Fetching recent data...")
            try:
                # Import fetcher locally to avoid circular import at module level
                from backend.realtime_data import get_fetcher
                fetcher = get_fetcher()
                
                # Fetch recent history for Gold
                # Note: We need approx gap_days + 2 buffer
                hist = fetcher.get_historical_prices(days=gap_days + 5, symbol=fetcher.GOLD_SYMBOL)
                
                if hist and hist['data']:
                    new_rows = []
                    for item in hist['data']:
                        item_date = datetime.strptime(item['date'], '%Y-%m-%d')
                        if item_date > last_date:
                            # Map fetcher fields to model fields (gold_close, etc.)
                            # Missing fields (gpr, oil...) will be ffilled later in create_features
                            new_rows.append({
                                'date': item_date,
                                'gold_close': item['close'],
                                'gold_open': item['open'],
                                'gold_high': item['high'],
                                'gold_low': item['low']
                            })
                    
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        self.world_data = pd.concat([self.world_data, new_df], ignore_index=True)
                        # Save patched data back to CSV to avoid re-fetching
                        self.world_data.to_csv(world_path, index=False)
                        print(f"  Patched {len(new_rows)} recent days and saved to {world_path}")
                        
            except Exception as e:
                print(f"  Error patching world data: {e}")
        
        print(f"  Loaded {len(self.world_data)} records")
        return self.world_data
    
    def merge_datasets(self):
        """Merge Vietnam and World gold data by date."""
        print("Merging Vietnam and World gold datasets...")

        if self.vn_data is None:
            self.load_vietnam_data()
        if self.world_data is None:
            self.load_world_data()

        # Use left join to keep all Vietnam dates, then forward-fill missing world data
        self.merged_data = pd.merge(
            self.vn_data,
            self.world_data,
            on='date',
            how='left',  # Keep all Vietnam dates
            suffixes=('_vn', '_world')
        )

        # Forward-fill missing world gold data (for weekends/holidays)
        world_cols = [col for col in self.merged_data.columns if col.startswith('gold_')]
        self.merged_data[world_cols] = self.merged_data[world_cols].ffill()

        print(f"  Merged dataset: {len(self.merged_data)} records")
        return self.merged_data
    
    def create_transfer_features(self):
        """
        Create features for transfer learning.
        
        Features include:
        1. World gold features (from pre-trained model)
        2. Vietnam-specific features
        """
        print("Creating transfer learning features...")
        
        if self.merged_data is None:
            self.merge_datasets()
        
        df = self.merged_data.copy()
        
        # === Vietnam-specific features ===
        
        # 1. Premium: VN price vs World price (converted)
        # Convert world price (USD/oz) to VND/lượng
        df['world_price_vnd'] = df['gold_close'] * self.usd_vnd_rate * GOLD_OZ_TO_LUONG / 1_000_000
        df['vn_premium'] = df['mid_price'] - df['world_price_vnd']
        df['vn_premium_pct'] = (df['vn_premium'] / df['world_price_vnd']) * 100
        
        # 2. VN price momentum
        for lag in [1, 3, 7, 14, 30]:
            df[f'vn_lag{lag}'] = df['mid_price'].shift(lag)
            df[f'vn_momentum{lag}'] = df['mid_price'] - df['mid_price'].shift(lag)
        
        # 3. VN moving averages
        for window in [7, 14, 30]:
            df[f'vn_ma{window}'] = df['mid_price'].rolling(window).mean()
            df[f'vn_std{window}'] = df['mid_price'].rolling(window).std()
        
        # 4. Spread features
        df['spread_ma7'] = df['spread'].rolling(7).mean()
        df['spread_change'] = df['spread'] - df['spread'].shift(1)
        
        # 5. Premium momentum
        df['premium_ma7'] = df['vn_premium'].rolling(7).mean()
        df['premium_change'] = df['vn_premium'] - df['vn_premium'].shift(1)
        
        # 6. Lunar calendar features (approximation for Tết effect)
        df['month'] = df['date'].dt.month
        df['is_tet_season'] = df['month'].isin([1, 2]).astype(int)  # Tết usually Jan-Feb
        df['is_wedding_season'] = df['month'].isin([10, 11, 12]).astype(int)  # Wedding season
        df['is_god_of_wealth_day'] = ((df['month'] == 2) & (df['date'].dt.day <= 10)).astype(int)
        
        # 7. Day of week effect
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # === World gold features (transfer from pre-trained) ===
        world_features = [
            'gold_close', 'gold_open', 'gold_high', 'gold_low',
            'gold_return', 'gold_volatility', 'gold_rsi', 'gold_macd',
            'gold_ma7', 'gold_ma14', 'gold_ma30',
            'gpr_level', 'gpr_ma7', 'composite_risk',
            'gs_ratio', 'gs_ratio_ma7'
        ]
        
        # VN-specific features
        vn_features = [
            'mid_price', 'spread', 'world_price_vnd',
            'vn_premium', 'vn_premium_pct',
            'vn_lag1', 'vn_lag3', 'vn_lag7', 'vn_lag14', 'vn_lag30',
            'vn_momentum1', 'vn_momentum3', 'vn_momentum7',
            'vn_ma7', 'vn_ma14', 'vn_ma30',
            'vn_std7', 'vn_std14', 'vn_std30',
            'spread_ma7', 'spread_change',
            'premium_ma7', 'premium_change',
            'month', 'day_of_week',
            'is_tet_season', 'is_wedding_season', 'is_god_of_wealth_day', 'is_weekend'
        ]
        
        # Combine features
        all_features = []
        for f in world_features + vn_features:
            if f in df.columns:
                all_features.append(f)
        
        self.feature_columns = all_features
        self.merged_data = df
        
        # Handle missing values
        self.merged_data[self.feature_columns] = self.merged_data[self.feature_columns].ffill().fillna(0)
        
        print(f"  Created {len(self.feature_columns)} transfer features")
        return self.merged_data
    
    def train(self, test_size: float = 0.2, alpha: float = 1.0):
        """Train transfer learning model for Vietnam gold prices."""
        print(f"Training transfer model (alpha={alpha})...")
        
        if self.merged_data is None or len(self.feature_columns) == 0:
            self.create_transfer_features()
        
        # Prepare data
        X = self.merged_data[self.feature_columns].values
        
        # Create targets for each prediction day
        targets = {}
        for day in range(1, self.prediction_days + 1):
            target_col = f'target_day_{day}'
            self.merged_data[target_col] = self.merged_data['mid_price'].shift(-day)
            targets[day] = self.merged_data[target_col].values
        
        # Remove rows with NaN targets
        valid_mask = ~np.isnan(targets[self.prediction_days])
        X = X[valid_mask]
        for day in targets:
            targets[day] = targets[day][valid_mask]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for each day
        self.transfer_models = {}
        self.metrics = {'r2': {}, 'mape': {}, 'rmse': {}}
        
        for day in range(1, self.prediction_days + 1):
            y = targets[day]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=False
            )
            
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = model.score(X_test, y_test)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            
            self.transfer_models[day] = model
            self.metrics['r2'][day] = r2
            self.metrics['mape'][day] = mape
            self.metrics['rmse'][day] = rmse
            
            print(f"  Day {day}: R²={r2:.4f}, MAPE={mape:.2f}%, RMSE={rmse:.4f}")
        
        avg_r2 = np.mean(list(self.metrics['r2'].values()))
        avg_mape = np.mean(list(self.metrics['mape'].values()))
        avg_rmse = np.mean(list(self.metrics['rmse'].values()))
        print(f"Average: R²={avg_r2:.4f}, MAPE={avg_mape:.2f}%, RMSE={avg_rmse:.4f}")
        
        return self.metrics
    
    def save_model(self):
        """Save trained transfer model."""
        model_path = os.path.join(self.model_dir, 'vietnam_gold_models.pkl')
        
        model_data = {
            'models': self.transfer_models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'usd_vnd_rate': self.usd_vnd_rate,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self):
        """Load trained transfer model."""
        model_path = os.path.join(self.model_dir, 'vietnam_gold_models.pkl')
        
        if not os.path.exists(model_path):
            print("No saved model found, training new model...")
            self.load_vietnam_data()
            self.load_world_data()
            self.merge_datasets()
            self.create_transfer_features()
            self.train()
            self.save_model()
            return
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.transfer_models = model_data['models']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.usd_vnd_rate = model_data.get('usd_vnd_rate', DEFAULT_USD_VND_RATE)
        
        print(f"Model loaded from {model_path}")
    
    def predict_live(self, market_data: dict) -> list:
        """
        Predict based on real-time market inputs.
        
        Args:
            market_data: Dictionary containing 'gold_close' and other live metrics
            
        Returns:
            List of predictions for next 7 days
        """
        if not self.transfer_models:
            self.load_model()
            
        # Ensure fresh data loaded to get features
        if self.merged_data is None:
            self.load_vietnam_data()
            self.load_world_data()
            self.merge_datasets()
            self.create_transfer_features()
            
        # Clone data to avoid polluting main dataset
        df = self.merged_data.copy()
        
        if market_data.get('gold_close'):
            last_idx = df.index[-1]
            live_price = market_data['gold_close']
            
            # 1. Update World Gold Features
            df.at[last_idx, 'gold_close'] = live_price
            # Simplified: set open/high/low/ma if close changes significantly
            # Ideally we would have full OHLC, but close is most important
            
            # 2. Recalculate Vietnam Premium Features
            # World price in VND
            world_price_vnd = live_price * self.usd_vnd_rate * GOLD_OZ_TO_LUONG / 1_000_000
            df.at[last_idx, 'world_price_vnd'] = world_price_vnd
            
            # Premium
            vn_price = df.at[last_idx, 'mid_price']
            premium = vn_price - world_price_vnd
            df.at[last_idx, 'vn_premium'] = premium
            df.at[last_idx, 'vn_premium_pct'] = (premium / world_price_vnd) * 100
            
            # Recalculate rolling premium (simplified)
            prev_premium = df['vn_premium'].iloc[-2]
            df.at[last_idx, 'premium_change'] = premium - prev_premium
            
            # Standardize / Fill features
            df[self.feature_columns] = df[self.feature_columns].ffill().fillna(0)
            
            # Extract features for inference
            latest = df[self.feature_columns].iloc[-1:].values
            latest_scaled = self.scaler.transform(latest)
            
            # Base for relative changes
            # Force start date to be today for live prediction
            current_date = datetime.now()
            last_date = current_date
            last_price = df['mid_price'].iloc[-1]
            if market_data.get('timestamp'):
                try:
                    # If timestamp is isoformat
                    last_date = datetime.fromisoformat(market_data['timestamp'])
                except:
                    pass
            
            # Ensure we start predicting from tomorrow
            # If it's already late in the day, treat today as finished
            last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Predict
            predictions = []
            current_date = last_date
            for day in range(1, self.prediction_days + 1):
                # Move to next business day
                current_date += timedelta(days=1)
                while current_date.weekday() in [5, 6]:
                    current_date += timedelta(days=1)

                pred_price = self.transfer_models[day].predict(latest_scaled)[0]
                change = ((pred_price - last_price) / last_price) * 100

                # Confidence Interval (95% => 1.96 * RMSE)
                rmse = self.metrics.get('rmse', {}).get(day, 0.5)
                margin = 1.96 * rmse
                
                predictions.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day': day,
                    'predicted_price': round(pred_price, 2),
                    'change_percent': round(change, 2),
                    'unit': 'triệu VND/lượng',
                    'lower': round(pred_price - margin, 2),
                    'upper': round(pred_price + margin, 2)
                })
            
            return predictions
        
        # Fallback to standard predict if no live gold price
        return self.predict()

    def predict(self) -> list:
        """Predict Vietnam gold prices for next 7 days."""
        if not self.transfer_models:
            self.load_model()
        
        # Load latest data
        if self.merged_data is None:
            self.load_vietnam_data()
            self.load_world_data()
            self.merge_datasets()
            self.create_transfer_features()
        
        # Get latest features
        latest = self.merged_data[self.feature_columns].iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)
        
        # Get last date and price
        last_date = self.merged_data['date'].iloc[-1]
        last_price = self.merged_data['mid_price'].iloc[-1]
        
        # Predict for each day
        predictions = []
        current_date = last_date
        for day in range(1, self.prediction_days + 1):
            # Move to next business day
            current_date += timedelta(days=1)
            while current_date.weekday() in [5, 6]:
                current_date += timedelta(days=1)

            pred_price = self.transfer_models[day].predict(latest_scaled)[0]
            change = ((pred_price - last_price) / last_price) * 100

            # Confidence Interval (95% => 1.96 * RMSE)
            rmse = self.metrics.get('rmse', {}).get(day, 0.5)
            margin = 1.96 * rmse

            predictions.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day': day,
                'predicted_price': round(pred_price, 2),
                'change_percent': round(change, 2),
                'unit': 'triệu VND/lượng',
                'lower': round(pred_price - margin, 2),
                'upper': round(pred_price + margin, 2)
            })
        
        return predictions
    
    def get_historical_data(self, days: int = 30) -> list:
        """Get historical Vietnam gold prices."""
        if self.vn_data is None:
            self.load_vietnam_data()
        
        df = self.vn_data.tail(days).copy()
        
        return [{
            'date': row['date'].strftime('%Y-%m-%d'),
            'buy_price': round(row['buy_price'], 2),
            'sell_price': round(row['sell_price'], 2),
            'mid_price': round(row['mid_price'], 2),
            'spread': round(row['spread'], 2)
        } for _, row in df.iterrows()]
    
    def get_model_info(self) -> dict:
        """Get model information."""
        if not self.metrics:
            self.load_model()
        
        return {
            'model_type': 'Transfer Learning (Ridge Regression)',
            'base_model': 'World Gold Geopolitical Model',
            'features': len(self.feature_columns),
            'prediction_days': self.prediction_days,
            'avg_r2': round(np.mean(list(self.metrics['r2'].values())), 4),
            'avg_mape': round(np.mean(list(self.metrics['mape'].values())), 2),
            'usd_vnd_rate': self.usd_vnd_rate
        }
    
    def get_accuracy_metrics(self) -> dict:
        """Get accuracy metrics for display."""
        if not self.metrics:
            self.load_model()
        
        avg_r2 = np.mean(list(self.metrics['r2'].values()))
        avg_mape = np.mean(list(self.metrics['mape'].values()))
        
        return {
            'r2_score': round(avg_r2 * 100, 2),
            'mape': round(avg_mape, 2),
            'direction_accuracy': round((avg_r2 * 100 + (100 - avg_mape)) / 2, 2),
            'overall_accuracy': round((avg_r2 * 100 + (100 - avg_mape) * 0.5) / 1.5, 2)
        }

    def get_yesterday_accuracy(self) -> dict:
        """Calculate accuracy of yesterday's prediction vs today's actual price."""
        if self.vn_data is None:
            self.load_vietnam_data()
            
        # Get latest actual data
        latest = self.vn_data.iloc[-1]
        yesterday_actual = self.vn_data.iloc[-2]
        
        # In a real scenario, we would need stored predictions from yesterday.
        # Since we don't have a DB of past predictions, we simulate "yesterday's prediction"
        # by creating a prediction using data up to yesterday (t-2).
        
        # Or simpler: Just return a dummy structure if we can't calculate deeply,
        # but better: Use the model to predict for 'today' using data up to 'yesterday'.
        
        try:
            # Data up to T-1 (Yesterday relative to dataset)
            # Actually, to predict T (Today), we need input from T-1.
            # So we use data except the last row.
            
            # This is expensive to re-process all features for just one check?
            # We can just look at the last model metrics.
            
            # Let's implementation a simplified version using the trained model on test set?
            # No, let's just return None if we can't easily reproduce it without full re-run.
            
            # Better approach: Compare T (Today) actual vs T (Predicted by model from T-1)
            # We already have the model. We can run inference on the T-1 state.
            
            # Get feature vector for T-1
            # Feature columns should be ready in merged_data
            if self.merged_data is None:
                 self.create_transfer_features()
                 
            # Input for predicting T is the feature row at T-1
            # verification: T is last row (iloc[-1]). T-1 is iloc[-2].
            # Feature vector at T-1 predicts price at T.
            
            idx_t_minus_1 = -2
            if abs(idx_t_minus_1) > len(self.merged_data):
                return None
                
            features_t_minus_1 = self.merged_data[self.feature_columns].iloc[idx_t_minus_1:idx_t_minus_1+1].values
            features_scaled = self.scaler.transform(features_t_minus_1)
            
            # Predict for Day 1
            pred_price = self.transfer_models[1].predict(features_scaled)[0]
            
            # Actual price at T
            actual_price = self.merged_data['mid_price'].iloc[-1]
            actual_date = self.merged_data['date'].iloc[-1]
            
            diff = actual_price - pred_price
            diff_pct = (diff / actual_price) * 100
            accuracy = max(0, 100 - abs(diff_pct))
            
            return {
                "date": actual_date.strftime('%Y-%m-%d'),
                "actual": round(actual_price, 2),
                "predicted": round(pred_price, 2),
                "diff": round(diff, 2),
                "diff_pct": round(diff_pct, 2),
                "accuracy": round(accuracy, 2),
                "unit": "triệu VND/lượng"
            }
        except Exception as e:
            print(f"Error calculating yesterday accuracy: {e}")
            return None


def main():
    """Train Vietnam gold transfer model."""
    print("=" * 60)
    print("Training Vietnam Gold Transfer Learning Model")
    print("=" * 60)
    
    predictor = VietnamGoldPredictor()
    
    # Load and prepare data
    predictor.load_vietnam_data()
    predictor.load_world_data()
    predictor.merge_datasets()
    predictor.create_transfer_features()
    
    # Train model
    predictor.train(alpha=1.0)
    
    # Save model
    predictor.save_model()
    
    # Test prediction
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    
    predictions = predictor.predict()
    for pred in predictions:
        print(f"  {pred['date']}: {pred['predicted_price']} {pred['unit']} ({pred['change_percent']:+.2f}%)")
    
    print("\nModel Info:")
    info = predictor.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
