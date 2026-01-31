"""
Ridge Regression Model for Silver Price Prediction
A simpler, faster alternative to LSTM that can work well with proper feature engineering.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class RidgeSilverPredictor:
    """
    Ridge Regression model for multi-step silver price prediction.
    Uses feature engineering and lagged variables.
    """
    
    def __init__(self, sequence_length=60, prediction_days=7):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.models = []  # One Ridge model per prediction day
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.data = None
        self.feature_columns = None
        
    def load_data(self, filepath: str):
        """Load data from Investing.com format."""
        print(f"\n📥 Loading data from {filepath}...")
        
        self.data = pd.read_csv(filepath)
        
        column_mapping = {
            'Ngày': 'date', 'Lần cuối': 'close',
            'Mở': 'open', 'Cao': 'high',
            'Thấp': 'low', 'KL': 'volume',
            '% Thay đổi': 'change_pct'
        }
        self.data = self.data.rename(columns=column_mapping)
        
        self.data['date'] = pd.to_datetime(self.data['date'], format='%d/%m/%Y')
        for col in ['close', 'open', 'high', 'low']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data['price'] = self.data['close']
        self.data = self.data.sort_values('date').reset_index(drop=True)
        self.data = self.data.dropna(subset=['price'])
        
        print(f"✓ Loaded {len(self.data):,} records")
        print(f"  Date range: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Latest price: ${self.data['price'].iloc[-1]:.2f}")
        
        return self.data
    
    def create_features(self):
        """Create comprehensive features for Ridge Regression."""
        print("\n🔧 Creating features...")
        
        df = self.data.copy()
        price = df['price']
        
        # 1. Lag features (most important for Ridge)
        for lag in range(1, self.sequence_length + 1):
            df[f'price_lag_{lag}'] = price.shift(lag)
        
        # 2. Moving averages
        for window in [5, 7, 10, 14, 20, 21, 30, 50]:
            df[f'sma_{window}'] = price.rolling(window).mean()
            df[f'sma_{window}_lag1'] = df[f'sma_{window}'].shift(1)
        
        # 3. Exponential moving averages
        for span in [5, 10, 20]:
            df[f'ema_{span}'] = price.ewm(span=span).mean()
        
        # 4. Returns and momentum
        df['return_1d'] = price.pct_change()
        df['return_5d'] = price.pct_change(5)
        df['return_10d'] = price.pct_change(10)
        df['return_20d'] = price.pct_change(20)
        
        # Lagged returns
        for lag in range(1, 6):
            df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
        
        # 5. Volatility
        df['volatility_5'] = price.rolling(5).std()
        df['volatility_10'] = price.rolling(10).std()
        df['volatility_20'] = price.rolling(20).std()
        
        # 6. RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. MACD
        ema12 = price.ewm(span=12).mean()
        ema26 = price.ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 8. Price relative to MAs
        df['price_vs_sma5'] = price / df['sma_5']
        df['price_vs_sma10'] = price / df['sma_10']
        df['price_vs_sma20'] = df['sma_10'] / df['sma_20']
        
        # 9. Bollinger Bands
        df['bb_middle'] = df['sma_20']
        df['bb_std'] = price.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # 10. Rate of change
        df['roc_5'] = (price - price.shift(5)) / price.shift(5)
        df['roc_10'] = (price - price.shift(10)) / price.shift(10)
        
        # 11. High-Low features (if available)
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_range'] = (df['high'] - df['low']) / price
            df['atr'] = df['hl_range'].rolling(14).mean()
        
        # 12. Trend features
        df['trend_5'] = (price - price.shift(5)) / 5
        df['trend_10'] = (price - price.shift(10)) / 10
        df['trend_20'] = (price - price.shift(20)) / 20
        
        self.data = df
        
        # Clean data
        self.data = self.data.iloc[self.sequence_length:].reset_index(drop=True)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()
        self.data = self.data.replace([np.inf, -np.inf], 0)
        
        # Select feature columns
        exclude = ['date', 'close', 'open', 'high', 'low', 'volume', 'change_pct', 'price']
        self.feature_columns = [c for c in self.data.columns if c not in exclude]
        
        print(f"✓ Created {len(self.feature_columns)} features")
        print(f"✓ Data shape: {len(self.data):,} rows")
        
        return self.data
    
    def prepare_data(self):
        """Prepare X and y for training."""
        print("\n📊 Preparing training data...")
        
        X = self.data[self.feature_columns].values
        
        # Create multiple targets (one for each prediction day)
        y_list = []
        for day in range(1, self.prediction_days + 1):
            y_list.append(self.data['price'].shift(-day).values)
        
        # Stack targets
        Y = np.column_stack(y_list)
        
        # Remove rows with NaN targets
        valid_mask = ~np.isnan(Y).any(axis=1)
        X = X[valid_mask]
        Y = Y[valid_mask]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Scale targets (each column separately or together)
        Y_scaled = self.target_scaler.fit_transform(Y)
        
        print(f"✓ X shape: {X_scaled.shape}")
        print(f"✓ Y shape: {Y_scaled.shape}")
        
        return X_scaled, Y_scaled
    
    def train(self, X, Y, test_size=0.1):
        """Train Ridge models."""
        print("\n🚀 Training Ridge Regression models...")
        
        # Split data
        n = len(X)
        train_end = int(n * (1 - test_size))
        
        X_train, X_test = X[:train_end], X[train_end:]
        Y_train, Y_test = Y[:train_end], Y[train_end:]
        
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        
        # Train one Ridge model for each prediction day
        self.models = []
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for day in range(self.prediction_days):
            print(f"  Training model for Day {day + 1}...", end=" ")
            
            # Use RidgeCV to find best alpha
            model = RidgeCV(alphas=alphas, cv=5)
            model.fit(X_train, Y_train[:, day])
            
            self.models.append(model)
            print(f"alpha={model.alpha_:.2f}")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        
        Y_pred = np.zeros_like(Y_test)
        for day, model in enumerate(self.models):
            Y_pred[:, day] = model.predict(X_test)
        
        # Inverse transform
        Y_pred_prices = self.target_scaler.inverse_transform(Y_pred)
        Y_true_prices = self.target_scaler.inverse_transform(Y_test)
        
        # Flatten for overall metrics
        y_pred_flat = Y_pred_prices.flatten()
        y_true_flat = Y_true_prices.flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / y_true_flat)) * 100
        
        self.metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
        
        print(f"\n📊 Test Results:")
        print(f"   • RMSE: ${rmse:.2f}")
        print(f"   • MAE:  ${mae:.2f}")
        print(f"   • R²:   {r2:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        
        # Per-day metrics
        print("\n📅 Per-Day Metrics:")
        for day in range(self.prediction_days):
            day_rmse = np.sqrt(mean_squared_error(Y_true_prices[:, day], Y_pred_prices[:, day]))
            day_r2 = r2_score(Y_true_prices[:, day], Y_pred_prices[:, day])
            print(f"   Day {day + 1}: RMSE=${day_rmse:.2f}, R²={day_r2:.4f}")
        
        return self.metrics
    
    def save(self, model_dir):
        """Save models and scalers."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        model_path = os.path.join(model_dir, 'ridge_models.pkl')
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns
        }, model_path)
        print(f"✓ Saved models to {model_path}")
        
        # Save info
        info = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Ridge Regression (Multi-output)',
            'n_features': len(self.feature_columns),
            'prediction_days': self.prediction_days,
            'metrics': {k: float(v) for k, v in self.metrics.items()},
            'alphas': [float(m.alpha_) for m in self.models],
            'data_info': {
                'records': len(self.data),
                'date_range': {
                    'start': self.data['date'].min().strftime('%Y-%m-%d'),
                    'end': self.data['date'].max().strftime('%Y-%m-%d')
                },
                'latest_price': float(self.data['price'].iloc[-1])
            }
        }
        
        info_path = os.path.join(model_dir, 'ridge_training_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Saved info to {info_path}")
        
        return model_path
    
    def predict_next_days(self):
        """Predict next 7 days from latest data."""
        # Get latest features
        latest_features = self.data[self.feature_columns].iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # Predict
        predictions_scaled = []
        for model in self.models:
            pred = model.predict(latest_features_scaled)
            predictions_scaled.append(pred[0])
        
        predictions_scaled = np.array(predictions_scaled).reshape(1, -1)
        predictions = self.target_scaler.inverse_transform(predictions_scaled)[0]
        
        return predictions


def main():
    print("\n" + "=" * 60)
    print("🥈 RIDGE REGRESSION SILVER PRICE MODEL")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    model_dir = os.path.join(base_dir, 'models')
    
    predictor = RidgeSilverPredictor(sequence_length=60, prediction_days=7)
    
    predictor.load_data(data_path)
    predictor.create_features()
    X, Y = predictor.prepare_data()
    
    metrics = predictor.train(X, Y, test_size=0.1)
    predictor.save(model_dir)
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    predictions = predictor.predict_next_days()
    latest_price = predictor.data['price'].iloc[-1]
    latest_date = predictor.data['date'].iloc[-1]
    
    print(f"\nLatest price ({latest_date.strftime('%Y-%m-%d')}): ${latest_price:.2f}")
    print("\nPredictions for next 7 trading days:")
    for i, pred in enumerate(predictions):
        change = pred - latest_price
        change_pct = (change / latest_price) * 100
        print(f"  Day {i+1}: ${pred:.2f} ({change_pct:+.2f}%)")
    
    print("\n" + "=" * 60)
    print("✅ RIDGE REGRESSION TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   • RMSE: ${metrics['rmse']:.2f}")
    print(f"   • MAE:  ${metrics['mae']:.2f}")
    print(f"   • R²:   {metrics['r2']:.4f}")
    print(f"   • MAPE: {metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
