"""
Final Optimized Training Script
Using best configuration found: LSTM with low dropout, selected features
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, 
    BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib


class FinalOptimizedPredictor:
    """Final optimized model with best hyperparameters."""
    
    def __init__(self, sequence_length=60, prediction_days=7):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
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
    
    def add_features(self):
        """Add carefully selected features."""
        print("\n🔧 Adding optimized features...")
        
        df = self.data.copy()
        price = df['price']
        
        # Key features only
        df['returns'] = price.pct_change()
        df['sma_7'] = price.rolling(7).mean()
        df['sma_14'] = price.rolling(14).mean()
        df['sma_21'] = price.rolling(21).mean()
        df['ema_12'] = price.ewm(span=12).mean()
        df['ema_26'] = price.ewm(span=26).mean()
        df['volatility_14'] = price.rolling(14).std()
        
        # RSI
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Relative position
        df['price_vs_sma14'] = price / df['sma_14']
        
        # Lag features
        for lag in [1, 2, 3]:
            df[f'price_lag_{lag}'] = price.shift(lag)
        
        self.data = df
        
        # Clean
        self.data = self.data.iloc[30:].reset_index(drop=True)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()
        self.data = self.data.replace([np.inf, -np.inf], 0)
        
        print(f"✓ Added features, {len(self.data):,} rows remaining")
        
        return self.data
    
    def prepare_data(self):
        """Prepare training sequences."""
        print("\n📊 Preparing data...")
        
        # Features that work well
        feature_cols = ['price', 'sma_7', 'sma_14', 'sma_21', 
                       'ema_12', 'ema_26', 'volatility_14', 'rsi',
                       'price_vs_sma14', 'price_lag_1', 'price_lag_2', 'price_lag_3']
        feature_cols = [c for c in feature_cols if c in self.data.columns]
        self.feature_columns = feature_cols
        
        feature_data = self.data[feature_cols].values
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(self.scaled_data) - self.prediction_days + 1):
            X.append(self.scaled_data[i - self.sequence_length:i])
            y.append(self.scaled_data[i:i + self.prediction_days, 0])  # price is first
        
        X, y = np.array(X), np.array(y)
        
        print(f"✓ Features: {feature_cols}")
        print(f"✓ X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def build_model(self, input_shape):
        """Build optimized LSTM model."""
        model = Sequential([
            Input(shape=input_shape),
            
            # Layer 1
            Bidirectional(LSTM(128, return_sequences=True,
                              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
            BatchNormalization(),
            Dropout(0.2),
            
            # Layer 2
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Layer 3
            Bidirectional(LSTM(32, return_sequences=False)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(self.prediction_days)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='huber', metrics=['mae'])
        
        return model
    
    def train(self, X, y, epochs=150, batch_size=32, model_path=None):
        """Train the model."""
        print(f"\n🚀 Training model...")
        
        # Split
        n = len(X)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"  Training:   {len(X_train):,} samples")
        print(f"  Validation: {len(X_val):,} samples")
        print(f"  Test:       {len(X_test):,} samples")
        
        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self.build_model(input_shape)
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                             patience=7, min_lr=1e-6, verbose=1)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss',
                               save_best_only=True, verbose=1)
            )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n📊 Evaluating on test set...")
        y_pred = self.model.predict(X_test, verbose=0)
        
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_test.flatten()
        
        y_pred_prices = self.inverse_transform_price(y_pred_flat)
        y_true_prices = self.inverse_transform_price(y_true_flat)
        
        rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
        mae = mean_absolute_error(y_true_prices, y_pred_prices)
        r2 = r2_score(y_true_prices, y_pred_prices)
        mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
        
        self.metrics = {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
        
        print(f"\n📊 Test Results:")
        print(f"   • RMSE: ${rmse:.2f}")
        print(f"   • MAE:  ${mae:.2f}")
        print(f"   • R²:   {r2:.4f}")
        print(f"   • MAPE: {mape:.2f}%")
        
        return history
    
    def inverse_transform_price(self, scaled_values):
        """Inverse transform predictions."""
        dummy = np.zeros((len(scaled_values), len(self.feature_columns)))
        dummy[:, 0] = scaled_values
        result = self.scaler.inverse_transform(dummy)[:, 0]
        return result
    
    def save(self, model_path, scaler_path, info_path):
        """Save model and scaler."""
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'Bidirectional LSTM (Optimized)',
            'features': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'metrics': {k: float(v) for k, v in self.metrics.items()},
            'data_info': {
                'records': len(self.data),
                'date_range': {
                    'start': self.data['date'].min().strftime('%Y-%m-%d'),
                    'end': self.data['date'].max().strftime('%Y-%m-%d')
                },
                'latest_price': float(self.data['price'].iloc[-1])
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"✓ Info saved to {info_path}")


def main():
    print("\n" + "=" * 60)
    print("🥈 FINAL OPTIMIZED SILVER PRICE MODEL")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    model_path = os.path.join(base_dir, 'models', 'silver_lstm_model.h5')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    info_path = os.path.join(base_dir, 'models', 'training_info.json')
    
    predictor = FinalOptimizedPredictor(sequence_length=60, prediction_days=7)
    
    predictor.load_data(data_path)
    predictor.add_features()
    X, y = predictor.prepare_data()
    
    predictor.train(X, y, epochs=150, batch_size=32, model_path=model_path)
    predictor.save(model_path, scaler_path, info_path)
    
    # Export standard data
    export_df = predictor.data[['date', 'price']].copy()
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    export_df.to_csv(os.path.join(base_dir, 'dataset', 'silver_price.csv'), index=False)
    print(f"✓ Exported data to silver_price.csv")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Results:")
    print(f"   • RMSE: ${predictor.metrics['rmse']:.2f}")
    print(f"   • MAE:  ${predictor.metrics['mae']:.2f}")
    print(f"   • R²:   {predictor.metrics['r2']:.4f}")
    print(f"   • MAPE: {predictor.metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
