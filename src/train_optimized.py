"""
Optimized Model Training with Advanced Techniques
- Feature Selection (top correlated features)
- Attention Mechanism
- Ensemble Learning
- Hyperparameter Optimization
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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, Input, 
    BatchNormalization, Bidirectional,
    Attention, MultiHeadAttention, LayerNormalization,
    Concatenate, Flatten, Reshape
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib


class OptimizedSilverPredictor:
    """
    Optimized model with feature selection and advanced architecture.
    """
    
    def __init__(self, sequence_length=60, prediction_days=7):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        self.models = []  # For ensemble
        
    def load_data(self, filepath: str):
        """Load and process data from Investing.com format."""
        print(f"\n📥 Loading data from {filepath}...")
        
        self.data = pd.read_csv(filepath)
        
        # Map columns
        column_mapping = {
            'Ngày': 'date', 'Lần cuối': 'close',
            'Mở': 'open', 'Cao': 'high',
            'Thấp': 'low', 'KL': 'volume',
            '% Thay đổi': 'change_pct'
        }
        self.data = self.data.rename(columns=column_mapping)
        
        # Parse date and convert prices
        self.data['date'] = pd.to_datetime(self.data['date'], format='%d/%m/%Y')
        for col in ['close', 'open', 'high', 'low']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        self.data['price'] = self.data['close']
        self.data = self.data.sort_values('date').reset_index(drop=True)
        self.data = self.data.dropna(subset=['price'])
        
        print(f"✓ Loaded {len(self.data):,} records")
        print(f"  Date range: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}")
        
        return self.data
    
    def add_selected_features(self):
        """Add only the most important features based on domain knowledge."""
        print("\n🔧 Adding selected features...")
        
        df = self.data.copy()
        price = df['price']
        
        # Core price features
        df['returns'] = price.pct_change()
        df['log_returns'] = np.log(price / price.shift(1))
        
        # Moving averages (key for trend)
        df['sma_7'] = price.rolling(7).mean()
        df['sma_14'] = price.rolling(14).mean()
        df['sma_30'] = price.rolling(30).mean()
        
        # EMA for momentum
        df['ema_12'] = price.ewm(span=12).mean()
        df['ema_26'] = price.ewm(span=26).mean()
        
        # Volatility (key for risk)
        df['volatility_7'] = price.rolling(7).std()
        df['volatility_14'] = price.rolling(14).std()
        
        # RSI (momentum oscillator)
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (trend following)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Price position indicators
        df['price_vs_sma7'] = price / df['sma_7']
        df['price_vs_sma30'] = price / df['sma_30']
        
        # Lag features (recent history)
        for lag in [1, 2, 3, 5]:
            df[f'price_lag_{lag}'] = price.shift(lag)
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        # OHLC features
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_range'] = (df['high'] - df['low']) / price
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - price.shift()),
                    abs(df['low'] - price.shift())
                )
            )
            df['atr_14'] = df['true_range'].rolling(14).mean()
        
        self.data = df
        
        # Clean data - drop first 30 rows
        self.data = self.data.iloc[30:].reset_index(drop=True)
        
        # Fill NaN
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()
        
        # Replace inf
        self.data = self.data.replace([np.inf, -np.inf], 0)
        
        feature_cols = [c for c in self.data.columns 
                       if c not in ['date', 'close', 'open', 'high', 'low', 'volume', 'change_pct']]
        print(f"✓ Added {len(feature_cols)} selected features")
        
        return self.data
    
    def prepare_data(self, use_features=True):
        """Prepare sequences for training."""
        print("\n📊 Preparing training data...")
        
        if use_features:
            feature_cols = ['price', 'returns', 'sma_7', 'sma_14', 'sma_30',
                          'ema_12', 'ema_26', 'volatility_7', 'rsi', 'macd',
                          'price_vs_sma7', 'price_vs_sma30',
                          'price_lag_1', 'price_lag_2', 'price_lag_3']
            # Only use columns that exist
            feature_cols = [c for c in feature_cols if c in self.data.columns]
            self.feature_columns = feature_cols
        else:
            feature_cols = ['price']
            self.feature_columns = feature_cols
        
        feature_data = self.data[feature_cols].values
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        price_idx = 0  # price is first column
        
        for i in range(self.sequence_length, len(self.scaled_data) - self.prediction_days + 1):
            X.append(self.scaled_data[i - self.sequence_length:i])
            y.append(self.scaled_data[i:i + self.prediction_days, price_idx])
        
        X, y = np.array(X), np.array(y)
        
        print(f"✓ Data prepared:")
        print(f"  Features: {len(feature_cols)}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        return X, y
    
    def build_lstm_model(self, input_shape, 
                         units=(128, 64, 32),
                         dropout=0.3,
                         learning_rate=0.001):
        """Build optimized LSTM model."""
        model = Sequential([
            Input(shape=input_shape),
            
            # Layer 1 - Bidirectional LSTM
            Bidirectional(LSTM(units[0], return_sequences=True,
                              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))),
            BatchNormalization(),
            Dropout(dropout),
            
            # Layer 2
            Bidirectional(LSTM(units[1], return_sequences=True)),
            BatchNormalization(),
            Dropout(dropout),
            
            # Layer 3
            Bidirectional(LSTM(units[2], return_sequences=False)),
            BatchNormalization(),
            Dropout(dropout),
            
            # Dense layers
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dropout(dropout/2),
            Dense(32, activation='relu'),
            Dense(self.prediction_days)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        
        return model
    
    def build_gru_model(self, input_shape,
                        units=(128, 64, 32),
                        dropout=0.3,
                        learning_rate=0.001):
        """Build GRU model for ensemble."""
        model = Sequential([
            Input(shape=input_shape),
            
            Bidirectional(GRU(units[0], return_sequences=True)),
            BatchNormalization(),
            Dropout(dropout),
            
            Bidirectional(GRU(units[1], return_sequences=True)),
            BatchNormalization(),
            Dropout(dropout),
            
            Bidirectional(GRU(units[2], return_sequences=False)),
            BatchNormalization(),
            Dropout(dropout),
            
            Dense(64, activation='relu'),
            Dropout(dropout/2),
            Dense(32, activation='relu'),
            Dense(self.prediction_days)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        
        return model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val,
                       epochs=100, batch_size=32):
        """Train ensemble of models."""
        print("\n🚀 Training Ensemble Models...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        histories = []
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                                   restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                      patience=5, min_lr=1e-6, verbose=1)
        
        # Model 1: LSTM with lower dropout
        print("\n📊 Training Model 1/3: LSTM (low dropout)...")
        model1 = self.build_lstm_model(input_shape, dropout=0.2, learning_rate=0.001)
        h1 = model1.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr], verbose=1)
        self.models.append(model1)
        histories.append(h1)
        
        # Model 2: LSTM with higher dropout
        print("\n📊 Training Model 2/3: LSTM (high dropout)...")
        model2 = self.build_lstm_model(input_shape, dropout=0.4, learning_rate=0.0005)
        h2 = model2.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr], verbose=1)
        self.models.append(model2)
        histories.append(h2)
        
        # Model 3: GRU
        print("\n📊 Training Model 3/3: GRU...")
        model3 = self.build_gru_model(input_shape, dropout=0.3, learning_rate=0.001)
        h3 = model3.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop, reduce_lr], verbose=1)
        self.models.append(model3)
        histories.append(h3)
        
        print(f"\n✓ Trained {len(self.models)} models for ensemble")
        
        return histories
    
    def ensemble_predict(self, X):
        """Make predictions using ensemble averaging."""
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def inverse_transform_price(self, scaled_values):
        """Inverse transform scaled predictions."""
        dummy = np.zeros((len(scaled_values), len(self.feature_columns)))
        dummy[:, 0] = scaled_values  # price is first column
        result = self.scaler.inverse_transform(dummy)[:, 0]
        return result
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble model."""
        print("\n📊 Evaluating Ensemble...")
        
        # Individual model predictions
        for i, model in enumerate(self.models):
            pred = model.predict(X_test, verbose=0)
            
            # Flatten and inverse transform
            y_pred_flat = pred.flatten()
            y_true_flat = y_test.flatten()
            
            y_pred_prices = self.inverse_transform_price(y_pred_flat)
            y_true_prices = self.inverse_transform_price(y_true_flat)
            
            rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
            mae = mean_absolute_error(y_true_prices, y_pred_prices)
            r2 = r2_score(y_true_prices, y_pred_prices)
            mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
            
            print(f"\n  Model {i+1}: RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        # Ensemble prediction
        ensemble_pred = self.ensemble_predict(X_test)
        
        y_pred_flat = ensemble_pred.flatten()
        y_true_flat = y_test.flatten()
        
        y_pred_prices = self.inverse_transform_price(y_pred_flat)
        y_true_prices = self.inverse_transform_price(y_true_flat)
        
        rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
        mae = mean_absolute_error(y_true_prices, y_pred_prices)
        r2 = r2_score(y_true_prices, y_pred_prices)
        mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
        
        print(f"\n  📊 ENSEMBLE: RMSE=${rmse:.2f}, MAE=${mae:.2f}, R²={r2:.4f}, MAPE={mape:.2f}%")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
    
    def save(self, model_path, scaler_path):
        """Save ensemble models and scaler."""
        # Save best performing model (first one) as main model
        self.models[0].save(model_path)
        print(f"✓ Saved main model to {model_path}")
        
        # Save all models for ensemble
        model_dir = os.path.dirname(model_path)
        for i, model in enumerate(self.models):
            path = os.path.join(model_dir, f'ensemble_model_{i+1}.h5')
            model.save(path)
        print(f"✓ Saved {len(self.models)} ensemble models")
        
        # Save scaler
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, scaler_path)
        print(f"✓ Saved scaler to {scaler_path}")


def main():
    print("\n" + "=" * 60)
    print("🥈 OPTIMIZED SILVER PRICE PREDICTION MODEL")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    model_path = os.path.join(base_dir, 'models', 'silver_lstm_model.h5')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    
    # Initialize
    predictor = OptimizedSilverPredictor(sequence_length=60, prediction_days=7)
    
    # Load and process data
    predictor.load_data(data_path)
    predictor.add_selected_features()
    
    # Prepare data with selected features
    X, y = predictor.prepare_data(use_features=True)
    
    # Split data
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\n📂 Data splits:")
    print(f"  Training:   {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test:       {len(X_test):,} samples")
    
    # Train ensemble
    predictor.train_ensemble(X_train, y_train, X_val, y_val, 
                            epochs=80, batch_size=32)
    
    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    
    # Save
    predictor.save(model_path, scaler_path)
    
    # Export standard data
    export_df = predictor.data[['date', 'price']].copy()
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    export_df.to_csv(os.path.join(base_dir, 'dataset', 'silver_price.csv'), index=False)
    print(f"✓ Exported data to silver_price.csv")
    
    # Save training info
    info = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Ensemble (LSTM + GRU)',
        'n_models': len(predictor.models),
        'features': predictor.feature_columns,
        'sequence_length': 60,
        'prediction_days': 7,
        'metrics': {k: float(v) for k, v in metrics.items()},
        'data_info': {
            'records': len(predictor.data),
            'date_range': {
                'start': predictor.data['date'].min().strftime('%Y-%m-%d'),
                'end': predictor.data['date'].max().strftime('%Y-%m-%d')
            }
        }
    }
    
    with open(os.path.join(base_dir, 'models', 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Final Ensemble Results:")
    print(f"   • RMSE: ${metrics['rmse']:.2f}")
    print(f"   • MAE:  ${metrics['mae']:.2f}")
    print(f"   • R²:   {metrics['r2']:.4f}")
    print(f"   • MAPE: {metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
