"""
Advanced Data Processor for Silver Price Prediction
With Feature Engineering for improved model accuracy
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedSilverDataProcessor:
    """
    Advanced data processor with comprehensive feature engineering.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_days: int = 7):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.data = None
        self.scaled_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        self.feature_columns = None
        
    def load_investing_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from Investing.com format.
        Handles Vietnamese column names and date format.
        """
        print(f"\n📥 Loading data from {filepath}...")
        
        self.data = pd.read_csv(filepath)
        
        # Map Vietnamese columns to English
        column_mapping = {
            'Ngày': 'date',
            'Lần cuối': 'close',
            'Mở': 'open',
            'Cao': 'high',
            'Thấp': 'low',
            'KL': 'volume',
            '% Thay đổi': 'change_pct'
        }
        
        self.data = self.data.rename(columns=column_mapping)
        
        # Parse date (format: DD/MM/YYYY)
        self.data['date'] = pd.to_datetime(self.data['date'], format='%d/%m/%Y')
        
        # Convert price columns to float (remove commas if any)
        for col in ['close', 'open', 'high', 'low']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Add price column (alias for close)
        self.data['price'] = self.data['close']
        
        # Sort by date ascending
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Remove rows with missing prices
        original_len = len(self.data)
        self.data = self.data.dropna(subset=['price'])
        removed = original_len - len(self.data)
        
        print(f"✓ Loaded {len(self.data):,} records")
        print(f"  Date range: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}")
        print(f"  Price range: ${self.data['price'].min():.2f} - ${self.data['price'].max():.2f}")
        if removed > 0:
            print(f"  Removed {removed} rows with missing data")
        
        return self.data
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add comprehensive technical indicators for better predictions.
        """
        print("\n🔧 Adding technical indicators...")
        
        df = self.data.copy()
        price = df['price']
        
        # 1. Moving Averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = price.rolling(window=window).mean()
            df[f'ema_{window}'] = price.ewm(span=window, adjust=False).mean()
        
        # 2. Price relative to moving averages
        df['price_sma20_ratio'] = price / df['sma_20']
        df['price_sma50_ratio'] = price / df['sma_50']
        
        # 3. Volatility indicators
        df['volatility_20'] = price.rolling(window=20).std()
        df['volatility_10'] = price.rolling(window=10).std()
        
        # 4. Bollinger Bands
        df['bb_middle'] = df['sma_20']
        df['bb_upper'] = df['sma_20'] + 2 * df['volatility_20']
        df['bb_lower'] = df['sma_20'] - 2 * df['volatility_20']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 5. RSI (Relative Strength Index)
        delta = price.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 6. MACD
        exp1 = price.ewm(span=12, adjust=False).mean()
        exp2 = price.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 7. Momentum indicators
        df['momentum_5'] = price.pct_change(5)
        df['momentum_10'] = price.pct_change(10)
        df['momentum_20'] = price.pct_change(20)
        
        # 8. Rate of Change (ROC)
        df['roc_5'] = ((price - price.shift(5)) / price.shift(5)) * 100
        df['roc_10'] = ((price - price.shift(10)) / price.shift(10)) * 100
        
        # 9. Average True Range (ATR)
        if 'high' in df.columns and 'low' in df.columns:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - price.shift()).abs()
            low_close = (df['low'] - price.shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(window=14).mean()
        
        # 10. Price change features
        df['daily_return'] = price.pct_change()
        df['daily_return_5d_avg'] = df['daily_return'].rolling(5).mean()
        
        # 11. High-Low spread
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_spread'] = (df['high'] - df['low']) / price
        
        # 12. Lag features
        for lag in [1, 2, 3, 5, 7]:
            df[f'price_lag_{lag}'] = price.shift(lag)
            df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
        # 13. Calendar features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        self.data = df
        
        # Count features added
        indicator_cols = [c for c in df.columns if c not in ['date', 'price', 'close', 'open', 'high', 'low', 'volume', 'change_pct']]
        print(f"✓ Added {len(indicator_cols)} technical indicators")
        
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean data and handle missing values.
        """
        print("\n🧹 Cleaning data...")
        
        original_len = len(self.data)
        
        # Drop rows with NaN in critical columns
        self.data = self.data.dropna(subset=['price'])
        
        # Forward fill for feature columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].ffill()
        
        # Backward fill any remaining NaN at the start
        self.data[numeric_cols] = self.data[numeric_cols].bfill()
        
        # Replace any remaining inf values
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()
        
        # Drop only the first N rows where technical indicators are NaN
        # (typically 50 rows for sma_50 which needs 50 days of data)
        first_valid_idx = 50  # Skip first 50 rows
        self.data = self.data.iloc[first_valid_idx:].reset_index(drop=True)
        
        removed = original_len - len(self.data)
        print(f"✓ Cleaned data: {len(self.data):,} rows (removed {removed} rows)")
        
        return self.data
    
    def prepare_features(self, use_all_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training.
        
        Args:
            use_all_features: If True, use all technical indicators. If False, use only price.
        """
        print("\n📊 Preparing features for training...")
        
        if use_all_features:
            # Select numeric columns excluding date and target
            exclude_cols = ['date', 'change_pct']
            self.feature_columns = [col for col in self.data.columns 
                                   if col not in exclude_cols 
                                   and self.data[col].dtype in ['float64', 'int64', 'int32', 'float32']]
        else:
            self.feature_columns = ['price']
        
        print(f"  Using {len(self.feature_columns)} features")
        
        # Get feature data
        feature_data = self.data[self.feature_columns].values
        
        # Scale features
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = self._create_sequences()
        
        print(f"✓ Prepared data:")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  y shape: {y.shape} (samples, prediction_days)")
        
        return X, y
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model.
        """
        X, y = [], []
        
        # Get price column index
        price_idx = self.feature_columns.index('price') if 'price' in self.feature_columns else 0
        
        for i in range(self.sequence_length, len(self.scaled_data) - self.prediction_days + 1):
            X.append(self.scaled_data[i - self.sequence_length:i])
            y.append(self.scaled_data[i:i + self.prediction_days, price_idx])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict:
        """
        Split data into train, validation, and test sets.
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }
        
        print(f"\n📂 Data splits:")
        print(f"  Training:   {len(splits['X_train']):,} samples ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(splits['X_val']):,} samples ({val_ratio*100:.0f}%)")
        print(f"  Test:       {len(splits['X_test']):,} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return splits
    
    def get_latest_sequence(self) -> np.ndarray:
        """
        Get the latest sequence for making predictions.
        """
        if self.scaled_data is None:
            raise ValueError("No scaled data available. Call prepare_features() first.")
        
        latest_seq = self.scaled_data[-self.sequence_length:]
        return latest_seq.reshape(1, self.sequence_length, -1)
    
    def inverse_transform_price(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions to actual prices.
        """
        # Get price column index
        price_idx = self.feature_columns.index('price') if 'price' in self.feature_columns else 0
        
        # Create dummy array with same shape as scaler expects
        dummy = np.zeros((len(scaled_values), len(self.feature_columns)))
        dummy[:, price_idx] = scaled_values
        
        # Inverse transform
        result = self.scaler.inverse_transform(dummy)[:, price_idx]
        
        return result
    
    def get_dates(self) -> pd.Series:
        """Get the date column."""
        return self.data['date']
    
    def get_prices(self) -> pd.Series:
        """Get the price column."""
        return self.data['price']
    
    def save_scaler(self, filepath: str):
        """Save the scaler to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"✓ Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load the scaler from a file."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"✓ Loaded scaler from {filepath}")
    
    def export_to_standard_format(self, output_path: str):
        """
        Export processed data to standard CSV format for the model.
        """
        export_df = self.data[['date', 'price']].copy()
        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
        export_df.to_csv(output_path, index=False)
        print(f"✓ Exported data to {output_path}")


def process_investing_data(input_path: str, output_path: str):
    """
    Process Investing.com data and save to standard format.
    """
    processor = AdvancedSilverDataProcessor()
    processor.load_investing_data(input_path)
    processor.add_technical_indicators()
    processor.clean_data()
    processor.export_to_standard_format(output_path)
    
    return processor


if __name__ == "__main__":
    # Test the processor
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'dataset', 'dataset_silver.csv')
    output_path = os.path.join(base_dir, 'dataset', 'silver_price.csv')
    
    processor = process_investing_data(input_path, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Data processing complete!")
    print("=" * 60)
