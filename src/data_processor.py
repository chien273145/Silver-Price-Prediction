"""
Data Processor Module for Silver Price Prediction
Handles data loading, preprocessing, feature engineering, and sequence creation for LSTM model.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import joblib
import os


class SilverDataProcessor:
    """
    Processor for silver price data that handles:
    - Data loading and cleaning
    - Feature engineering (technical indicators)
    - Normalization
    - Sequence creation for LSTM
    """
    
    def __init__(self, sequence_length: int = 60, prediction_days: int = 7):
        """
        Initialize the data processor.
        
        Args:
            sequence_length: Number of past days to use for prediction (default: 60)
            prediction_days: Number of future days to predict (default: 7)
        """
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load silver price data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        self.data = pd.read_csv(filepath)
        
        # Standardize column names
        self.data.columns = self.data.columns.str.lower().str.strip()
        
        # Handle date column
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Handle price column naming
        if 'price' not in self.data.columns and 'close' in self.data.columns:
            self.data['price'] = self.data['close']
        
        print(f"✓ Loaded {len(self.data)} records from {filepath}")
        print(f"  Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_count = len(self.data)
        
        # Remove rows with missing price values
        self.data = self.data.dropna(subset=['price'])
        
        # Forward fill any remaining NaN values
        self.data = self.data.ffill()
        
        # Remove duplicate dates
        self.data = self.data.drop_duplicates(subset=['date'], keep='last')
        
        # Remove extreme outliers (prices that are 0 or negative)
        self.data = self.data[self.data['price'] > 0]
        
        removed_count = initial_count - len(self.data)
        print(f"✓ Cleaned data: removed {removed_count} invalid records")
        print(f"  Remaining records: {len(self.data)}")
        
        return self.data
    
    def add_features(self) -> pd.DataFrame:
        """
        Add technical indicators and time-based features.
        
        Returns:
            DataFrame with additional features
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data.copy()
        
        # Price-based features
        df['price_change'] = df['price'].pct_change()
        df['price_change_abs'] = df['price'].diff()
        
        # Moving Averages
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_14'] = df['price'].rolling(window=14).mean()
        df['ma_30'] = df['price'].rolling(window=30).mean()
        df['ma_60'] = df['price'].rolling(window=60).mean()
        
        # Exponential Moving Averages
        df['ema_7'] = df['price'].ewm(span=7, adjust=False).mean()
        df['ema_14'] = df['price'].ewm(span=14, adjust=False).mean()
        df['ema_30'] = df['price'].ewm(span=30, adjust=False).mean()
        
        # Volatility
        df['volatility_7'] = df['price'].rolling(window=7).std()
        df['volatility_14'] = df['price'].rolling(window=14).std()
        df['volatility_30'] = df['price'].rolling(window=30).std()
        
        # Price momentum
        df['momentum_7'] = df['price'] - df['price'].shift(7)
        df['momentum_14'] = df['price'] - df['price'].shift(14)
        df['momentum_30'] = df['price'] - df['price'].shift(30)
        
        # Rate of Change (ROC)
        df['roc_7'] = ((df['price'] - df['price'].shift(7)) / df['price'].shift(7)) * 100
        df['roc_14'] = ((df['price'] - df['price'].shift(14)) / df['price'].shift(14)) * 100
        
        # Relative Strength Index (RSI)
        df['rsi_14'] = self._calculate_rsi(df['price'], 14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['price'])
        
        # Bollinger Bands
        df['bb_middle'] = df['price'].rolling(window=20).mean()
        df['bb_std'] = df['price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5, 7, 14]:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
        
        # Drop rows with NaN values created by feature engineering
        df = df.dropna()
        
        self.data = df
        print(f"✓ Added {len([c for c in df.columns if c not in ['date', 'price']])} features")
        print(f"  Records after feature engineering: {len(self.data)}")
        
        return self.data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, 
                        fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def prepare_data(self, use_features: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model by scaling and creating sequences.
        
        Args:
            use_features: Whether to use additional features or just price
            
        Returns:
            Tuple of (X, y) arrays ready for training
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Select columns
        if use_features:
            # Select numeric columns (excluding date)
            feature_cols = [col for col in self.data.columns 
                          if col not in ['date'] and self.data[col].dtype in ['float64', 'int64', 'int32']]
            data_values = self.data[feature_cols].values
        else:
            data_values = self.data['price'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(data_values)
        
        # Create sequences
        X, y = self._create_sequences(self.scaled_data, self.data['price'].values)
        
        print(f"✓ Prepared data for training")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  y shape: {y.shape} (samples, prediction_days)")
        
        return X, y
    
    def prepare_simple_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare simple data (price only) for LSTM model.
        
        Returns:
            Tuple of (X, y) arrays ready for training
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Use only price
        price_data = self.data['price'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(price_data)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(self.scaled_data) - self.prediction_days + 1):
            X.append(self.scaled_data[i - self.sequence_length:i, 0])
            y.append(self.scaled_data[i:i + self.prediction_days, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        print(f"✓ Prepared simple data for training")
        print(f"  X shape: {X.shape} (samples, timesteps, features)")
        print(f"  y shape: {y.shape} (samples, prediction_days)")
        
        return X, y
    
    def _create_sequences(self, scaled_data: np.ndarray, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model.
        
        Args:
            scaled_data: Normalized data
            prices: Original price data for target
            
        Returns:
            X and y arrays
        """
        X, y = [], []
        
        # Scale prices for y
        price_scaled = self.scaler.transform(prices.reshape(-1, 1) if prices.ndim == 1 
                                             else prices[:, 0:1])
        
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_days + 1):
            X.append(scaled_data[i - self.sequence_length:i])
            y.append(price_scaled[i:i + self.prediction_days, 0])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.8, val_ratio: float = 0.1) -> dict:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input features
            y: Target values
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            Dictionary with train, val, test splits
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
        
        print(f"✓ Split data:")
        print(f"  Training:   {len(splits['X_train'])} samples ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(splits['X_val'])} samples ({val_ratio*100:.0f}%)")
        print(f"  Test:       {len(splits['X_test'])} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return splits
    
    def inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled values back to original price scale.
        
        Args:
            scaled_values: Normalized values
            
        Returns:
            Original scale values
        """
        # Handle different input shapes
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        
        # If multi-dimensional, we need to handle it properly
        if scaled_values.shape[1] > 1:
            # Assume first column is price
            result = []
            for i in range(scaled_values.shape[1]):
                col = scaled_values[:, i].reshape(-1, 1)
                # Pad to match scaler's expected input
                padded = np.zeros((len(col), self.scaler.n_features_in_))
                padded[:, 0] = col[:, 0]
                inv = self.scaler.inverse_transform(padded)[:, 0]
                result.append(inv)
            return np.array(result).T
        else:
            # Single column - pad to match scaler
            padded = np.zeros((len(scaled_values), self.scaler.n_features_in_))
            padded[:, 0] = scaled_values[:, 0]
            return self.scaler.inverse_transform(padded)[:, 0]
    
    def save_scaler(self, filepath: str):
        """Save the scaler to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)
        print(f"✓ Saved scaler to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load the scaler from a file."""
        self.scaler = joblib.load(filepath)
        print(f"✓ Loaded scaler from {filepath}")
    
    def get_latest_sequence(self) -> np.ndarray:
        """
        Get the latest sequence for making predictions.
        
        Returns:
            Latest sequence ready for prediction
        """
        if self.scaled_data is None:
            raise ValueError("No scaled data available. Call prepare_data() first.")
        
        latest_seq = self.scaled_data[-self.sequence_length:]
        return latest_seq.reshape(1, self.sequence_length, -1)
    
    def get_dates(self) -> pd.Series:
        """Get the date column from the data."""
        if self.data is None:
            raise ValueError("No data loaded.")
        return self.data['date']
    
    def get_prices(self) -> pd.Series:
        """Get the price column from the data."""
        if self.data is None:
            raise ValueError("No data loaded.")
        return self.data['price']


def main():
    """Test the data processor."""
    # Initialize processor
    processor = SilverDataProcessor(sequence_length=60, prediction_days=7)
    
    # Load and process data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'silver_price.csv')
    processor.load_data(data_path)
    processor.clean_data()
    
    # Prepare simple data (price only)
    X, y = processor.prepare_simple_data()
    
    # Split data
    splits = processor.split_data(X, y)
    
    # Save scaler
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
    processor.save_scaler(scaler_path)
    
    print("\n✓ Data processing complete!")
    

if __name__ == "__main__":
    main()
