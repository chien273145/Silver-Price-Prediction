"""
Unified Predictor - Supports both LSTM and Ridge Regression models
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_USD_VND_RATE = 25000


class UnifiedPredictor:
    """
    Unified predictor that supports multiple model types.
    Currently supports: Ridge Regression (primary) and LSTM (fallback).
    """
    
    def __init__(self,
                 model_dir: str = None,
                 data_path: str = None,
                 sequence_length: int = 60,
                 prediction_days: int = 7,
                 model_type: str = 'ridge'):  # 'ridge' or 'lstm'
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = model_dir or os.path.join(self.base_dir, 'models')
        self.data_path = data_path or os.path.join(self.base_dir, 'dataset', 'dataset_silver.csv')
        
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.model_type = model_type
        
        self.data = None
        self.models = None
        self.scaler = None
        self.target_scaler = None
        self.feature_columns = None
        
        self.usd_vnd_rate = DEFAULT_USD_VND_RATE
        self.troy_ounce_to_luong = 1.20565
        
        # Vietnam market premium (~23-25%)
        # Giá bạc VN thường cao hơn giá thế giới do:
        # - Chi phí nhập khẩu, vận chuyển
        # - Thuế và phí
        # - Biên lợi nhuận của cửa hàng
        # Tính toán: Giá VN ~3,273,000 / Giá quốc tế ~2,647,000 = 1.236
        self.vietnam_premium = 1.24  # 24% premium
        
    def load(self):
        """Load model and data."""
        print(f"🔄 Loading {self.model_type.upper()} model...")
        
        if self.model_type == 'ridge':
            self._load_ridge()
        else:
            self._load_lstm()
        
        self._load_data()
        print("✓ Model and data loaded successfully")
        
    def _load_ridge(self):
        """Load Ridge Regression models."""
        model_path = os.path.join(self.model_dir, 'ridge_models.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Ridge model not found at {model_path}")
        
        data = joblib.load(model_path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.target_scaler = data['target_scaler']
        self.feature_columns = data['feature_columns']
        
        print(f"✓ Loaded Ridge models ({len(self.models)} models, {len(self.feature_columns)} features)")
        
    def _load_lstm(self):
        """Load LSTM model."""
        from tensorflow import keras
        
        model_path = os.path.join(self.model_dir, 'silver_lstm_model.h5')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        self.keras_model = keras.models.load_model(model_path, compile=False)
        
        scaler_data = joblib.load(scaler_path)
        if isinstance(scaler_data, dict):
            self.scaler = scaler_data['scaler']
            self.feature_columns = scaler_data.get('feature_columns', ['price'])
        else:
            self.scaler = scaler_data
            self.feature_columns = ['price']
        
        print(f"✓ Loaded LSTM model")
        
    def _load_data(self):
        """Load and prepare data."""
        self.data = pd.read_csv(self.data_path)
        
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
        
        if self.model_type == 'ridge':
            self._create_ridge_features()
        
        print(f"✓ Loaded {len(self.data):,} records")
        
    def _create_ridge_features(self):
        """Create features for Ridge model."""
        df = self.data.copy()
        price = df['price']
        
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
        
        # Price vs MAs
        df['price_vs_sma5'] = price / df['sma_5']
        df['price_vs_sma10'] = price / df['sma_10']
        df['price_vs_sma20'] = df['sma_10'] / df['sma_20']
        
        # Bollinger
        df['bb_middle'] = df['sma_20']
        df['bb_std'] = price.rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ROC
        df['roc_5'] = (price - price.shift(5)) / price.shift(5)
        df['roc_10'] = (price - price.shift(10)) / price.shift(10)
        
        # HL features
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_range'] = (df['high'] - df['low']) / price
            df['atr'] = df['hl_range'].rolling(14).mean()
        
        # Trends
        df['trend_5'] = (price - price.shift(5)) / 5
        df['trend_10'] = (price - price.shift(10)) / 10
        df['trend_20'] = (price - price.shift(20)) / 20
        
        self.data = df
        self.data = self.data.iloc[self.sequence_length:].reset_index(drop=True)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()
        self.data = self.data.replace([np.inf, -np.inf], 0)
        
    def set_exchange_rate(self, rate: float):
        """Set USD to VND exchange rate."""
        self.usd_vnd_rate = rate
        print(f"✓ Exchange rate set to: {rate:,.0f} VND/USD")
        
    def predict(self, in_vnd: bool = True) -> Dict:
        """Make predictions for the next 7 days."""
        if self.models is None and not hasattr(self, 'keras_model'):
            self.load()
        
        # Try to get real-time data to update the latest price
        try:
            from backend.realtime_data import get_current_silver_price
            realtime_data = get_current_silver_price()
            
            if realtime_data.get('price'):
                current_price = realtime_data['price']
                current_date = datetime.now()
                print(f"✓ Using real-time price: ${current_price} ({realtime_data.get('symbol')})")
                
                # Update the last row of data or append new row
                last_date = self.data['date'].iloc[-1]
                
                # If same day, update price. If new day, append.
                if current_date.date() > last_date.date():
                    # Create new row
                    new_row = self.data.iloc[-1:].copy()
                    new_row['date'] = current_date
                    new_row['price'] = current_price
                    new_row['close'] = current_price
                    
                    # Recalculate features for the new row (simplified)
                    self.data = pd.concat([self.data, new_row], ignore_index=True)
                    if self.model_type == 'ridge':
                        self._create_ridge_features()
                else:
                    # Update today's price
                    self.data.iloc[-1, self.data.columns.get_loc('price')] = current_price
                    self.data.iloc[-1, self.data.columns.get_loc('close')] = current_price
                    if self.model_type == 'ridge':
                        self._create_ridge_features()
                        
        except Exception as e:
            print(f"⚠️ Could not fetch real-time data: {e}")
            print("   Using latest historical data.")

        if self.model_type == 'ridge':
            predictions_usd = self._predict_ridge()
        else:
            predictions_usd = self._predict_lstm()
        
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
            'model_type': self.model_type,
            'source': realtime_data.get('symbol', 'Historical') if 'realtime_data' in locals() and realtime_data.get('price') else 'Historical',
            'currency': currency,
            'unit': unit,
            'exchange_rate': self.usd_vnd_rate if in_vnd else None,
            'last_known': {
                'date': last_date.strftime('%Y-%m-%d'),
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
    
    def _predict_ridge(self):
        """Make predictions using Ridge models."""
        latest_features = self.data[self.feature_columns].iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        predictions_scaled = []
        for model in self.models:
            pred = model.predict(latest_features_scaled)
            predictions_scaled.append(pred[0])
        
        predictions_scaled = np.array(predictions_scaled).reshape(1, -1)
        predictions = self.target_scaler.inverse_transform(predictions_scaled)[0]
        
        return predictions
    
    def _predict_lstm(self):
        """Make predictions using LSTM model."""
        from src.data_processor import SilverDataProcessor
        
        processor = SilverDataProcessor(
            sequence_length=self.sequence_length,
            prediction_days=self.prediction_days
        )
        
        # Load standard format data
        standard_data_path = os.path.join(self.base_dir, 'dataset', 'silver_price.csv')
        processor.load_data(standard_data_path)
        processor.clean_data()
        processor.prepare_simple_data()
        processor.scaler = self.scaler
        
        latest_sequence = processor.get_latest_sequence()
        predictions_scaled = self.keras_model.predict(latest_sequence, verbose=0)
        
        scaled_reshaped = predictions_scaled[0].reshape(-1, 1)
        predictions = processor.scaler.inverse_transform(scaled_reshaped).flatten()
        
        return predictions
    
    def _convert_to_vnd(self, prices_usd: np.ndarray) -> np.ndarray:
        """
        Convert USD/oz to VND/lượng with Vietnam market premium.
        
        Formula: price_usd * troy_ounce_to_luong * usd_vnd_rate * vietnam_premium
        
        - 1 lượng = 37.5g = 1.20565 troy oz
        - Vietnam premium accounts for import costs, taxes, and retail markup
        """
        return prices_usd * self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
    
    def _convert_to_vnd_single(self, price_usd: float) -> float:
        """Convert a single USD price to VND with Vietnam premium."""
        return price_usd * self.troy_ounce_to_luong * self.usd_vnd_rate * self.vietnam_premium
    
    def _get_future_trading_dates(self, last_date: datetime, num_days: int) -> List[datetime]:
        dates = []
        current_date = last_date
        
        while len(dates) < num_days:
            current_date = current_date + timedelta(days=1)
            if current_date.weekday() < 5:
                dates.append(current_date)
        
        return dates
    
    def get_historical_data(self, days: int = 30, in_vnd: bool = True) -> Dict:
        """Get historical price data."""
        if self.data is None:
            self.load()
        
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
                    'date': date.strftime('%Y-%m-%d'),
                    'price': float(price)
                }
                for date, price in zip(historical_dates, historical_prices)
            ]
        }
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if self.model_type == 'ridge':
            info_path = os.path.join(self.model_dir, 'ridge_training_info.json')
        else:
            info_path = os.path.join(self.model_dir, 'training_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        
        return {'error': 'Training info not found'}


# Create default predictor (Ridge)
def get_predictor(model_type='ridge'):
    """Factory function to get predictor."""
    predictor = UnifiedPredictor(model_type=model_type)
    predictor.load()
    return predictor


if __name__ == "__main__":
    # Test
    predictor = UnifiedPredictor(model_type='ridge')
    predictor.load()
    
    result = predictor.predict(in_vnd=True)
    
    print("\n" + "=" * 60)
    print("🔮 DỰ ĐOÁN GIÁ BẠC 7 NGÀY TỚI")
    print("=" * 60)
    print(f"Model: {result['model_type'].upper()}")
    print(f"Ngày cuối: {result['last_known']['date']}")
    print(f"Giá hiện tại: {result['last_known']['price']:,.0f} VND")
    
    print("\nDự đoán:")
    for pred in result['predictions']:
        print(f"  Ngày {pred['day']}: {pred['price']:,.0f} VND ({pred['change']['percentage']:+.2f}%)")
