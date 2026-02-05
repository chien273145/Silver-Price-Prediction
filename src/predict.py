"""
Prediction Module for Silver Price
Loads trained model and makes predictions for the next 7 days.
Supports conversion to VND currency.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import SilverDataProcessor
from src.model import SilverPriceLSTM


# Default USD to VND exchange rate (sẽ được cập nhật từ API)
DEFAULT_USD_VND_RATE = 24500


class SilverPredictor:
    """
    Predictor class for silver prices.
    Handles model loading, prediction, and currency conversion.
    """
    
    def __init__(self,
                 model_path: str = None,
                 scaler_path: str = None,
                 data_path: str = None,
                 sequence_length: int = 60,
                 prediction_days: int = 7):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to saved scaler
            data_path: Path to price data CSV
            sequence_length: Number of past days used by model
            prediction_days: Number of days to predict
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.model_path = model_path or os.path.join(self.base_dir, 'models', 'silver_lstm_model.h5')
        self.scaler_path = scaler_path or os.path.join(self.base_dir, 'models', 'scaler.pkl')
        self.data_path = data_path or os.path.join(self.base_dir, 'dataset', 'silver_price.csv')
        
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        
        self.model = None
        self.keras_model = None
        self.processor = None
        self.usd_vnd_rate = DEFAULT_USD_VND_RATE
        
        # Silver is priced per troy ounce - 1 lượng = 1.20565 troy ounce
        # Giá bạc Việt Nam thường tính theo lượng (37.5g)
        self.troy_ounce_to_luong = 1.20565
        
    def load(self):
        """Load model and prepare data processor."""
        print("[LOADING] Loading model and data...")
        
        # Load model directly using keras
        from tensorflow import keras
        self.keras_model = keras.models.load_model(self.model_path, compile=False)
        print(f"✓ Model loaded from {self.model_path}")
        
        # Load scaler (support both old and new format)
        import joblib
        scaler_data = joblib.load(self.scaler_path)
        
        if isinstance(scaler_data, dict):
            # New format with feature_columns
            self.scaler = scaler_data['scaler']
            self.feature_columns = scaler_data.get('feature_columns', ['price'])
            print(f"✓ Scaler loaded with {len(self.feature_columns)} features")
        else:
            # Old format (just the scaler)
            self.scaler = scaler_data
            self.feature_columns = ['price']
            print(f"✓ Scaler loaded (legacy format)")
        
        # Load processor and data
        self.processor = SilverDataProcessor(
            sequence_length=self.sequence_length,
            prediction_days=self.prediction_days
        )
        self.processor.load_data(self.data_path)
        self.processor.clean_data()
        self.processor.prepare_simple_data()
        
        # Set the scaler to the processor
        self.processor.scaler = self.scaler
        
        print("✓ Model and data loaded successfully")
        
    def set_exchange_rate(self, rate: float):
        """Set USD to VND exchange rate."""
        self.usd_vnd_rate = rate
        print(f"✓ Exchange rate set to: {rate:,.0f} VND/USD")
        
    def predict(self, in_vnd: bool = True) -> Dict:
        """
        Make predictions for the next 7 days.
        
        Args:
            in_vnd: Whether to convert prices to VND
            
        Returns:
            Dictionary with predictions and metadata
        """
        if self.keras_model is None or self.processor is None:
            self.load()
        
        # Get latest sequence
        latest_sequence = self.processor.get_latest_sequence()
        
        # Make prediction
        predictions_scaled = self.keras_model.predict(latest_sequence, verbose=0)
        
        # Inverse transform to get actual prices (USD per troy ounce)
        predictions_usd = self._inverse_scale(predictions_scaled[0])
        
        # Get last known date and price
        dates = self.processor.get_dates()
        prices = self.processor.get_prices()
        last_date = dates.iloc[-1]
        last_price_usd = prices.iloc[-1]
        
        # Generate future dates (skip weekends for trading days)
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
        
        # Build result
        result = {
            'timestamp': datetime.now().isoformat(),
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
    
    def _inverse_scale(self, scaled_values: np.ndarray) -> np.ndarray:
        """Inverse scale predictions to original price range."""
        # The scaler was fitted on price data with shape (n, 1)
        scaled_reshaped = scaled_values.reshape(-1, 1)
        return self.processor.scaler.inverse_transform(scaled_reshaped).flatten()
    
    def _convert_to_vnd(self, prices_usd: np.ndarray) -> np.ndarray:
        """
        Convert USD/oz prices to VND/lượng.
        
        1 lượng (tael) = 37.5 grams
        1 troy ounce = 31.1035 grams
        So 1 lượng = 37.5/31.1035 = 1.20565 troy ounces
        
        Price per lượng = Price per oz * 1.20565 * USD/VND rate
        """
        return prices_usd * self.troy_ounce_to_luong * self.usd_vnd_rate
    
    def _convert_to_vnd_single(self, price_usd: float) -> float:
        """Convert a single USD price to VND."""
        return price_usd * self.troy_ounce_to_luong * self.usd_vnd_rate
    
    def _get_future_trading_dates(self, last_date: datetime, num_days: int) -> List[datetime]:
        """Generate future trading dates (excluding weekends)."""
        dates = []
        current_date = last_date
        
        while len(dates) < num_days:
            current_date = current_date + timedelta(days=1)
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:
                dates.append(current_date)
        
        return dates
    
    def get_historical_data(self, days: int = 30, in_vnd: bool = True) -> Dict:
        """
        Get historical price data.
        
        Args:
            days: Number of days of historical data
            in_vnd: Whether to convert to VND
            
        Returns:
            Dictionary with historical data
        """
        if self.processor is None:
            self.load()
        
        dates = self.processor.get_dates()
        prices = self.processor.get_prices()
        
        # Get last N days
        historical_dates = dates.tail(days).tolist()
        historical_prices = prices.tail(days).tolist()
        
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
        """Get information about the trained model."""
        info_path = os.path.join(self.base_dir, 'models', 'training_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                return json.load(f)
        
        return {
            'error': 'Training info not found',
            'model_path': self.model_path
        }


def main():
    """Test the predictor."""
    predictor = SilverPredictor()
    predictor.load()
    
    # Make prediction in VND
    print("\n" + "=" * 60)
    print("🔮 DỰ ĐOÁN GIÁ BẠC 7 NGÀY TỚI (VND)")
    print("=" * 60)
    
    result = predictor.predict(in_vnd=True)
    
    print(f"\n📅 Ngày cuối cùng có dữ liệu: {result['last_known']['date']}")
    print(f"💰 Giá hiện tại: {result['last_known']['price']:,.0f} {result['unit']}")
    print(f"💱 Tỷ giá: {result['exchange_rate']:,.0f} VND/USD")
    
    print(f"\n[DATA] Dự đoán:")
    print("-" * 50)
    for pred in result['predictions']:
        change_symbol = "[UP]" if pred['change']['percentage'] > 0 else "[DOWN]"
        print(f"  Ngày {pred['day']} ({pred['date']}): {pred['price']:,.0f} VND "
              f"{change_symbol} {pred['change']['percentage']:+.2f}%")
    
    print(f"\n[UP] Tổng kết:")
    print(f"   Xu hướng: {'Tăng ⬆️' if result['summary']['trend'] == 'up' else 'Giảm ⬇️'}")
    print(f"   Thay đổi: {result['summary']['total_change']:+,.0f} VND ({result['summary']['total_change_pct']:+.2f}%)")
    print(f"   Giá thấp nhất: {result['summary']['min_price']:,.0f} VND")
    print(f"   Giá cao nhất: {result['summary']['max_price']:,.0f} VND")


if __name__ == "__main__":
    main()
