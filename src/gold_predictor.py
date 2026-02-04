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
        
        self.usd_vnd_rate = DEFAULT_USD_VND_RATE
        self.troy_ounce_to_luong = 1.20565
        self.troy_ounce_to_luong = 1.20565
        self.vietnam_premium = 1.125 # Adjusted to match market (172M vs 189M)
        
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
        
        extended_features = base_features + [
            'momentum_3', 'momentum_7', 'momentum_14', 'momentum_30',
            'roc_7', 'roc_14', 'roc_30',
            'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
            'rolling_min_7', 'rolling_max_7'
        ] + market_features
        
        self.feature_columns = [col for col in extended_features if col in self.data.columns]
        print(f"Selected {len(self.feature_columns)} extended features (incl. VIX, DXY, Oil, US10Y)")
        
        # Handle missing values
        self.data[self.feature_columns] = self.data[self.feature_columns].ffill().fillna(0)
        
    def train(self, test_size: float = 0.2, use_pca: bool = True, pca_variance: float = 0.95, alpha: float = 1.0):
        """Train Ridge Regression models with extended features."""
        print(f"Training with extended features (alpha={alpha})...")
        
        # Prepare data
        X = self.data[self.feature_columns].values
        
        # Create targets for each prediction day
        targets = {}
        for day in range(1, self.prediction_days + 1):
            target_col = f'target_day_{day}'
            self.data[target_col] = self.data['price'].shift(-day)
            targets[day] = self.data[target_col].values
        
        # Remove rows with NaN targets
        valid_mask = ~np.isnan(targets[self.prediction_days])
        X = X[valid_mask]
        for day in targets:
            targets[day] = targets[day][valid_mask]
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        if use_pca:
            self.pca = PCA(n_components=pca_variance)
            X_pca = self.pca.fit_transform(X_scaled)
            print(f"PCA: {X_scaled.shape[1]} -> {X_pca.shape[1]} components")
            X_train_data = X_pca
        else:
            X_train_data = X_scaled
        
        # Train models for each day
        self.models = {}
        self.metrics = {'r2': {}, 'mape': {}}
        
        for day in range(1, self.prediction_days + 1):
            y = targets[day]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_train_data, y, test_size=test_size, shuffle=False
            )
            
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = model.score(X_test, y_test)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            self.models[day] = model
            self.metrics['r2'][day] = r2
            self.metrics['mape'][day] = mape
            
            print(f"  Day {day}: R2={r2:.4f}, MAPE={mape:.2f}%")
        
        avg_r2 = np.mean(list(self.metrics['r2'].values()))
        avg_mape = np.mean(list(self.metrics['mape'].values()))
        print(f"Average R2={avg_r2:.4f}, MAPE={avg_mape:.2f}%")
        
    def save_model(self):
        """Save trained model."""
        os.makedirs(self.model_dir, exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'prediction_days': self.prediction_days
        }
        
        path = os.path.join(self.model_dir, 'gold_ridge_models.pkl')
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
        
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
        
        print(f"Loaded gold model ({len(self.models)} day predictions)")
        
    def predict(self, in_vnd: bool = True) -> Dict:
        """Make predictions for the next 7 days."""
        if self.models is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Get latest data point
        # Prepare features
        latest = self.data.iloc[-1]
        latest_features = latest[self.feature_columns].values.reshape(1, -1)
        
        # Scale
        if self.scaler:
            latest_features_scaled = self.scaler.transform(latest_features)
            # PCA
            if self.pca:
                latest_features_scaled = self.pca.transform(latest_features_scaled)
        else:
             latest_features_scaled = latest_features

        # Predict (Result is USD/oz)
        predictions_usd_raw = []
        for day in range(1, self.prediction_days + 1):
            pred = self.models[day].predict(latest_features_scaled)[0]
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
            'model_type': 'Extended Feature Ridge Regression',
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
            
        # Features at t-1
        features_yesterday = self.data[self.feature_columns].iloc[-2].values.reshape(1, -1)
        
        # Scale & PCA
        if self.scaler:
            feat_scaled = self.scaler.transform(features_yesterday)
            if self.pca:
                feat_scaled = self.pca.transform(feat_scaled)
        else:
            feat_scaled = features_yesterday
            
        # Predict Day 1 using model[1]
        pred_usd = self.models[1].predict(feat_scaled)[0]
        
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
