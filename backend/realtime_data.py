"""
Real-time Data Module
Fetches real-time silver prices and USD/VND exchange rates from APIs.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("⚠️ yfinance not installed. Real-time data will be limited.")


class RealTimeDataFetcher:
    """
    Fetches real-time silver prices and exchange rates.
    Uses Yahoo Finance for silver prices and free APIs for exchange rates.
    """
    
    # Silver symbols
    SILVER_SYMBOL = "XAGUSD=X"  # Silver Spot (Matching Investing.com data)
    SILVER_FUTURES = "SI=F"     # Silver Futures (Backup)
    SILVER_ETF = "SLV"          # iShares Silver Trust (Last resort)
    
    def __init__(self, cache_duration_minutes: int = 5):
        """
        Initialize the fetcher.
        
        Args:
            cache_duration_minutes: How long to cache data
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.cache = {}
        
    def get_silver_price(self) -> Dict:
        """
        Get current silver price in USD per troy ounce.
        
        Returns:
            Dictionary with price information
        """
        cache_key = 'silver_price'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        result = {
            'symbol': self.SILVER_SYMBOL,
            'timestamp': datetime.now().isoformat(),
            'source': None,
            'price': None,
            'change': None,
            'change_percent': None,
            'high': None,
            'low': None,
            'previous_close': None,
            'error': None
        }
        
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(self.SILVER_SYMBOL)
                info = ticker.info
                
                # Get current price
                result['price'] = info.get('regularMarketPrice') or info.get('previousClose')
                result['previous_close'] = info.get('previousClose')
                result['high'] = info.get('dayHigh') or info.get('regularMarketDayHigh')
                result['low'] = info.get('dayLow') or info.get('regularMarketDayLow')
                result['source'] = 'yahoo_finance'
                
                if result['price'] and result['previous_close']:
                    result['change'] = result['price'] - result['previous_close']
                    result['change_percent'] = (result['change'] / result['previous_close']) * 100
                    
            except Exception as e:
                result['error'] = str(e)
                
                # 1. Try Futures as first backup
                try:
                    ticker = yf.Ticker(self.SILVER_FUTURES)
                    info = ticker.info
                    price = info.get('regularMarketPrice') or info.get('previousClose')
                    if price:
                        result['price'] = price
                        result['previous_close'] = info.get('previousClose')
                        result['source'] = 'yahoo_finance_futures'
                        result['symbol'] = self.SILVER_FUTURES
                        result['error'] = None
                except:
                    # 2. Try ETF as last resort
                    try:
                        ticker = yf.Ticker(self.SILVER_ETF)
                        hist = ticker.history(period='1d')
                        if not hist.empty:
                            result['price'] = float(hist['Close'].iloc[-1])
                            result['source'] = 'yahoo_finance_etf'
                            result['symbol'] = self.SILVER_ETF
                            result['error'] = None
                    except:
                        pass
        
        # Fallback to free API if yfinance fails
        if result['price'] is None:
            try:
                # Use free metals API (backup)
                result = self._fetch_from_backup_api(result)
            except Exception as e:
                result['error'] = f"All APIs failed: {str(e)}"
        
        # Cache result
        self._update_cache(cache_key, result)
        
        return result
    
    def get_usd_vnd_rate(self) -> Dict:
        """
        Get current USD to VND exchange rate.
        
        Returns:
            Dictionary with exchange rate information
        """
        cache_key = 'usd_vnd_rate'
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        result = {
            'pair': 'USD/VND',
            'timestamp': datetime.now().isoformat(),
            'source': None,
            'rate': None,
            'error': None
        }
        
        # Try free exchange rate APIs
        apis = [
            self._fetch_rate_exchangerate_api,
            self._fetch_rate_fixer_backup,
        ]
        
        for api_func in apis:
            try:
                rate = api_func()
                if rate:
                    result['rate'] = rate
                    result['source'] = api_func.__name__
                    break
            except:
                continue
        
        # Fallback to default rate
        if result['rate'] is None:
            result['rate'] = 24500  # Default rate
            result['source'] = 'default'
            result['error'] = 'Could not fetch live rate, using default'
        
        # Cache result
        self._update_cache(cache_key, result)
        
        return result
    
    def _fetch_rate_exchangerate_api(self) -> Optional[float]:
        """Fetch rate from ExchangeRate-API (free tier)."""
        try:
            url = "https://open.er-api.com/v6/latest/USD"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('result') == 'success':
                    return data['rates'].get('VND')
        except:
            pass
        return None
    
    def _fetch_rate_fixer_backup(self) -> Optional[float]:
        """Backup exchange rate source."""
        # In production, you'd use a real API key here
        # For now, return None to use default
        return None
    
    def _fetch_from_backup_api(self, result: Dict) -> Dict:
        """Fetch silver price from backup API."""
        # This is a placeholder - in production you'd use a real metals API
        # For now, we'll return the result as-is (will use cached/historical data)
        result['error'] = 'Live API unavailable'
        return result
    
    def get_historical_prices(self, days: int = 30) -> Dict:
        """
        Get historical silver prices.
        
        Args:
            days: Number of days of history
            
        Returns:
            Dictionary with historical price data
        """
        result = {
            'symbol': self.SILVER_SYMBOL,
            'period': f'{days}d',
            'timestamp': datetime.now().isoformat(),
            'data': [],
            'error': None
        }
        
        if HAS_YFINANCE:
            try:
                ticker = yf.Ticker(self.SILVER_SYMBOL)
                hist = ticker.history(period=f'{days}d')
                
                for date, row in hist.iterrows():
                    result['data'].append({
                        'date': date.strftime('%Y-%m-%d'),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']) if row['Volume'] else 0
                    })
                    
            except Exception as e:
                result['error'] = str(e)
        else:
            result['error'] = 'yfinance not available'
        
        return result
    
    def update_csv_with_latest(self, csv_path: str) -> bool:
        """
        Update CSV file with latest price data.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            True if successful
        """
        import pandas as pd
        
        try:
            # Load existing data
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            last_date = df['date'].max()
            
            # Get new data from Yahoo Finance
            if HAS_YFINANCE:
                ticker = yf.Ticker(self.SILVER_SYMBOL)
                # Get data from last date to now
                new_data = ticker.history(start=last_date + timedelta(days=1))
                
                if not new_data.empty:
                    # Prepare new rows
                    new_rows = []
                    for date, row in new_data.iterrows():
                        new_rows.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'price': float(row['Close'])
                        })
                    
                    # Append to dataframe
                    new_df = pd.DataFrame(new_rows)
                    df = pd.concat([df, new_df], ignore_index=True)
                    
                    # Remove duplicates
                    df = df.drop_duplicates(subset=['date'], keep='last')
                    
                    # Save
                    df.to_csv(csv_path, index=False)
                    print(f"✓ Added {len(new_rows)} new records to {csv_path}")
                    return True
                else:
                    print("No new data available")
                    return True
                    
        except Exception as e:
            print(f"Error updating CSV: {e}")
            return False
        
        return False
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key]['timestamp']
        return datetime.now() - cached_time < self.cache_duration
    
    def _update_cache(self, key: str, data: Dict):
        """Update cache with new data."""
        self.cache[key] = {
            'timestamp': datetime.now(),
            'data': data
        }


# Global instance
_fetcher = None

def get_fetcher() -> RealTimeDataFetcher:
    """Get global fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = RealTimeDataFetcher()
    return _fetcher


def get_current_silver_price() -> Dict:
    """Get current silver price."""
    return get_fetcher().get_silver_price()


def get_exchange_rate() -> Dict:
    """Get USD/VND exchange rate."""
    return get_fetcher().get_usd_vnd_rate()


if __name__ == "__main__":
    # Test the fetcher
    fetcher = RealTimeDataFetcher()
    
    print("=" * 50)
    print("Testing Real-time Data Fetcher")
    print("=" * 50)
    
    print("\n📈 Silver Price:")
    silver = fetcher.get_silver_price()
    print(json.dumps(silver, indent=2))
    
    print("\n💱 USD/VND Rate:")
    rate = fetcher.get_usd_vnd_rate()
    print(json.dumps(rate, indent=2))
    
    print("\n📊 Historical Data (last 7 days):")
    hist = fetcher.get_historical_prices(7)
    print(f"   Got {len(hist['data'])} records")
    if hist['data']:
        print(f"   Latest: {hist['data'][-1]}")
