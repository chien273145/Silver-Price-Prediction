"""
Daily Update Script
Orchestrates the daily data update and model retraining process.
1. Fetches latest market data (Silver, Gold, DXY, VIX, Oil, US10Y).
2. Updates and cleans datasets.
3. Retrains all 3 AI models (Silver, Gold, VN Gold).
4. Saves models for the API to reload.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("daily_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DailyUpdate")

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from backend.realtime_data import RealTimeDataFetcher
from src.enhanced_predictor import EnhancedPredictor
from src.gold_predictor import GoldPredictor
from src.vietnam_gold_predictor import VietnamGoldPredictor

def fetch_and_update_data():
    """Fetch latest data and update CSV datasets."""
    logger.info("Step 1: Fetching latest market data...")
    
    fetcher = RealTimeDataFetcher()
    
    # 1. Update World Gold/Silver/Macro Data (for dataset_enhanced.csv & gold_enhanced_dataset.csv)
    # We need to fetch data for all ticking symbols
    symbols = {
        'Silver': fetcher.SILVER_SYMBOL,
        'Gold': fetcher.GOLD_SYMBOL,
        'DXY': fetcher.DXY_SYMBOL,
        'VIX': fetcher.VIX_SYMBOL,
        'Oil': fetcher.OIL_SYMBOL,
        'US10Y': '^TNX'
    }
    
    # Check current datasets to find last date
    silver_path = os.path.join(BASE_DIR, 'dataset', 'dataset_enhanced.csv')
    try:
        df_silver = pd.read_csv(silver_path)
        df_silver['Date'] = pd.to_datetime(df_silver['Date'])
        last_date = df_silver['Date'].max()
        logger.info(f"Current Silver dataset ends at: {last_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        logger.error(f"Could not read silver dataset: {e}")
        return False

    # Fetch missing days (last_date + 1 to now)
    days_to_fetch = (datetime.now() - last_date).days + 2
    if days_to_fetch < 1:
        logger.info("Data is already up to date.")
        return True
        
    logger.info(f"Fetching last {days_to_fetch} days of data...")
    
    new_data = {}
    for name, symbol in symbols.items():
        try:
            hist = fetcher.get_historical_prices(days=days_to_fetch, symbol=symbol)
            if hist and hist['data']:
                df_hist = pd.DataFrame(hist['data'])
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                new_data[name] = df_hist.set_index('date')
                logger.info(f"  Fetched {name}: {len(df_hist)} rows")
            else:
                logger.warning(f"  No data found for {name}")
        except Exception as e:
            logger.error(f"  Error fetching {name}: {e}")

    # If we have Silver and Gold (minimum requirement), proceed to patch
    if 'Silver' in new_data and 'Gold' in new_data:
        # Patch Silver Dataset (dataset_enhanced.csv)
        # Columns mapped: Date, Silver_Close, Silver_Open, Silver_High, Silver_Low, Gold, DXY, VIX...
        
        # Merge all new data on date index
        df_new = new_data['Silver'][['close', 'open', 'high', 'low']].rename(columns={
            'close': 'Silver_Close', 'open': 'Silver_Open', 'high': 'Silver_High', 'low': 'Silver_Low'
        })
        
        for name in ['Gold', 'DXY', 'VIX']:
            if name in new_data:
                # We assume column name in dataset is just the name (Gold, DXY...) based on previous analysis
                # Checking repair_and_retrain.py or similar might clarify, but let's assume standard names
                # In fetch_external_data.py: 'Gold' comes from 'gold_geopolitical_dataset.csv' which has 'gold_close'?
                # Wait, dataset_enhanced.csv usually has 'Gold' column for price.
                # Let's check columns of existing df_silver
                pass 
                
        # Actually, let's use the logic from src/enhanced_predictor.py _patch_missing_data
        # It joins df_silver with df_gold, df_dxy, df_vix
        
        if 'Gold' in new_data: df_new['Gold'] = new_data['Gold']['close']
        if 'DXY' in new_data: df_new['DXY'] = new_data['DXY']['close']
        if 'VIX' in new_data: df_new['VIX'] = new_data['VIX']['close']
        
        # Filter strictly new dates
        df_new = df_new[df_new.index > last_date]
        
        if not df_new.empty:
            df_new = df_new.reset_index().rename(columns={'date': 'Date'})
            # Fill missing columns with 0 or ffill
            for col in df_silver.columns:
                if col not in df_new.columns:
                    df_new[col] = 0 # Or handling specific columns like 'Basic_Resource_Reserves'
            
            # Concat
            df_silver_updated = pd.concat([df_silver, df_new], ignore_index=True)
            # Simple dedupe
            df_silver_updated = df_silver_updated.drop_duplicates(subset=['Date'], keep='last')
            
            # Save
            df_silver_updated.to_csv(silver_path, index=False)
            logger.info(f"Updated {silver_path} with {len(df_new)} new rows.")
        else:
            logger.info("No new rows to add to Silver dataset.")

    # Patch Gold Dataset (gold_enhanced_dataset.csv)
    # Similar logic, just different file and columns
    gold_path = os.path.join(BASE_DIR, 'dataset', 'gold_enhanced_dataset.csv')
    if os.path.exists(gold_path):
        try:
            df_gold = pd.read_csv(gold_path)
            df_gold['date'] = pd.to_datetime(df_gold['date'])
            last_date_gold = df_gold['date'].max()
            
            # Prepare new data
            if 'Gold' in new_data:
                df_new_gold = new_data['Gold'][['close', 'open', 'high', 'low']].rename(columns={
                    'close': 'gold_close', 'open': 'gold_open', 'high': 'gold_high', 'low': 'gold_low'
                })
                # Add others
                if 'Silver' in new_data: df_new_gold['silver_close'] = new_data['Silver']['close']
                if 'DXY' in new_data: df_new_gold['dxy'] = new_data['DXY']['close']
                if 'VIX' in new_data: df_new_gold['vix'] = new_data['VIX']['close']
                if 'Oil' in new_data: df_new_gold['oil'] = new_data['Oil']['close']
                if 'US10Y' in new_data: df_new_gold['us10y'] = new_data['US10Y']['close']

                df_new_gold = df_new_gold[df_new_gold.index > last_date_gold]
                
                if not df_new_gold.empty:
                    df_new_gold = df_new_gold.reset_index() # date becomes column
                    # Align columns
                    for col in df_gold.columns:
                        if col not in df_new_gold.columns:
                            df_new_gold[col] = 0 # Fill GPR, etc with 0 or NaN
                    
                    df_gold_updated = pd.concat([df_gold, df_new_gold], ignore_index=True)
                    df_gold_updated.to_csv(gold_path, index=False)
                    logger.info(f"Updated {gold_path} with {len(df_new_gold)} new rows.")
        except Exception as e:
            logger.error(f"Error updating gold dataset: {e}")

    return True

def clean_data():
    """Remove anomalous outliers."""
    logger.info("Step 2: Cleaning data (removing anomalies)...")
    
    # Silver
    try:
        path = os.path.join(BASE_DIR, 'dataset', 'dataset_enhanced.csv')
        df = pd.read_csv(path)
        # Rule: Silver price <= 100 (Safe buffer, real is ~30)
        clean_df = df[df['Silver_Close'] <= 100]
        if len(clean_df) < len(df):
            logger.warning(f"Removed {len(df) - len(clean_df)} anomalous Silver rows.")
            clean_df.to_csv(path, index=False)
    except Exception as e:
        logger.error(f"Error cleaning Silver: {e}")

    # Gold
    try:
        path = os.path.join(BASE_DIR, 'dataset', 'gold_enhanced_dataset.csv')
        df = pd.read_csv(path)
        # Rule: Gold <= 5000 (Safe buffer, real is ~2700)
        col = 'gold_close' if 'gold_close' in df.columns else 'price'
        if col in df.columns:
            clean_df = df[df[col] <= 5000]
            if len(clean_df) < len(df):
                logger.warning(f"Removed {len(df) - len(clean_df)} anomalous Gold rows.")
                clean_df.to_csv(path, index=False)
    except Exception as e:
        logger.error(f"Error cleaning Gold: {e}")

def retrain_models():
    """Retrain all 3 models."""
    logger.info("Step 3: Retraining models...")
    
    # 1. Silver Model
    try:
        logger.info("  Training Silver EnhancedPredictor...")
        silver_model = EnhancedPredictor()
        silver_model.load_data()
        metrics = silver_model.train()
        silver_model.save_model()
        logger.info(f"  [SUCCESS] Silver Model Saved. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"  [FAILED] Silver Model: {e}")

    # 2. World Gold Model
    try:
        logger.info("  Training World GoldPredictor...")
        gold_model = GoldPredictor()
        gold_model.load_data()
        gold_model.create_features()
        metrics = gold_model.train()
        gold_model.save_model()
        logger.info(f"  [SUCCESS] Gold Model Saved. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"  [FAILED] Gold Model: {e}")

    # 3. Vietnam Gold Model
    try:
        logger.info("  Training VietnamGoldPredictor...")
        vn_model = VietnamGoldPredictor()
        vn_model.load_vietnam_data()
        vn_model.load_world_data()
        vn_model.merge_datasets()
        vn_model.create_transfer_features()
        metrics = vn_model.train()
        vn_model.save_model()
        logger.info(f"  [SUCCESS] VN Gold Model Saved. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"  [FAILED] VN Gold Model: {e}")
        import traceback
        traceback.print_exc()

def run_daily_update():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info(f"STARTING DAILY UPDATE: {datetime.now()}")
    logger.info("=" * 60)
    
    try:
        # 1. Fetch
        fetch_and_update_data()
        
        # 2. Clean
        clean_data()
        
        # 3. Retrain
        retrain_models()
        
        logger.info("Daily update completed successfully.")
        return True
    except Exception as e:
        logger.critical(f"Daily update CRASHED: {e}")
        return False

if __name__ == "__main__":
    run_daily_update()
