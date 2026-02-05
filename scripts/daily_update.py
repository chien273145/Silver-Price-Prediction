
import os
import sys
import pandas as pd
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scrapers.sjc_scraper import SJCScraper
from src.vietnam_gold_predictor import VietnamGoldPredictor
from backend.realtime_data import RealTimeDataFetcher

DATASET_DIR = os.path.join('data', 'datasets')
SJC_CSV_PATH = os.path.join(DATASET_DIR, 'gold_price_sjc_complete.csv')

def update_sjc_data():
    """Fetch today's SJC price and append to CSV."""
    print("🚀 updating SJC data...")
    scraper = SJCScraper()
    result = scraper.scrape()
    
    if not result.success or not result.items:
        print(f"❌ Failed to scrape SJC: {result.error}")
        return False
        
    # Find SJC 1L - 10L - HCM (Standard benchmark)
    # The scraper returns items like "Vàng SJC 1L - 10L - TP.HCM"
    # Or just generic SJC at HCM location
    target_item = None
    for item in result.items:
        if "HCM" in item.location and "SJC" in item.brand:
            # Prefer 1L-10L type if available
            if "1L" in item.product_type:
                target_item = item
                break
            # Fallback to any SJC HCM
            target_item = item
            
    if not target_item:
        print("⚠️ No suitable SJC item found for HCM.")
        return False
        
    print(f"✓ Found: {target_item.product_type} - Buy: {target_item.buy_price} - Sell: {target_item.sell_price}")
    
    # Load existing CSV
    if not os.path.exists(SJC_CSV_PATH):
        print("❌ CSV not found.")
        return False
        
    df = pd.read_csv(SJC_CSV_PATH)
    
    # Check if today exists
    today_str = datetime.now().strftime('%Y-%m-%d')
    if today_str in df['date'].values:
        print(f"⚠️ Data for {today_str} already exists. Skipping append.")
    else:
        # Append new row
        new_row = {
            'date': today_str,
            'buy_price': target_item.buy_price / 1_000_000, # Convert to Million VND
            'sell_price': target_item.sell_price / 1_000_000,
            'updated_at': datetime.now().isoformat()
        }
        
        # Calculate derived fields if they exist in CSV (mid_price, etc) handled by loader usually?
        # But loader calculates them on load. We just need raw columns.
        # Check raw columns of CSV. Assuming date,buy_price,sell_price are core.
        
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(SJC_CSV_PATH, index=False)
        print(f"✅ Added data for {today_str} to {SJC_CSV_PATH}")
        
    return True

def run_retraining():
    """Run model retraining pipeline."""
    print("\n🔄 Starting Retraining Pipeline...")
    
    try:
        predictor = VietnamGoldPredictor()
        
        # 1. Load SJC Data (Updated above)
        print("1. Loading Vietnam Data...")
        predictor.load_vietnam_data()
        
        # 2. Load & Patch World Data
        # This automatically calls fetcher and patches missing days
        print("2. Loading & Patching World Data...")
        predictor.load_world_data() 
        
        # 3. Merge
        predictor.merge_datasets()
        
        # 4. Train
        print("3. Training Model...")
        predictor.create_transfer_features()
        metrics = predictor.train()
        
        # 5. Save
        predictor.save_model()
        print("✅ Vietnam Gold Retraining Complete!")
        print(f"Metrics: {metrics}")
        
        # 6. SILVER Retraining
        print("\n🥈 Starting Silver Model Retraining...")
        from src.enhanced_predictor import EnhancedPredictor
        silver_predictor = EnhancedPredictor()
        
        # Load & Auto-Patch Silver Data
        print("4. Loading Silver Data (Auto-Patching)...")
        silver_predictor.load_data() 
        # load_data now triggers _patch_missing_data automatically
        
        print("5. Training Silver Model...")
        silver_predictor.create_features()
        silver_metrics = silver_predictor.train()
        silver_predictor.save_model()
        print("✅ Silver Retraining Complete!")
        print(f"Metrics: {silver_metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Retraining Failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"📅 Daily Update Job: {datetime.now()}")
    
    # Ensure dirs
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    success_sjc = update_sjc_data()
    success_train = run_retraining()
    
        
    if success_sjc and success_train:
        print("\n✨ ALL MODELS RETRAINED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("\n⚠️ SOME TASKS FAILED")
        # Don't exit error code to ensure git commit happens for parcial success?
        # Ideally exit 1 if critical failure.
        sys.exit(1 if not success_train else 0)
