import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

def test_endpoint(name, url):
    print(f"\n--- Testing {name} ---")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: 200 OK")
            return data
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def verify_silver():
    # 1. Test Transparency (UI Accuracy)
    transparency = test_endpoint("Performance Transparency", f"{BASE_URL}/api/performance-transparency")
    if transparency and transparency.get('success'):
        perf = transparency['performance']
        print(f"   Date: {perf['date']}")
        print(f"   Accuracy: {perf['accuracy']['overall']}%")
        print(f"   Grade: {perf['accuracy']['grade']}")
        print("   ✅ UI 'Accuracy Section' should now be populated.")
    else:
        print("   ❌ Transparency endpoint failed.")

    # 2. Test Predictions (Future Dates)
    predictions = test_endpoint("Silver Predictions", f"{BASE_URL}/api/predict?currency=VND")
    if predictions and predictions.get('success'):
        preds = predictions['predictions']
        if not preds:
            print("   ⚠️ No predictions returned.")
        else:
            first_date = preds[0]['date']
            last_date = preds[-1]['date']
            print(f"   Prediction Range: {first_date} to {last_date}")
            
            # Check if dates are future
            today = datetime.now().date()
            last_date_obj = datetime.strptime(last_date, "%Y-%m-%d").date()
            
            if last_date_obj > today:
                print(f"   ✅ Future dates detected (Current Date: {today})")
                print("   ✅ Chart 'Old Dates' issue should be FIXED.")
            else:
                print(f"   ❌ DATES ARE OLD! Last prediction date ({last_date}) is not after today ({today}).")
                
        # Check if live data was used
        if predictions.get('is_live_prediction'):
             print("   ✅ using LIVE market data.")
        else:
             print("   ℹ️ Using standard cached data.")

if __name__ == "__main__":
    verify_silver()
