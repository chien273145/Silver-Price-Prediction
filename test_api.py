import requests

try:
    resp = requests.get("http://127.0.0.1:8000/api/prices/local", timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Response Keys: {resp.json().keys()}")
    data = resp.json()
    
    items = data.get('items', data.get('data', {}).get('items', []))
    print(f"Total items: {len(items)}")
    print("\n=== Silver Items ===")
    for item in items:
        if isinstance(item, dict):
            name = item.get('product_type', '')
            if 'bạc' in name.lower() or 'silver' in name.lower():
                print(f"[{item.get('brand')}] {name}: Buy {item.get('buy_price',0):,.0f} | Sell {item.get('sell_price',0):,.0f}")
except Exception as e:
    import traceback
    traceback.print_exc()
