import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
}

try:
    print("Fetching webgia.com...")
    response = requests.get('https://webgia.com/gia-vang/', headers=headers, timeout=10)
    response.raise_for_status()
    
    with open('webgia.html', 'w', encoding='utf-8') as f:
        f.write(response.text)
        
    print(f"Success! Saved {len(response.text)} bytes to webgia.html")
    
except Exception as e:
    print(f"Error: {e}")
