import requests
from bs4 import BeautifulSoup
import re

url = "https://giabac.vn/"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers, timeout=15)

print(f"Status: {response.status_code}")

soup = BeautifulSoup(response.content, 'html.parser')

# Find priceTable
table_div = soup.find('div', id='priceTable')
print(f"priceTable found: {table_div is not None}")

if table_div:
    table = table_div.find('table')
    print(f"Table found: {table is not None}")
    
    if table:
        rows = table.find_all('tr')
        print(f"Rows found: {len(rows)}")
        
        for i, row in enumerate(rows[:10]):
            cols = row.find_all('td')
            print(f"Row {i}: {len(cols)} cols -> {[c.get_text(strip=True)[:30] for c in cols]}")
