import requests
from bs4 import BeautifulSoup

url = "https://giabac.vn/"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')

# Find text containing "Bạc"
print(f"Status Code: {response.status_code}")
print(f"Content Length: {len(response.content)}")
print("First 500 chars:")
print(response.text[:500])

elements = soup.find_all(string=lambda text: "bạc" in text.lower() if text else False)

# Save to file
with open("giabac_vn_source.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Saved to giabac_vn_source.html") 
