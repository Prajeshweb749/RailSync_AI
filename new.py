import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
resp = requests.get("https://etrain.info/in?TRAIN=12180", headers=HEADERS)
soup = BeautifulSoup(resp.text, "html.parser")
table = max(soup.find_all("table"), key=lambda t: len(t.find_all("tr")))

for row in table.find_all("tr")[:5]:  # First 5 rows
    tds = [td.get_text(strip=True) for td in row.find_all("td")]
    if len(tds) >= 5:
        print(tds)
