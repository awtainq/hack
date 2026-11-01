
import requests

url = "https://web.archive.org/cdx/search/cdx"

params = {
    "url": "priem.unecon.ru/stat/daily_zayav/index.php",
    "output": "json",
    "from": "2024",
    "to": "2024",
    "filter": "statuscode:200",
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

rows = response.json()

for row in rows[1:]:
    original = row[2]
    timestamp = row[1]
    print(
        timestamp,
        f"https://web.archive.org/web/{timestamp}/{original}"
    )
