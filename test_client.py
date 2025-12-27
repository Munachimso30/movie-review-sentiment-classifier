import requests

url = "http://127.0.0.1:8000/predict"
payload = {"text": "This movie was great, i love it."}

r = requests.post(url, json=payload)
print(r.json())
