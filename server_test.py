import requests

url = "http://localhost:5001/query"
data = {"question": "Was sagt die SPD zur Klimapolitik?"}

response = requests.post(url, json=data)
print(response.json())