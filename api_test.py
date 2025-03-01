import requests

API_URL = "http://127.0.0.1:5001/query"  # Passe die URL an, falls sie anders ist

test_payload = {
    "question": "Was sagt die SPD zur Wirtschaft?",
    "party": "spd"
}

print("⚡ Sende API-Anfrage...")
try:
    response = requests.post(API_URL, json=test_payload)
    print(f"🔍 Status Code: {response.status_code}")

    if response.status_code == 200:
        try:
            response_data = response.json()
            print(f"✅ API-Antwort (JSON): {response_data}")
        except Exception as e:
            print(f"⚠️ Fehler beim Parsen der Antwort: {e}")
    else:
        print(f"❌ Fehlerhafte Antwort von der API: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"❌ API-Aufruf fehlgeschlagen: {e}")
