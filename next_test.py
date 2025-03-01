import requests

# API-Endpunkt
url = "http://localhost:5001/query"

# 🔹 Test 1: Bekannte Partei ohne Topic
data1 = {
    "question": "Was sagt die SPD zur Wirtschaft?",
    "party": "spd"
}

# 🔹 Test 2: Nicht vorhandene Partei
data2 = {
    "question": "Was sagt die Piratenpartei zum Datenschutz?",
    "party": "piratenpartei"
}

# 🔹 Test 3: Existierende Partei mit nicht existierendem Thema
data3 = {
    "question": "Was sagt die CDU zu Quantencomputern?",
    "party": "cdu",
    "topic": "quantencomputer"
}

# 🔹 Test 4: Partei und Topic (sollte ein Ergebnis liefern)
data4 = {
    "question": "Was sagt die CDU zum Klimaschutz?",
    "party": "cdu",
    "topic": "Klimaschutz"
}

# Funktion zum Senden der Anfrage
def send_request(data):
    response = requests.post(url, json=data)
    print(f"\n🔍 Anfrage: {data}")
    print(f"📩 Antwort: {response}")

print(f"🔍 API Rohantwort: {response.status_code}, {response.text}")

# Tests ausführen
send_request(data1)
send_request(data2)
send_request(data3)
send_request(data4)
