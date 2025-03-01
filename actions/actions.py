from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

def query_langchain(question, party=None, topic=None):
    """Sendet eine Anfrage an den LangChain-Server und verarbeitet die Antwort."""
    url = "https://langchain-electionbot.onrender.com"  # Flask-Server mit LangChain läuft hier
    
    payload = {"question": question}
    if party:
        payload["party"] = party
    if topic:
        payload["topic"] = topic

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Fehler abfangen

        data = response.json()

        if "response" not in data:
            print(f"⚠️ WARNUNG: Unerwartete API-Antwortstruktur: {data}")
            return {"response": "Ich konnte keine Wahlprogramminformationen finden."}

        return data

    except requests.exceptions.RequestException as e:
        print(f"❌ API-Fehler: {e}")
        return {"response": f"Es gab einen Fehler bei der Anfrage: {e}"}

class ActionRetrieveInfo(Action):
    def name(self):
        return "action_retrieve_info"

    def run(self, dispatcher, tracker, domain):
        """Ruft allgemeine Informationen über den LangChain-Server ab."""
        user_question = tracker.latest_message.get("text")

        data = query_langchain(user_question)

        answer = data.get("response", "Ich konnte keine passende Antwort finden.")

        print(f"🔍 DEBUG: Antwort von LangChain für '{user_question}': {answer}")  # Debug-Print
        dispatcher.utter_message(text=answer)
        return []

class ActionRetrieveElectionInfo(Action):
    def name(self):
        return "action_retrieve_election_info"

    def run(self, dispatcher, tracker, domain):
        """Ruft wahlprogrammspezifische Informationen aus der FAISS-Datenbank ab."""
        party = next(tracker.get_latest_entity_values("party"), None)
        topic = next(tracker.get_latest_entity_values("topic"), None)

        print(f"🔎 DEBUG: Erkannte Partei: {party}, Erkannter Topic: {topic}")

        if not party or not topic:
            dispatcher.utter_message(text="Ich konnte keine Informationen zur Partei oder zum Thema finden. Bitte stelle deine Frage präziser.")
            return []

        question = f"Was sagt die {party} zum Thema {topic}?"
        data = query_langchain(question, party, topic)

        print(f"🔍 DEBUG: Antwort von query_langchain(): {data}")  # Debug-Print

        election_info = data.get("response", "Ich konnte keine Wahlprogramminformationen finden.")

        print(f"📩 DEBUG: Antwort aus LangChain API: {election_info}")  # Debug-Print
        dispatcher.utter_message(text=election_info)

        return []
