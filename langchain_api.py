from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient, login
import traceback
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import spacy  # Neu: Für automatische Wortstamm-Erkennung
from spacy.lang.de.examples import sentences
print("CUDA verfügbar:", tf.test.is_built_with_cuda())
print("GPU erkannt:", tf.config.list_physical_devices('GPU'))

# 🔐 API-Login
HUGGINGFACEHUB_API_TOKEN = SECRETCODE
login(HUGGINGFACEHUB_API_TOKEN)

app = Flask(__name__)

# 📌 Modelle für Embeddings und semantische Suche
VECTOR_DB_PATH = "vector_db"
MODEL_NAME = "BAAI/bge-m3"
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = FAISS.load_local(VECTOR_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)
similarity_model = SentenceTransformer(MODEL_NAME)

# 📖 NLP-Modell für Lemmatisierung (automatische Wortstamm-Erkennung)
nlp = spacy.load("de_core_news_sm")  # Kleines, schnelles deutsches Modell

# 💡 KI-Modell für Generierung
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1", token=HUGGINGFACEHUB_API_TOKEN)

# 🔢 Ähnlichkeitsschwelle (je niedriger, desto mehr Treffer)
SIMILARITY_THRESHOLD = 0.4  

@app.route("/query", methods=["POST"])
def query():
    try:
        if db is None:
            return jsonify({"error": "Datenbank wurde nicht gefunden!"}), 500

        data = request.json
        question = data.get("question", "").strip().lower()
        topic = data.get("topic", "").strip().lower()
        party = data.get("party", "").strip().lower()

        # 📌 **Automatische Lemmatisierung des Topics**
        topic_lemma = " ".join([token.lemma_ for token in nlp(topic)])

        # 🔎 1️⃣ Semantische Suche mit FAISS
        docs = db.similarity_search(question, k=10)

        # 🛠 Debugging: Zeigt alle extrahierten Textstellen
        extracted_texts = [(doc.page_content, doc.metadata.get("source", "Unbekannt")) for doc in docs]
        print("\n🔎 DEBUG: Rohdaten aus FAISS (vor Filterung):")
        for i, (text, source) in enumerate(extracted_texts):
            print(f"{i+1}. [Quelle: {source}] {text[:300]}...")

        # 🎯 2️⃣ Partei- und Themenfilterung mit semantischer Ähnlichkeit
        def text_passes_filters(text, source):
            source_lower = source.lower()

            # ✅ Partei-Filter
            if party and party not in source_lower:
                return False

            # ✅ Exakte Wortsuche als Backup
            if topic in text.lower() or topic_lemma in text.lower():
                print(f"✅ Direkte Übereinstimmung gefunden in: {text[:100]}...")
                return True

            # ✅ Semantische Ähnlichkeitsprüfung
            text_embedding = similarity_model.encode(text, convert_to_tensor=True)
            topic_embedding = similarity_model.encode(topic_lemma, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(topic_embedding, text_embedding).item()

            print(f"🔍 DEBUG: Ähnlichkeit zwischen '{topic}' (Lemma: {topic_lemma}) und Text = {similarity_score}")

            return similarity_score >= SIMILARITY_THRESHOLD  # Nur relevante Texte behalten

        # ➡️ Filterung anwenden
        filtered_texts = [text for text, source in extracted_texts if text_passes_filters(text, source)]

        # 🛠 Debugging: Zeigt gefilterte Textstellen
        print("\n🔍 DEBUG: Gefilterte Textstellen nach Partei & semantischer Ähnlichkeit:")
        for i, text in enumerate(filtered_texts):
            print(f"{i+1}. {text[:300]}...")

        # ❌ Falls kein relevanter Inhalt gefunden → Fehler ausgeben
        if not filtered_texts:
            return jsonify({"response": f"Ich konnte keine Wahlprogramminformationen zur Partei '{party}' und Thema '{topic}' finden."})

        # 📌 3️⃣ KI erstellt eine Antwort
        context = "\n\n".join([f"- {text}" for text in filtered_texts[:5]])  # Maximal 5 relevante Treffer nutzen

        # 🔍 Debug: Prompt an KI-Modell
        print("\n🛠 DEBUG: Prompt an KI-Modell:")
        prompt_text = f"""
Hier sind relevante Aussagen aus dem deutschen Wahlprogramm der Partei '{party}' zum Thema '{topic}':\n\n{context}\n\n
Bitte fasse nur die Aussagen zusammen, die inhaltlich  mit '{topic}' zu tun haben und beantworte damit die Frage, was '{party}' zum Thema '{topic}' sagt.
Beachte dabei, dass die Aussagen aus unterschiedlichen Textabschnitten kommen können und inhaltlich unterschiedliche Themen ansprechen können. Stelle sicher, dass du diese Themen dann auch getrennt voneinander darstellst. 
Erwähne nichts, was nicht in den Quellen steht.
Versuche, alle Schlüsselbegriffe ebenfalls zu verwenden. 
Nutze klare und sachliche Sprache, vermeide Wiederholungen und stelle die Inhalte objektiv dar.
Formuliere deine Antwort auf Deutsch und achte dabei auf grammatikalische Richtigkeit. 
"""
        print(prompt_text)

        # 🧠 KI-Aufruf
        response_obj = client.chat_completion(messages=[{"role": "user", "content": prompt_text}])

        # 🔍 Debug: Zeigt die Roh-Antwort der KI
        print("\n🛠 DEBUG: Antwort von Hugging Face API:")
        print(response_obj)

        # ✅ Überprüfung der API-Antwort
        if isinstance(response_obj, dict) and "choices" in response_obj:
            try:
                response_text = response_obj["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                response_text = "Fehler: Unerwartete API-Antwortstruktur."
        else:
            response_text = "Fehler: Keine gültige Antwort vom KI-Modell erhalten."

        # 📩 JSON-Antwort für Rasa
        response_json = {"response": response_text.strip()}
        print("\n📩 DEBUG: JSON-Antwort für Rasa:", response_json)

        return jsonify(response_json)

    except Exception as e:
        error_details = traceback.format_exc()
        print(error_details)
        return jsonify({"error": "Interner Serverfehler", "details": error_details}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
