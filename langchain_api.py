from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
import pdfplumber
import os
print(f"🔍 Render-Umgebungsvariablen: {os.environ}")

app = Flask(__name__)

# Modell für semantische Suche
VECTOR_DB_PATH = "vector_db"
#MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
MODEL_NAME = "BAAI/bge-m3"
DATA_FOLDER = r"C:\Users\jukoe\Documents\Masterstudium\Kurse\Creative Prompting Techniques\rasa\pdfs"


def extract_text_from_pdf(pdf_path):
    """Extrahiert sauberen Text aus PDFs und entfernt Kopf-/Fußzeilen."""
    extracted_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                cleaned_lines = [
                    line.strip() for line in lines
                    if not is_header_or_footer(line)  # Kopf-/Fußzeilen herausfiltern
                ]
                extracted_text.append("\n".join(cleaned_lines))
    
    return "\n".join(extracted_text)

def is_header_or_footer(text):
    """Filtert typische Kopf-/Fußzeilen (z. B. Seitenzahlen, Kapitelüberschriften)."""
    header_patterns = ["Seite ", "Kapitel ", "Inhaltsverzeichnis", "Quelle:", "©"]
    return any(pattern.lower() in text.lower() for pattern in header_patterns)

def extract_text_with_ocr(pdf_path):
    """Falls eine PDF-Seite nur ein Bild ist, wendet OCR an."""
    text = extract_text_from_pdf(pdf_path)  # Erst normale Extraktion versuchen
    if not text.strip():  # Falls nichts erkannt wurde, OCR verwenden
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                img = page.to_image()
                text += pytesseract.image_to_string(img)
    return text



# Prüfe, ob der Index existiert – falls nicht, PDFs verarbeiten
if not os.path.exists(VECTOR_DB_PATH):
    process_pdfs()

# Lade den FAISS-Index
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = FAISS.load_local(VECTOR_DB_PATH, embeddings=embedding, allow_dangerous_deserialization=True)


@app.route("/query", methods=["POST"])
def query():
    if db is None:
        return jsonify({"error": "Datenbank wurde nicht gefunden!"}), 500

    data = request.json
    question = data.get("question", "")
    party = data.get("party", "").lower()
    topic = data.get("topic", "").lower()

    # Semantische Suche mit dem neuen Modell
    try:
        docs = db.similarity_search(question, k=5)
    except Exception as e:
        return jsonify({"error": "Fehler bei der Datenbanksuche", "details": str(e)}), 500

    filtered_docs = []
    for doc in docs:
        source = doc.metadata.get("source", "").lower()
        if party and party not in source:
            continue
        if topic and topic not in doc.page_content.lower():
            continue
        filtered_docs.append(doc.page_content)

    # Antwort generieren
    formatted_answer = generate_readable_response(filtered_docs)

    return jsonify({"response": formatted_answer})


def generate_readable_response(documents):
    """Erstellt eine verständliche Antwort aus den gefundenen Dokumenten."""
    if not documents:
        return "Ich konnte leider keine passenden Informationen in den Wahlprogrammen finden."

    # Fasse die wichtigsten Punkte zusammen
    summary = "\n".join(["- " + doc.replace("\n", " ") for doc in documents[:5]])  # Max. 5 Sätze
    
    response = (
        "Hier sind einige relevante Aussagen aus dem Wahlprogramm:\n\n"
        f"{summary}\n\n"
        "Falls du mehr Informationen benötigst, stelle gerne eine weitere Frage! 😊"
    )
    
    return response

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render nutzt standardmäßig Port 10000
    print(f"🚀 Starte Flask mit Gunicorn auf Port {port} ...")  # Debug-Print
    app.run(host="0.0.0.0", port=port)
