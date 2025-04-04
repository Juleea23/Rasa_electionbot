import pdfplumber
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 📌 Funktion zur Überschriften-Extraktion
def extract_headings_from_pdf(pdf_path):
    headings = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                for line in lines:
                    if line.strip() and line.isupper():  # Annahme: Überschriften sind großgeschrieben
                        headings.append(line.strip())
    return headings

# 📌 Funktion zur Text-Extraktion
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# 📌 Speichert die extrahierten Texte aus allen PDFs in einer Textdatei
def save_texts_from_pdfs(pdf_folder, output_file):
    all_text = []
    pdf_metadata = {}

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"🔍 Extrahiere Text aus: {pdf_file}")  # Debug-Ausgabe
            
            text = extract_text_from_pdf(pdf_path)
            headings = extract_headings_from_pdf(pdf_path)

            all_text.append(text)
            pdf_metadata[pdf_file] = headings  # Speichert die Überschriften je Datei

    # Speichert den gesamten Text in einer Datei
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    
    print(f"✅ Alle Texte gespeichert in: {output_file}")
    return pdf_metadata  # Gibt die Überschriften-Daten zurück

# 📌 Erstellt die Vektordatenbank
def create_vector_db(pdf_folder):
    """Erstellt eine FAISS-Datenbank mit Textabschnitten aus PDFs und speichert die Quelle (PDF-Dateiname)."""
    texts = []
    metadata = []

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"🔍 Extrahiere Text aus: {pdf_file}")  # Debug-Ausgabe
            
            # Extrahiere den gesamten Text der PDF
            full_text = extract_text_from_pdf(pdf_path)

            # Falls keine Inhalte gefunden wurden, gehe zur nächsten Datei
            if not full_text.strip():
                print(f"⚠️ Keine Texte in {pdf_file} gefunden. Datei wird übersprungen.")
                continue

            # Text in kleinere Abschnitte splitten
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            documents = text_splitter.create_documents([full_text])

            # Füge Metadaten für jeden Abschnitt hinzu (Quelle = PDF-Dateiname)
            for doc in documents:
                doc.metadata = {"source": pdf_file}
                texts.append(doc.page_content)
                metadata.append(doc.metadata)

    # Embeddings-Modell laden
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # FAISS-Datenbank erstellen mit Texten + Metadaten (Quelle)
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadata)
    vectorstore.save_local("vector_db")

    print("✅ Vektordatenbank wurde erfolgreich gespeichert!")

# 📌 Ordnerpfade setzen (Korrigiert)
pdf_folder = r"C:/Users/jukoe/Documents/Masterstudium/Kurse/Creative Prompting Techniques/rasa/pdfs"  
output_text_file = r"C:/Users/jukoe/Documents/Masterstudium/Kurse/Creative Prompting Techniques/rasa/wahlprogramme.txt"

# 1️⃣ Zuerst die Texte extrahieren & speichern
pdf_metadata = save_texts_from_pdfs(pdf_folder, output_text_file)

# 2️⃣ Danach die Vektordatenbank mit den PDFs erstellen
create_vector_db(pdf_folder)
