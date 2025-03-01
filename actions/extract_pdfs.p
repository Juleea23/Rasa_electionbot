import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def save_texts_from_pdfs(pdf_folder, output_file):
    all_text = ""
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"🔍 Extrahiere Text aus: {pdf_file}")  # Debug-Ausgabe
            all_text += extract_text_from_pdf(pdf_path) + "\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(all_text)
    print(f"✅ Alle Texte gespeichert in: {output_file}")

# Starte das Skript mit dem Ordner "pdfs"
save_texts_from_pdfs("pdfs", "wahlprogramme.txt")
