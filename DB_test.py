from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Lade die Vektordatenbank
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vector_db", embeddings=embedding_model, allow_dangerous_deserialization=True)

# Testanfrage stellen
query = "Welche Pläne hat die Partei SPD zur Bildungspolitik?"
retriever = vectorstore.as_retriever()
docs = retriever.get_relevant_documents(query)

# Ergebnisse ausgeben
for doc in docs:
    print(f"📄 Gefundener Abschnitt: {doc.page_content}")
    print(f"🔗 Quelle: {doc.metadata.get('source', 'Unbekannt')}\n")
