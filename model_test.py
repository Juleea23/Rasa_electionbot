from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Lade das neue Modell
model_name = "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
embedding = HuggingFaceEmbeddings(model_name=model_name)

# Lade den FAISS-Index
db = FAISS.load_local("vector_db", embeddings=embedding, allow_dangerous_deserialization=True)

# Testfrage
question = "Was sagt die SPD zur Klimapolitik?"
docs = db.similarity_search(question, k=5)

# Zeige die Ergebnisse
for doc in docs:
    print(doc.page_content)
