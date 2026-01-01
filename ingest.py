import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

# Init
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Load text
with open("data/Alloftech_info.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Simple chunking
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(text)

# Embed & upsert
vectors = []
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()
    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {"text": chunk}
    })

index.upsert(vectors=vectors)

print("✅ Ingestion completed")
