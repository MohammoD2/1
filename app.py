import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

app = FastAPI()

# ------------------------------
# Allow CORS for your frontend domain
# ------------------------------
origins = [
    "http://localhost:8000",  # for local testing
    "http://127.0.0.1:5500",  # optional
    "https://www.alloftech.site",  # production domain
    "https://alloftech.site",  # production domain (without www)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models & clients
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AllofTech AI Assistant API is running"}

@app.get("/health")
def health():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

@app.post("/chat")
def chat(req: ChatRequest):
    # 1️⃣ Embed user query
    query_vector = embedder.encode(req.message).tolist()

    # 2️⃣ Pinecone search
    result = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    context = "\n".join(
        match["metadata"]["text"]
        for match in result["matches"]
    )

    # 3️⃣ Prompt assembly
    prompt = f"""
You are AllofTech's official AI assistant.
Answer professionally and clearly using ONLY the context below.

Context:
{context}

User Question:
{req.message}
"""

    # 4️⃣ LLM call (OpenRouter)
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/devstral-2512:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=30
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get response from LLM: {str(e)}", "reply": None}
    except (KeyError, IndexError) as e:
        return {"error": f"Invalid response format from LLM: {str(e)}", "reply": None}

    return {"reply": answer}
