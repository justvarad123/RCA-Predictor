from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import uuid
import os
import requests

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION = "tickets"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# -----------------------------
# INIT COLLECTION
# -----------------------------
def init_collection():
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

init_collection()

# -----------------------------
# REQUEST MODEL
# -----------------------------
class Ticket(BaseModel):
    ticket_id: str = ""
    subject: str = ""
    description: str = ""
    issue_type: str = ""
    rca_description: str = ""
    rca_type: str = ""
    conversation: str = ""

# -----------------------------
# HELPERS
# -----------------------------
def build_text(t):
    return f"{t.subject} {t.description} {t.issue_type} {t.conversation}"

def embed(text):
    return model.encode(text).tolist()

# -----------------------------
# TRAIN → STORE IN QDRANT
# -----------------------------
@app.post("/train_batch")
def train_batch(tickets: list[Ticket]):

    points = []

    for t in tickets:
        if not t.rca_description:
            continue

        vector = embed(build_text(t))

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "ticket_id": t.ticket_id,
                    "rca": t.rca_description,
                    "rca_type": t.rca_type,
                    "text": build_text(t)
                }
            )
        )

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    return {"stored": len(points)}

# -----------------------------
# SEARCH
# -----------------------------
def search(query):
    vector = embed(query)

    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=20
    )

    return [
        {
            "ticket_id": r.payload["ticket_id"],
            "rca": r.payload["rca"],
            "rca_type": r.payload["rca_type"],
            "doc": r.payload["text"],
            "score": r.score
        }
        for r in results
    ]

# -----------------------------
# SIMPLE RERANK
# -----------------------------
def rerank(query, results):
    q_words = set(query.lower().split())

    for r in results:
        d_words = set(r["doc"].lower().split())
        overlap = len(q_words & d_words)
        r["rerank"] = r["score"] + (0.1 * overlap)

    return sorted(results, key=lambda x: x["rerank"], reverse=True)

# -----------------------------
# LLM
# -----------------------------
def llm_reasoning(new_ticket_text, top_matches):

    try:
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

        if not GROQ_API_KEY:
            return "No GROQ API key found"

        context = "\n\n".join([
            f"""
Ticket ID: {m['ticket_id']}
RCA: {m['rca']}
Type: {m['rca_type']}
Similarity: {round(m['similarity'],2)}
"""
            for m in top_matches
        ])

        prompt = f"""
You are an expert L3 support engineer.

NEW TICKET:
{new_ticket_text}

SIMILAR TICKETS:
{context}

INSTRUCTIONS:
- Find strongest pattern
- Ignore weak matches
- Combine RCA + context + conversation
- Think step-by-step

TASK:
- Predict most accurate root cause
- Give clear RCA type
- Provide exact resolution steps
- Be confident and precise

RETURN JSON:
{{
  "predicted_rca": "...",
  "rca_type": "...",
  "confidence": 0-100,
  "reasoning": "...",
  "resolution_steps": ["step1","step2","step3"]
}}
"""

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",  # 🔥 best model
                "messages": [
                    {"role": "system", "content": "You are a senior support engineer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            },
            timeout=10
        )

        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"LLM Error: {str(e)}"

# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(ticket: Ticket):

    query = build_text(ticket)

    results = search(query)

    if not results:
        return {"error": "No data in vector DB"}

    reranked = rerank(query, results)
    top = reranked[:5]

    scores = {}
    for r in top:
        key = (r["rca"], r["rca_type"])
        scores[key] = scores.get(key, 0) + r["rerank"]

    best = max(scores, key=scores.get)

    total = sum(scores.values())
    confidence = int((scores[best] / total) * 100)

    llm = llm_reasoning(query, top)

    return {
        "predicted_rca": best[0],
        "rca_type": best[1],
        "confidence": confidence,
        "top_matches": top,
        "llm_reasoning": llm
    }

@app.get("/")
def health():
    return {"status": "Qdrant running 🚀"}
