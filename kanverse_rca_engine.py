from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from sklearn.cluster import KMeans
import requests

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.json"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# GLOBAL CACHE
# -----------------------------
embedding_cache = {}
cluster_cache = None

# -----------------------------
# LOAD / INIT STORAGE
# -----------------------------
dimension = 384

if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "r") as f:
        metadata = json.load(f)

    documents = metadata.get("documents", [])
    rca_list = metadata.get("rca_list", [])
    ticket_ids = metadata.get("ticket_ids", [])

else:
    index = faiss.IndexFlatL2(dimension)
    documents = []
    rca_list = []
    ticket_ids = []

# -----------------------------
# REQUEST MODEL
# -----------------------------
class Ticket(BaseModel):
    ticket_id: str = ""
    subject: str = ""
    description: str = ""
    product: str = ""
    module: str = ""
    environment: str = ""
    issue_type: str = ""
    rca_description: str = ""
    resolution_notes: str = ""
    conversation: str = ""

# -----------------------------
# HELPERS
# -----------------------------
def build_text(ticket):
    return f"""
    Subject: {ticket.subject}
    Description: {ticket.description}
    Product: {ticket.product}
    Module: {ticket.module}
    Environment: {ticket.environment}
    Issue Type: {ticket.issue_type}
    RCA: {ticket.rca_description}
    Resolution: {ticket.resolution_notes}
    Conversation: {ticket.conversation}
    """

def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]

    emb = model.encode(text)
    embedding_cache[text] = emb
    return emb

def save_data():
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w") as f:
        json.dump({
            "documents": documents,
            "rca_list": rca_list,
            "ticket_ids": ticket_ids
        }, f)

# -----------------------------
# TRAIN (BATCH)
# -----------------------------
@app.post("/train_batch")
def train_batch(tickets: list[Ticket]):

    global cluster_cache
    cluster_cache = None  # invalidate cluster cache

    new_texts = []
    valid_tickets = []

    for ticket in tickets:

        if not ticket.rca_description or not ticket.issue_type:
            continue

        if ticket.ticket_id in ticket_ids:
            continue

        text = build_text(ticket)

        new_texts.append(text)
        valid_tickets.append(ticket)

    if not new_texts:
        return {"message": "No valid tickets"}

    # 🚀 BATCH EMBEDDING
    batch_embeddings = model.encode(new_texts)

    for i, emb in enumerate(batch_embeddings):

        index.add(np.array([emb]))

        documents.append(new_texts[i])
        rca_list.append(valid_tickets[i].rca_description)
        ticket_ids.append(valid_tickets[i].ticket_id)

    save_data()

    return {"stored": len(valid_tickets), "total": len(documents)}

# -----------------------------
# CLUSTERING (CACHED)
# -----------------------------
def cluster_rca():

    global cluster_cache

    if cluster_cache:
        return cluster_cache

    if len(documents) < 5:
        return {}

    embeddings = model.encode(documents)

    kmeans = KMeans(n_clusters=min(5, len(documents)), random_state=42)
    labels = kmeans.fit_predict(embeddings)

    cluster_map = {}

    for i, label in enumerate(labels):
        cluster_map.setdefault(label, []).append(rca_list[i])

    cluster_cache = cluster_map
    return cluster_map

# -----------------------------
# LLM REASONING (ADVANCED)
# -----------------------------
def llm_reasoning(new_ticket_text, top_matches):

    try:

        context = "\n\n".join([
            f"""
Ticket ID: {m['ticket_id']}
RCA: {m['rca']}
Similarity: {round(m['similarity'],2)}
Details: {m['doc']}
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

RETURN STRICT JSON:
{{
  "predicted_rca": "...",
  "rca_type": "...",
  "confidence": 0-100,
  "reasoning": "...",
  "resolution_steps": ["step1","step2","step3"]
}}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )

        return response.json().get("response", "")

    except:
        return "LLM unavailable"

# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(ticket: Ticket):

    if len(documents) == 0:
        return {"error": "No training data"}

    text = build_text(ticket)

    query_embedding = get_embedding(text)

    # -----------------------------
    # FAST SEARCH
    # -----------------------------
    k = min(10, len(documents))
    D, I = index.search(np.array([query_embedding]), k=k)

    matches = []

    for idx, dist in zip(I[0], D[0]):

        similarity = 1 - dist  # faster + better

        matches.append({
            "ticket_id": ticket_ids[idx],
            "rca": rca_list[idx],
            "doc": documents[idx],
            "similarity": similarity
        })

    # -----------------------------
    # FILTER STRONG MATCHES
    # -----------------------------
    matches = [m for m in matches if m["similarity"] > 0.2]

    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)

    top_matches = matches[:5]

    # -----------------------------
    # HYBRID SCORING
    # -----------------------------
    rca_scores = {}

    for m in top_matches:

        rca = m["rca"]

        if rca not in rca_scores:
            rca_scores[rca] = {"score": 0, "count": 0}

        rca_scores[rca]["score"] += (
            0.5 * m["similarity"] +
            0.5 * 1
        )

        rca_scores[rca]["count"] += 1

    best_rca = None
    max_score = 0

    for rca, data in rca_scores.items():
        if data["score"] > max_score:
            max_score = data["score"]
            best_rca = rca

    total_score = sum([v["score"] for v in rca_scores.values()])

    confidence = int((max_score / total_score) * 100) if total_score else 0

    # -----------------------------
    # CLUSTERING
    # -----------------------------
    clusters = cluster_rca()

    # -----------------------------
    # LLM REASONING
    # -----------------------------
    llm_output = llm_reasoning(text, top_matches)

    return {
        "predicted_rca": best_rca,
        "confidence": confidence,
        "top_matches": top_matches,
        "clusters": clusters,
        "llm_reasoning": llm_output
    }

@app.get("/")
def health():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    print("Starting server on port:", port)

    uvicorn.run("kanverse_rca_engine:app", host="0.0.0.0", port=port)
