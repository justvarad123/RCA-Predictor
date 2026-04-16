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
model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return model

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
    rca_type_list = metadata.get("rca_type_list", [])
    ticket_ids = metadata.get("ticket_ids", [])

else:
    index = faiss.IndexFlatIP(dimension)  # 🔥 cosine similarity
    documents = []
    rca_list = []
    rca_type_list = []
    ticket_ids = []

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
def build_text(ticket):
    return f"""
    Subject: {ticket.subject}
    Description: {ticket.description}
    Issue Type: {ticket.issue_type}
    RCA: {ticket.rca_description}
    RCA Type: {ticket.rca_type}
    Conversation: {ticket.conversation}
    """

def normalize(vec):
    return vec / np.linalg.norm(vec)

def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]

    model = get_model()
    emb = model.encode(text)
    emb = normalize(emb)

    embedding_cache[text] = emb
    return emb

def save_data():
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w") as f:
        json.dump({
            "documents": documents,
            "rca_list": rca_list,
            "rca_type_list": rca_type_list,
            "ticket_ids": ticket_ids
        }, f)

# -----------------------------
# TRAIN (BATCH)
# -----------------------------
@app.post("/train_batch")
def train_batch(tickets: list[Ticket]):

    global cluster_cache
    cluster_cache = None

    model = get_model()

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
    embeddings = model.encode(new_texts)

    for i, emb in enumerate(embeddings):
        emb = normalize(emb)

        index.add(np.array([emb]))

        documents.append(new_texts[i])
        rca_list.append(valid_tickets[i].rca_description)
        rca_type_list.append(valid_tickets[i].rca_type)
        ticket_ids.append(valid_tickets[i].ticket_id)

    save_data()

    return {"stored": len(valid_tickets), "total": len(documents)}

# -----------------------------
# CLUSTERING (CACHED)
# -----------------------------
# def cluster_rca():

    global cluster_cache

    if cluster_cache:
        return cluster_cache

    if len(documents) < 5:
        return {}

    model = get_model()
    embeddings = model.encode(documents)

    kmeans = KMeans(n_clusters=min(5, len(documents)), random_state=42)
    labels = kmeans.fit_predict(embeddings)

    cluster_map = {}

    for i, label in enumerate(labels):
        cluster_map.setdefault(int(label), []).append(rca_list[i])

    cluster_cache = cluster_map
    return cluster_map

# -----------------------------
# LLM REASONING (ADVANCED)
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

    if len(documents) == 0:
        return {"error": "No training data"}

    text = build_text(ticket)
    query_embedding = get_embedding(text)

    k = min(10, len(documents))
    D, I = index.search(np.array([query_embedding]), k=k)

    matches = []

    for idx, score in zip(I[0], D[0]):
        similarity = float(score)  # ✅ FIXED

        matches.append({
            "ticket_id": ticket_ids[idx],
            "rca": rca_list[idx],
            "rca_type": rca_type_list[idx],
            "doc": documents[idx],
            "similarity": similarity
        })

    # ✅ SORT DIRECTLY (no 1 - dist)
    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)

    top_matches = matches[:5]

    if not top_matches:
        return {"error": "No matches found"}

    # -----------------------------
    # HYBRID SCORING (RCA + TYPE)
    # -----------------------------
    rca_scores = {}

    for m in top_matches:
        key = (m["rca"], m["rca_type"])

        if key not in rca_scores:
            rca_scores[key] = 0

        rca_scores[key] += m["similarity"]

    # ✅ pick best RCA + TYPE together
    best_pair = max(rca_scores, key=rca_scores.get)

    best_rca, best_rca_type = best_pair
    best_score = rca_scores[best_pair]

    total_score = sum(rca_scores.values())
    confidence = int((best_score / total_score) * 100) if total_score else 0

    # clusters = cluster_rca()
    llm_output = llm_reasoning(text, top_matches)

    return {
        "predicted_rca": best_rca,
        "rca_type": best_rca_type,  # ✅ FIXED
        "confidence": confidence,
        "top_matches": top_matches,
        # "clusters": clusters,
        "llm_reasoning": llm_output
    }

@app.get("/")
def health():
    return {"status": "running"}
