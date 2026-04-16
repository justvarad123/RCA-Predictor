from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import json
import os
from rank_bm25 import BM25Okapi
import requests

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
INDEX_FILE = "faiss_index.bin"
META_FILE = "metadata.json"

# -----------------------------
# MODELS (LAZY LOAD)
# -----------------------------
embed_model = None
cross_model = None

def get_embed_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return embed_model

# -----------------------------
# GLOBAL DATA
# -----------------------------
dimension = 384
documents = []
rca_list = []
rca_type_list = []
ticket_ids = []
tokenized_docs = []
bm25 = None

# -----------------------------
# LOAD DATA
# -----------------------------
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)

    with open(META_FILE, "r") as f:
        meta = json.load(f)

    documents = meta["documents"]
    rca_list = meta["rca_list"]
    rca_type_list = meta["rca_type_list"]
    ticket_ids = meta["ticket_ids"]

    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

else:
    index = faiss.IndexFlatIP(dimension)

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

def normalize(v):
    return v / np.linalg.norm(v)

def embed(text):
    model = get_embed_model()
    emb = model.encode(text)
    return normalize(emb)

def save():
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w") as f:
        json.dump({
            "documents": documents,
            "rca_list": rca_list,
            "rca_type_list": rca_type_list,
            "ticket_ids": ticket_ids
        }, f)

# -----------------------------
# TRAIN
# -----------------------------
@app.post("/train_batch")
def train_batch(tickets: list[Ticket]):
    global bm25, tokenized_docs

    texts = []
    valid = []

    for t in tickets:
        if not t.rca_description or not t.issue_type:
            continue
        if t.ticket_id in ticket_ids:
            continue

        txt = build_text(t)
        texts.append(txt)
        valid.append(t)

    if not texts:
        return {"message": "No valid tickets"}

    model = get_embed_model()
    embeddings = model.encode(texts)

    for i, emb in enumerate(embeddings):
        emb = normalize(emb)

        index.add(np.array([emb]))

        documents.append(texts[i])
        rca_list.append(valid[i].rca_description)
        rca_type_list.append(valid[i].rca_type)
        ticket_ids.append(valid[i].ticket_id)

    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    save()

    return {"stored": len(valid), "total": len(documents)}

# -----------------------------
# HYBRID SEARCH
# -----------------------------
def hybrid_search(query):

    query_emb = embed(query)

    # FAISS
    k = min(20, len(documents))
    D, I = index.search(np.array([query_emb]), k)

    vector_scores = {idx: float(score) for idx, score in zip(I[0], D[0])}

    # BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # COMBINE
    combined = {}
    for i in range(len(documents)):
        combined[i] = (
            0.6 * vector_scores.get(i, 0) +
            0.4 * bm25_scores[i]
        )

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked[:20]:
        results.append({
            "ticket_id": ticket_ids[idx],
            "rca": rca_list[idx],
            "rca_type": rca_type_list[idx],
            "doc": documents[idx],
            "similarity": score
        })

    return results

# -----------------------------
# CROSS ENCODER RERANK
# -----------------------------
# REMOVE get_cross_model() and the CrossEncoder import entirely

# REPLACE rerank() with this lightweight version:
def rerank(query, candidates):
    query_words = set(query.lower().split())
    
    for c in candidates:
        doc_words = set(c["doc"].lower().split())
        overlap = len(query_words & doc_words)
        c["rerank_score"] = c["similarity"] + (0.1 * overlap)
    
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

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

    if not documents:
        return {"error": "No training data"}

    text = build_text(ticket)

    results = hybrid_search(text)

    # 🔥 CROSS ENCODER
    reranked = rerank(text, results)

    top_matches = reranked[:5]

    # RCA voting
    scores = {}
    for m in top_matches:
        key = (m["rca"], m["rca_type"])
        scores[key] = scores.get(key, 0) + m["rerank_score"]

    best = max(scores, key=scores.get)

    total = sum(scores.values())
    confidence = int((scores[best] / total) * 100) if total else 0

    llm = llm_reasoning(text, top_matches)

    return {
        "predicted_rca": best[0],
        "rca_type": best[1],
        "confidence": confidence,
        "top_matches": top_matches,
        "llm_reasoning": llm
    }

@app.get("/")
def health():
    return {"status": "hybrid + rerank running"}
