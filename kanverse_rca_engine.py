from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
import json
import os

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

MEMORY_FILE = "rca_memory.json"

# load memory
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE) as f:
        rca_memory = json.load(f)
else:
    rca_memory = []

class Ticket(BaseModel):

    id:int
    subject:str=""
    description:str=""
    description_text:str=""
    product:str=""
    module:str=""
    environment:str=""
    issue_type:str=""
    rca_description:str=""
    resolution_notes:str=""
    conversations:list[str]=[]

# ------------------------------------------------
# Utility functions
# ------------------------------------------------

def build_ticket_context(ticket):

    text = f"""
    subject: {ticket.subject}
    description: {ticket.description}
    description_text: {ticket.description_text}
    product: {ticket.product}
    module: {ticket.module}
    environment: {ticket.environment}
    issue_type: {ticket.issue_type}
    rca: {ticket.rca_description}
    resolution: {ticket.resolution_notes}
    conversations: {" ".join(ticket.conversations)}
    """

    return text.lower()

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ------------------------------------------------
# Conversation reasoning
# ------------------------------------------------

def extract_troubleshooting_steps(conversations):

    steps=[]

    for msg in conversations:

        msg=msg.lower()

        if "please" in msg or "check" in msg or "try" in msg:

            steps.append(msg)

    return steps[:5]

# ------------------------------------------------
# RCA Pattern extraction
# ------------------------------------------------

def extract_rca_pattern(rca_text):

    words = rca_text.lower().split()

    important=[w for w in words if len(w)>4]

    pattern=" ".join(important[:5])

    return pattern

# ------------------------------------------------
# Save memory
# ------------------------------------------------

def save_memory():

    with open(MEMORY_FILE,"w") as f:
        json.dump(rca_memory,f)

# ------------------------------------------------
# Learning endpoint
# ------------------------------------------------

@app.post("/learn")

def learn(ticket:Ticket):

    if not ticket.rca_description:

        return {"status":"skipped"}

    context = build_ticket_context(ticket)

    embedding = model.encode(context).tolist()

    pattern = extract_rca_pattern(ticket.rca_description)

    steps = extract_troubleshooting_steps(ticket.conversations)

    memory_record={

        "ticket_id":ticket.id,
        "embedding":embedding,
        "pattern":pattern,
        "rca":ticket.rca_description,
        "resolution":ticket.resolution_notes,
        "steps":steps,
        "product":ticket.product,
        "module":ticket.module,
        "environment":ticket.environment

    }

    rca_memory.append(memory_record)

    save_memory()

    return {"status":"learned","memory_size":len(rca_memory)}

# ------------------------------------------------
# Ticket clustering
# ------------------------------------------------

def cluster_tickets():

    if len(rca_memory)<3:
        return None

    vectors=[m["embedding"] for m in rca_memory]

    clustering = DBSCAN(eps=0.4,min_samples=2).fit(vectors)

    return clustering.labels_

# ------------------------------------------------
# Prediction endpoint
# ------------------------------------------------

@app.post("/predict")

def predict(ticket:Ticket):

    context = build_ticket_context(ticket)

    query_vec = model.encode(context)

    matches=[]

    for memory in rca_memory:

        score=cosine_similarity(query_vec,memory["embedding"])

        matches.append({

            "score":float(score),
            "memory":memory

        })

    matches=sorted(matches,key=lambda x:x["score"],reverse=True)

    best=matches[:5]

    rca_scores={}
    recommended_steps=[]

    for m in best:

        rca=m["memory"]["rca"]

        rca_scores[rca]=rca_scores.get(rca,0)+m["score"]

        if m["memory"]["resolution"]:
            recommended_steps.append(m["memory"]["resolution"])

        recommended_steps+=m["memory"]["steps"]

    predicted_rca=None
    confidence=0

    if rca_scores:

        predicted_rca=max(rca_scores,key=rca_scores.get)
        confidence=rca_scores[predicted_rca]

    clusters=cluster_tickets()

    return {

        "predicted_rca":predicted_rca,
        "confidence":round(confidence*100,2),

        "recommended_steps":list(set(recommended_steps))[:5],

        "similar_tickets":[
            {
                "ticket_id":m["memory"]["ticket_id"],
                "similarity":round(m["score"],3)
            }
            for m in best
        ],

        "cluster_info": clusters.tolist() if clusters is not None else None
    }
