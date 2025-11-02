from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
import uvicorn

app = FastAPI()

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🚀 IntelliBot Pro Backend is running!"}

# ✅ Load data & model once
faq_df = pd.read_csv("faqs.csv")
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = model.encode(faq_df['Question'].tolist())

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    q = query.question.strip()
    if not q:
        return {"answer": "Please enter a valid question.", "confidence": 0.0}

    user_embedding = model.encode([q])
    similarities = cosine_similarity(user_embedding, faq_embeddings)[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    # 🔍 fuzzy score helps with typos
    fuzzy_scores = [fuzz.ratio(q.lower(), faq.lower()) / 100 for faq in faq_df['Question']]
    combined_score = (best_score * 0.7) + (max(fuzzy_scores) * 0.3)

    if combined_score < 0.4:
        return {"answer": "Hmm 🤔 I’m not sure about that yet, but I’ll keep learning!", "confidence": float(combined_score)}

    return {
        "answer": faq_df.iloc[best_index]['Answer'],
        "confidence": float(combined_score),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
