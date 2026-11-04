"""
Step 5 - Ask questions about NYS job postings.
Finds the right postings, then an AI writes an answer that cites vacancy IDs.
- Online (Streamlit Cloud): uses Groq if GROQ_API_KEY is set (free, no card).
- On your Mac: uses Ollama (llama3.2) if no Groq key is set.
Run:  python rag.py
"""
import datetime
import os
import pathlib
import re

import chromadb
import requests

DATA = pathlib.Path("data")

# ---------- Which AI to use ----------
def _get_groq_key():
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None

GROQ_KEY = _get_groq_key()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

# ---------- Embedding model (same all-MiniLM-L6-v2 model either way) ----------
try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed(texts):
        return _st_model.encode(texts).tolist()
except ImportError:
    # Lighter version used online (no PyTorch needed)
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    _ef = DefaultEmbeddingFunction()
    def embed(texts):
        return [list(map(float, v)) for v in _ef(texts)]

# Unzip the search database the first time (GitHub stores it as chroma.zip)
import zipfile
if not (DATA / "chroma").exists() and (DATA / "chroma.zip").exists():
    with zipfile.ZipFile(DATA / "chroma.zip") as z:
        z.extractall(DATA)

_col = chromadb.PersistentClient(path=str(DATA / "chroma")).get_collection("jobs")

PROMPT = """You answer questions about New York State government job postings.
Use ONLY the postings below. If the answer is not in them, say you don't know.
Cite every fact with its vacancy ID in square brackets, like [228122].
Write dates as Month Day, Year. Be brief and clear.

POSTINGS:
{context}

QUESTION: {question}"""


def retrieve(question, k=6, county=None, agency=None, open_only=True):
    """Find the most relevant chunks, then apply simple filters."""
    ids = re.findall(r"\b\d{6}\b", question)
    if ids:
        res = _col.get(where={"vacancy_id": {"$in": ids}})
        return list(zip(res["documents"], res["metadatas"]))

    res = _col.query(query_embeddings=embed([question]), n_results=30)
    today = datetime.date.today().isoformat()
    hits = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        if open_only and meta.get("deadline") and meta["deadline"] < today:
            continue
        if county and county.lower() not in meta.get("county", "").lower():
            continue
        if agency and agency.lower() not in meta.get("agency", "").lower():
            continue
        hits.append((doc, meta))
    return hits[:k]


def _call_llm(prompt):
    if GROQ_KEY:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_KEY}"}, json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def ask(question, **filters):
    hits = retrieve(question, **filters)
    if not hits:
        return "I couldn't find any matching open postings.", []
    context = "\n\n---\n\n".join(doc for doc, _ in hits)
    answer = _call_llm(PROMPT.format(context=context[:20000], question=question))

    seen, sources = set(), []
    for _, m in hits:
        if m["vacancy_id"] not in seen:
            seen.add(m["vacancy_id"])
            sources.append(m)
    return answer, sources


if __name__ == "__main__":
    print(f"Using: {'Groq' if GROQ_KEY else 'Ollama'}")
    print("NYS Job Posting Assistant – type a question (or 'quit')\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        answer, sources = ask(q)
        print(f"\nAssistant: {answer}\n")
        for s in sources:
            print(f"  • {s['vacancy_id']} – {s['title']} ({s['agency']}) {s['url']}")
        print()