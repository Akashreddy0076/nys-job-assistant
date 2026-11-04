"""
Step 5 - Ask questions about NYS job postings. Finds the right postings, then an AI
(Llama via Ollama, free and local) writes an answer that cites vacancy IDs.
Needs:  data/chroma (from index.py) and Ollama running with:  ollama pull llama3.2
Run:    python rag.py
"""
import datetime
import pathlib
import re

import chromadb
import requests
from sentence_transformers import SentenceTransformer

DATA = pathlib.Path("data")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
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
    # If the question names a vacancy ID, fetch that posting directly.
    ids = re.findall(r"\b\d{6}\b", question)
    if ids:
        res = _col.get(where={"vacancy_id": {"$in": ids}})
        return list(zip(res["documents"], res["metadatas"]))

    res = _col.query(query_embeddings=_embedder.encode([question]).tolist(), n_results=30)
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


def ask(question, **filters):
    hits = retrieve(question, **filters)
    if not hits:
        return "I couldn't find any matching open postings.", []
    context = "\n\n---\n\n".join(doc for doc, _ in hits)
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(context=context, question=question)}],
        "stream": False,
        "options": {"temperature": 0},
    }, timeout=300)
    r.raise_for_status()
    answer = r.json()["message"]["content"]

    # Unique list of sources for links
    seen, sources = set(), []
    for _, m in hits:
        if m["vacancy_id"] not in seen:
            seen.add(m["vacancy_id"])
            sources.append(m)
    return answer, sources


if __name__ == "__main__":
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