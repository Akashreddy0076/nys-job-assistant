"""
Steps 4-5 - Split postings into chunks, create embeddings, and store them in ChromaDB.
Reads:  data/jobs.db
Saves:  data/chroma/  (vector database)
Run:    python index.py
"""
import pathlib
import sqlite3

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = pathlib.Path("data")
SECTIONS = {
    "duties": "DUTIES",
    "minimum_quals": "MINIMUM QUALIFICATIONS",
    "preferred_quals": "PREFERRED QUALIFICATIONS",
    "additional_comments": "ADDITIONAL COMMENTS",
    "how_to_apply": "HOW TO APPLY",
}


def clean(v):
    return "" if pd.isna(v) else str(v).strip()


def make_chunks(jobs):
    """One chunk per section; each starts with a header so it makes sense alone."""
    ids, docs, metas = [], [], []
    for _, j in jobs.iterrows():
        header = (f"[Vacancy {j['vacancy_id']} | {clean(j['title'])} | {clean(j['agency'])} | "
                  f"{clean(j['county'])} | Deadline {clean(j['deadline'])} | Salary {clean(j.get('salary_range'))}]")
        meta = {
            "vacancy_id": clean(j["vacancy_id"]),
            "title": clean(j["title"]),
            "agency": clean(j["agency"]),
            "county": clean(j["county"]),
            "deadline": clean(j["deadline"]),
            "grade": clean(j.get("grade")),
            "url": clean(j["url"]),
        }
        # Overview chunk (always present)
        overview = (f"{header}\nTITLE: {clean(j['title'])}\nAGENCY: {clean(j['agency'])}\n"
                    f"SALARY: {clean(j.get('salary_range'))}\nCONTACT: {clean(j.get('contact_email'))}\n"
                    f"REFERENCE: {clean(j.get('reference'))}")
        ids.append(f"{j['vacancy_id']}-overview"); docs.append(overview); metas.append({**meta, "section": "overview"})
        for col, name in SECTIONS.items():
            text = clean(j.get(col))
            if text:
                ids.append(f"{j['vacancy_id']}-{col}")
                docs.append(f"{header}\n{name}: {text[:4000]}")
                metas.append({**meta, "section": col})
    return ids, docs, metas


def main():
    with sqlite3.connect(DATA / "jobs.db") as con:
        jobs = pd.read_sql("SELECT * FROM jobs", con)
    ids, docs, metas = make_chunks(jobs)
    print(f"Made {len(docs)} chunks from {len(jobs)} postings")

    print("Loading embedding model (first time downloads ~90 MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(docs, show_progress_bar=True).tolist()

    db = chromadb.PersistentClient(path=str(DATA / "chroma"))
    col = db.get_or_create_collection("jobs")
    for i in range(0, len(ids), 5000):
        col.upsert(ids=ids[i:i+5000], documents=docs[i:i+5000],
                   embeddings=vectors[i:i+5000], metadatas=metas[i:i+5000])
    print(f"Stored {col.count()} chunks in data/chroma")

    # Quick test search
    q = "data analyst jobs that need Python or SQL"
    res = col.query(query_embeddings=model.encode([q]).tolist(), n_results=3)
    print(f"\nTest search: '{q}'")
    for m in res["metadatas"][0]:
        print(f"  {m['vacancy_id']} | {m['title']} | {m['agency']} | section: {m['section']}")


if __name__ == "__main__":
    main()