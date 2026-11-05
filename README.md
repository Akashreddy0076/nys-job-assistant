# 🗽 NYS Job Posting Assistant

An AI assistant that answers plain-English questions about current **New York State government job postings** and cites the exact posting for every answer.

> Student portfolio project – not an official New York State tool. Always confirm details on the original posting at [StateJobsNY](https://statejobs.ny.gov).

**Live demo:** https://nys-job-assistant.streamlit.app

## What it does
- 💬 **Ask questions** – "Which Data Analyst jobs are open in Albany?", "What is the salary for 227354?", "How do I apply for 228149?"
- 🔗 **Cited answers** – every fact links back to its vacancy ID and the original posting
- 📋 **Browse & filter** – by agency, county, deadline and keyword, with CSV download

## How it works (RAG pipeline)

```
StateJobsNY ──► collect.py ──► parse.py ──► index.py ──► rag.py ──► app.py
 (website)     scrape pages   clean fields   embeddings   retrieve +   Streamlit
                              → SQLite       → ChromaDB   LLM answer   web app
```

1. **Collect** – `requests` + `BeautifulSoup` download 1,678 public postings (cached, rate-limited).
2. **Parse & validate** – `pandas` extracts duties, qualifications, salary, contacts and reference codes into **SQLite**, with data quality checks (missing values, duplicates, open/closed status).
3. **Index** – each posting is split into section chunks (8,443 total) and embedded with **sentence-transformers** (`all-MiniLM-L6-v2`) into **ChromaDB** for semantic search.
4. **Retrieve & generate** – the most relevant chunks go to a **Llama** model (Groq online, Ollama locally) with instructions to answer only from those postings and cite vacancy IDs.
5. **App** – a **Streamlit** interface with chat, filters and a results table.

## Results
| Test | Result |
|---|---|
| Postings indexed | 1,678 |
| Text chunks embedded | 8,443 |
| Accuracy on vacancy-ID lookup questions (agency, salary) | **100% (18/18)** |
| Answers citing the correct vacancy ID | **100%** |

Evaluation script: `evaluate.py` (auto-generates questions from the database and checks answers).
Broad, open-ended questions have not been formally scored yet.

## Tech stack
Python · requests · BeautifulSoup · pandas · SQLite · sentence-transformers · ChromaDB · Llama 3 (Groq / Ollama) · Streamlit

## Run it locally
```bash
git clone https://github.com/Akashreddy0076/nys-job-assistant.git
cd nys-job-assistant
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt sentence-transformers

# Option A: free Groq API key (console.groq.com)
export GROQ_API_KEY="gsk_..."
# Option B: local Ollama – install from ollama.com, then:  ollama pull llama3.2

streamlit run app.py
```

To refresh the data: `python collect.py && python parse.py && python index.py`

## Responsible scraping
Pages are fetched slowly (1.5 s between requests), cached locally so each is downloaded once, and only public postings are used.

## Limitations & next steps
- Postings change daily – add a scheduled refresh (GitHub Actions)
- Add a "Do I qualify?" checker using each posting's degree-substitution rules
- Score open-ended questions and add hybrid keyword + semantic search

## Author
**Akash Reddy Rakasi** – M.S. Information Science (AI & Data Analytics), University at Albany, SUNY
[LinkedIn](https://linkedin.com/in/akash-reddy)