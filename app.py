"""
Step 6 - Web app for the NYS Job Posting Assistant.
Run:  streamlit run app.py
"""
import datetime
import pathlib
import sqlite3

import pandas as pd
import streamlit as st

import rag

DATA = pathlib.Path("data")
st.set_page_config(page_title="NYS Job Posting Assistant", page_icon="🗽", layout="wide")


@st.cache_data
def load_jobs():
    with sqlite3.connect(DATA / "jobs.db") as con:
        return pd.read_sql("SELECT * FROM jobs", con)


jobs = load_jobs()
today = datetime.date.today().isoformat()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
agency = st.sidebar.selectbox("Agency", ["All"] + sorted(jobs["agency"].dropna().unique().tolist()))
county = st.sidebar.selectbox("County", ["All"] + sorted(jobs["county"].dropna().unique().tolist()))
open_only = st.sidebar.checkbox("Open postings only", value=True)
days = st.sidebar.slider("Deadline within (days)", 1, 120, 120)
keyword = st.sidebar.text_input("Keyword (title)")

view = jobs.copy()
if agency != "All":
    view = view[view["agency"] == agency]
if county != "All":
    view = view[view["county"] == county]
if open_only:
    view = view[view["deadline"].fillna("") >= today]
limit = (datetime.date.today() + datetime.timedelta(days=days)).isoformat()
view = view[view["deadline"].fillna("9999") <= limit]
if keyword:
    view = view[view["title"].str.contains(keyword, case=False, na=False)]

# ---------- Header ----------
st.title("🗽 NYS Job Posting Assistant")
st.caption(f"{len(jobs)} postings from StateJobsNY · Student project, not an official NYS tool · "
           "Always confirm details on the original posting.")

tab_chat, tab_table = st.tabs(["💬 Ask a question", "📋 Browse postings"])

# ---------- Chat ----------
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for role, text in st.session_state.history:
        st.chat_message(role).markdown(text)

    q = st.chat_input("e.g. Which jobs close this month? What are the qualifications for 228122?")
    if q:
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching postings and writing an answer..."):
                try:
                    answer, sources = rag.ask(
                        q,
                        county=None if county == "All" else county,
                        agency=None if agency == "All" else agency,
                        open_only=open_only,
                    )
                except Exception as e:
                    answer, sources = f"Something went wrong: {e}. Is the Ollama app running?", []
            links = "\n".join(f"- [{s['vacancy_id']} – {s['title']} ({s['agency']})]({s['url']})" for s in sources)
            full = answer + (f"\n\n**Sources**\n{links}" if links else "")
            st.markdown(full)
        st.session_state.history += [("user", q), ("assistant", full)]

# ---------- Table ----------
with tab_table:
    st.subheader(f"{len(view)} matching postings")
    cols = ["vacancy_id", "title", "agency", "county", "grade", "deadline", "salary_range", "url"]
    cols = [c for c in cols if c in view.columns]
    st.dataframe(
        view[cols].sort_values("deadline"),
                    width="stretch",
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("Link", display_text="Open")},
    )
    st.download_button("Download CSV", view[cols].to_csv(index=False), "nys_jobs.csv", "text/csv")