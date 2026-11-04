"""
Step 7 - Measure how accurate the assistant is.
Picks random postings, asks questions whose answers are already in the database,
and checks whether the AI's answer contains the right fact.
Saves: data/eval_results.csv
Run:   python evaluate.py
"""
import pathlib
import random
import re
import sqlite3
from datetime import datetime

import pandas as pd

import rag

DATA = pathlib.Path("data")
N_POSTINGS = 10          # 10 postings x 2 questions = 20 questions (~5-10 minutes)
random.seed(42)


def norm(s):
    return re.sub(r"[^a-z0-9$]", "", str(s).lower())


def check_agency(answer, agency):
    # Pass if the main words of the agency name appear in the answer
    words = [w for w in re.findall(r"[a-z]+", agency.lower()) if len(w) > 3 and w not in {"department", "office", "state"}]
    return bool(words) and all(w in answer.lower() for w in words[:2])


def check_deadline(answer, deadline):
    d = datetime.strptime(deadline, "%Y-%m-%d")
    options = [d.strftime("%B %-d"), d.strftime("%b %-d"), d.strftime("%m/%d"), d.strftime("%-m/%-d"), deadline]
    return any(o.lower() in answer.lower() for o in options)


def check_salary(answer, salary):
    nums = re.findall(r"\$?\s?([\d,]{4,})", salary)
    if not nums:
        return None  # no salary in the posting; skip
    first = nums[0].replace(",", "")
    return first in answer.replace(",", "")


def main():
    with sqlite3.connect(DATA / "jobs.db") as con:
        jobs = pd.read_sql("SELECT * FROM jobs", con)
    jobs = jobs.dropna(subset=["agency", "deadline"])
    sample = jobs.sample(N_POSTINGS, random_state=42)

    tests = []
    for _, j in sample.iterrows():
        vid = j["vacancy_id"]
        tests.append(("agency", f"Which agency posted vacancy {vid}?", j["agency"], vid))
        if str(j.get("salary_range") or "").strip():
            tests.append(("salary", f"What is the salary range for vacancy {vid}?", j["salary_range"], vid))
        else:
            tests.append(("deadline", f"What is the application deadline for vacancy {vid}?", j["deadline"], vid))

    rows = []
    for i, (kind, q, truth, vid) in enumerate(tests, 1):
        answer, sources = rag.ask(q, open_only=False)
        if kind == "agency":
            ok = check_agency(answer, truth)
        elif kind == "salary":
            ok = check_salary(answer, truth)
        else:
            ok = check_deadline(answer, truth)
        cited = vid in answer
        rows.append({"question": q, "type": kind, "correct_answer": truth,
                     "ai_answer": answer.replace("\n", " ")[:300], "correct": ok, "cited_vacancy_id": cited})
        mark = "OK " if ok else ("SKIP" if ok is None else "WRONG")
        print(f"[{i}/{len(tests)}] {mark} | {q}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "eval_results.csv", index=False)
    scored = df[df["correct"].notna()]
    acc = 100 * scored["correct"].astype(bool).mean()
    cite = 100 * df["cited_vacancy_id"].mean()
    print("\n===== RESULTS =====")
    print(f"Questions scored:   {len(scored)}")
    print(f"Answer accuracy:    {acc:.0f}%")
    print(f"Cited the vacancy:  {cite:.0f}%")
    for kind in scored["type"].unique():
        part = scored[scored["type"] == kind]
        print(f"  {kind:9s} {100 * part['correct'].astype(bool).mean():.0f}%  ({len(part)} questions)")
    print("\nDetails saved to data/eval_results.csv (open in Excel to review wrong answers)")


if __name__ == "__main__":
    main()