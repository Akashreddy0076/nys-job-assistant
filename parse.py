"""
Step 3 - Pull clean fields out of each saved posting and store them in a database.
Reads:  data/raw/*.html  and  data/vacancies.csv
Saves:  data/jobs.db (SQLite, table "jobs")  and  data/jobs_clean.csv
Run:    python parse.py
"""
import pathlib
import re
import sqlite3

import pandas as pd
from bs4 import BeautifulSoup

DATA = pathlib.Path("data")
RAW = DATA / "raw"

# Labels as they appear on StateJobsNY detail pages -> our column names.
LABELS = {
    "Vacancy ID": "vacancy_id_page",
    "Date Posted": "date_posted",
    "Applications Due": "applications_due",
    "Title": "title_page",
    "Occupational Category": "category",
    "Salary Grade": "salary_grade",
    "Bargaining Unit": "bargaining_unit",
    "Salary Range": "salary_range",
    "Employment Type": "employment_type",
    "Appointment Type": "appointment_type",
    "Jurisdictional Class": "jurisdictional_class",
    "Travel Percentage": "travel_percentage",
    "Workweek": "workweek",
    "Hours Per Week": "hours_per_week",
    "Telecommuting allowed?": "telecommuting",
    "County": "county_page",
    "City": "city",
    "Duties Description": "duties",
    "Minimum Qualifications": "minimum_quals",
    "Preferred Qualifications": "preferred_quals",
    "Additional Comments": "additional_comments",
    "Name": "contact_name",
    "Telephone": "contact_phone",
    "Email Address": "contact_email",
    "Notes on Applying": "how_to_apply",
    "Address": "_skip",
    "Street": "street",
    "State": "_skip",
    "Zip Code": "zip_code",
    "Fax": "contact_fax",
}


def page_lines(html):
    """Turn a page into a list of non-empty text lines."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text("\n")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def parse_page(html):
    """Read text after each known label until the next known label."""
    lines = page_lines(html)
    found, current = {}, None
    for ln in lines:
        label = next((lab for lab in LABELS if ln == lab or ln.startswith(lab + ":")), None)
        if label:
            current = LABELS[label]
            rest = ln[len(label):].lstrip(": ").strip()
            found.setdefault(current, [])
            if rest:
                found[current].append(rest)
        elif current:
            found[current].append(ln)
    return {k: " ".join(v).strip() for k, v in found.items()}


def find_reference(text):
    """Find a reference code like EW/DA3/78144 in the how-to-apply text."""
    m = re.search(r"\b[A-Z]{2,4}/[A-Z0-9]{2,6}/\d{3,6}\b", text or "")
    return m.group(0) if m else ""


def main():
    listing = pd.read_csv(DATA / "vacancies.csv", dtype=str)
    rows = []
    for f in sorted(RAW.glob("*.html")):
        vid = f.stem
        info = parse_page(f.read_text(encoding="utf-8", errors="ignore"))
        info["vacancy_id"] = vid
        info["reference"] = find_reference(info.get("how_to_apply", ""))
        m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", info.get("contact_email", ""))
        info["contact_email"] = m.group(0).rstrip(".") if m else info.get("contact_email", "")
        info.pop("_skip", None)
        rows.append(info)

    details = pd.DataFrame(rows)
    jobs = listing.merge(details, on="vacancy_id", how="inner")

    # Clean dates to YYYY-MM-DD and mark open/closed
    for col in ["posted", "deadline"]:
        jobs[col] = pd.to_datetime(jobs[col], errors="coerce").dt.strftime("%Y-%m-%d")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    jobs["is_open"] = jobs["deadline"].fillna("") >= today

    jobs.to_csv(DATA / "jobs_clean.csv", index=False)
    with sqlite3.connect(DATA / "jobs.db") as con:
        jobs.to_sql("jobs", con, if_exists="replace", index=False)

    # ---- Data quality report ----
    print(f"\nParsed {len(jobs)} postings -> data/jobs.db and data/jobs_clean.csv\n")
    key = ["title", "agency", "deadline", "duties", "minimum_quals", "how_to_apply", "contact_email"]
    print("Missing values (%):")
    for c in key:
        pct = 100 * (jobs[c].isna() | (jobs[c].astype(str).str.strip() == "")).mean() if c in jobs else 100
        print(f"  {c:16s} {pct:5.1f}%")
    print(f"\nDuplicate vacancy IDs: {jobs['vacancy_id'].duplicated().sum()}")
    print(f"Open postings: {int(jobs['is_open'].sum())} of {len(jobs)}")

    # Show one example so we can check the parsing
    ex = jobs.iloc[0]
    print("\nExample posting:")
    for c in ["vacancy_id", "title", "agency", "deadline", "salary_range", "contact_email", "reference"]:
        print(f"  {c}: {ex.get(c, '')}")
    print(f"  minimum_quals: {str(ex.get('minimum_quals', ''))[:200]}...")


if __name__ == "__main__":
    main()