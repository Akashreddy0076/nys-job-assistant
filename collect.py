"""
Step 2 - Collect the list of current NYS job postings from StateJobsNY.
Saves: data/vacancies.csv
Run:   python collect.py
"""
import pathlib
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

LIST_URL = "https://statejobs.ny.gov/public/vacancyTable.cfm"
DETAIL_URL = "https://statejobs.ny.gov/public/vacancyDetailsView.cfm?id="
HEADERS = {"User-Agent": "NYS-Job-Assistant student project"}
DATA = pathlib.Path("data")
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def get_vacancy_list():
    """Download the Current Vacancies table and return it as a DataFrame."""
    r = requests.get(LIST_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < 7 or not cells[0].isdigit():
            continue  # skip header and non-job rows
        rows.append({
            "vacancy_id": cells[0],
            "title": cells[1],
            "grade": cells[2],
            "posted": cells[3],
            "deadline": cells[4],
            "agency": cells[5],
            "county": cells[6],
            "url": DETAIL_URL + cells[0],
        })
    return pd.DataFrame(rows)


def download_details(ids, limit=20):
    """Save each posting's details page once (cached). Waits between requests."""
    done = 0
    for vid in ids:
        out = RAW / f"{vid}.html"
        if out.exists():
            continue
        r = requests.get(DETAIL_URL + vid, headers=HEADERS, timeout=60)
        if r.ok:
            out.write_text(r.text, encoding="utf-8")
            done += 1
            print(f"saved {vid}")
        time.sleep(1.5)  # be polite to the state website
        if done >= limit:
            break
    print(f"Downloaded {done} new detail pages into {RAW}/")


if __name__ == "__main__":
    df = get_vacancy_list()
    df.to_csv(DATA / "vacancies.csv", index=False)
    print(f"Found {len(df)} postings. Saved to data/vacancies.csv")
    print(df.head(10).to_string(index=False))

    # Start small: only 20 detail pages for testing. Raise the limit later.
    download_details(df["vacancy_id"].tolist(), limit=2000)