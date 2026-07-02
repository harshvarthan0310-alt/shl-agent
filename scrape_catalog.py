"""
scrape_catalog.py
-----------------
Scrapes the SHL Individual Test Solutions catalog and saves to data/catalog.json.
Run once before building the FAISS index.

    python scrape_catalog.py
"""

import json
import time
import re
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone

BASE = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT = os.path.join(os.path.dirname(__file__), "data", "catalog.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TEST_TYPE_MAP = {
    "Knowledge & Skills":             "K",
    "Personality & Behavior":         "P",
    "Ability & Aptitude":             "A",
    "Simulations":                    "S",
    "Assessment Exercises":           "E",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
}


def _get_test_type(keys):
    for k in keys:
        if k in TEST_TYPE_MAP:
            return TEST_TYPE_MAP[k]
    return "K"


def fetch_catalog_links():
    """Fetch all individual test product links from the catalog page (with pagination)."""
    links = []
    start = 0
    page_size = 12  # SHL uses 12 per page

    while True:
        url = f"{CATALOG_URL}?start={start}&type=1"
        print(f"  Fetching catalog page start={start} …")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} — stopping pagination.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find product rows — SHL catalog uses a table or card layout
        # Look for links matching /solutions/products/product-catalog/view/
        page_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/product-catalog/view/" in href or "/products/product-catalog/view/" in href:
                full = href if href.startswith("http") else BASE + href
                if full not in links and full not in page_links:
                    page_links.append(full)

        if not page_links:
            print("  No new links found — pagination complete.")
            break

        links.extend(page_links)
        print(f"  Found {len(page_links)} links (total: {len(links)})")
        start += page_size
        time.sleep(0.5)

    return links


def scrape_product(url: str) -> dict | None:
    """Scrape a single product page and return structured data."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # Name
        name_el = soup.find("h1") or soup.find("h2")
        name = name_el.get_text(strip=True) if name_el else url.split("/")[-2]

        # Description
        desc_el = soup.find("div", class_=re.compile(r"product.?description|description", re.I))
        if not desc_el:
            desc_el = soup.find("div", class_=re.compile(r"content|body|detail", re.I))
        description = desc_el.get_text(" ", strip=True)[:2000] if desc_el else ""

        # Parse fact sheet table / dl
        facts = {}
        for row in soup.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) == 2:
                key = cells[0].get_text(strip=True).lower()
                val = cells[1].get_text(strip=True)
                facts[key] = val
        for dl in soup.find_all("dl"):
            terms = dl.find_all("dt")
            defs = dl.find_all("dd")
            for dt, dd in zip(terms, defs):
                facts[dt.get_text(strip=True).lower()] = dd.get_text(strip=True)

        # Job levels
        jl_raw = facts.get("job level", facts.get("job levels", ""))
        job_levels = [j.strip() for j in jl_raw.split(",") if j.strip()]

        # Languages
        lang_raw = facts.get("language", facts.get("languages", ""))
        languages = [l.strip() for l in lang_raw.split(",") if l.strip()]

        # Duration
        dur_raw = facts.get("approximate completion time", facts.get("duration", ""))

        # Remote / Adaptive
        remote = "yes" if "remote" in resp.text.lower() else "no"
        adaptive = "yes" if "adaptive" in facts.get("adaptive", "").lower() else "no"

        # Categories / Keys — look for tag-like elements
        keys = []
        for tag in soup.find_all(class_=re.compile(r"tag|category|key|filter", re.I)):
            t = tag.get_text(strip=True)
            if t in TEST_TYPE_MAP:
                keys.append(t)
        keys = list(dict.fromkeys(keys))  # deduplicate

        entity_id = url.rstrip("/").split("/")[-1]

        return {
            "entity_id": entity_id,
            "name": name,
            "link": url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "job_levels": job_levels,
            "job_levels_raw": jl_raw,
            "languages": languages,
            "languages_raw": lang_raw,
            "duration": dur_raw,
            "duration_raw": f"Approximate Completion Time in minutes = {dur_raw}",
            "status": "ok",
            "remote": remote,
            "adaptive": adaptive,
            "description": description,
            "keys": keys,
            "test_type": _get_test_type(keys),
        }
    except Exception as e:
        print(f"  ERROR scraping {url}: {e}")
        return None


def main():
    print("Fetching catalog index …")
    links = fetch_catalog_links()
    print(f"Total product links found: {len(links)}")

    catalog = []
    for i, url in enumerate(links, 1):
        print(f"[{i}/{len(links)}] {url}")
        item = scrape_product(url)
        if item:
            catalog.append(item)
        time.sleep(0.3)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(catalog)} assessments → {OUTPUT}")


if __name__ == "__main__":
    main()
