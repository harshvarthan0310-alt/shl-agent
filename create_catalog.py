"""
write_catalog_from_api.py
-------------------------
Fetches the SHL catalog using their public API endpoint and writes to data/catalog.json.
The SHL catalog page uses a paginated server-side-rendered HTML table.

Run:  python write_catalog_from_api.py
"""
import json, os, re, time, requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone

OUTPUT = os.path.join("data", "catalog.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; catalog-fetcher/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
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

SHL_BASE = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

# Type=1 filter restricts to "Individual Test Solutions"
PARAMS_TEMPLATE = {"type": "1", "start": 0}


def get_test_type(keys):
    for k in keys:
        if k in TEST_TYPE_MAP:
            return TEST_TYPE_MAP[k]
    return "K"


def fetch_page(start: int) -> str | None:
    params = {**PARAMS_TEMPLATE, "start": start}
    try:
        r = requests.get(CATALOG_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  fetch_page(start={start}) failed: {e}")
        return None


def parse_page(html: str, seen: set) -> list[dict]:
    """
    Parse catalog HTML page. SHL renders a table:
    Col 0: Assessment name (link)
    Col 1: Test type tags  (span.product-catalogue__key)
    Col 2: Remote testing  (img alt or span)
    Col 3: Adaptive/IRT    (img alt or span)
    Col 4: Duration
    Col 5: Job Levels
    """
    soup = BeautifulSoup(html, "html.parser")
    items = []

    # Find all product rows
    rows = soup.select("table.custom-table tbody tr, .product-catalogue tr:not(:first-child)")
    if not rows:
        # Fallback: any tr with a product-catalog link
        rows = [tr for tr in soup.find_all("tr")
                if tr.find("a", href=re.compile(r"/product-catalog/view/"))]

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue

        # Name + link
        a = cells[0].find("a", href=re.compile(r"/product-catalog/view/|/products/product-catalog/view/"))
        if not a:
            continue
        name = a.get_text(strip=True)
        href = a["href"]
        link = href if href.startswith("http") else SHL_BASE + href
        if link in seen:
            continue
        seen.add(link)

        # Test type keys — look for spans with known category text
        keys = []
        if len(cells) > 1:
            for el in cells[1].find_all(["span", "p", "li"]):
                t = el.get_text(strip=True)
                if t in TEST_TYPE_MAP:
                    keys.append(t)
            if not keys:
                raw_text = cells[1].get_text(" ", strip=True)
                for k in TEST_TYPE_MAP:
                    if k in raw_text:
                        keys.append(k)

        # Remote
        remote = "no"
        if len(cells) > 2:
            c = cells[2]
            txt = c.get_text(strip=True).lower()
            img = c.find("img")
            alt = (img.get("alt", "") if img else "").lower()
            if "yes" in txt or "✓" in txt or "yes" in alt or "check" in alt:
                remote = "yes"

        # Adaptive
        adaptive = "no"
        if len(cells) > 3:
            c = cells[3]
            txt = c.get_text(strip=True).lower()
            img = c.find("img")
            alt = (img.get("alt", "") if img else "").lower()
            if "yes" in txt or "✓" in txt or "yes" in alt or "check" in alt:
                adaptive = "yes"

        # Duration
        duration_raw = ""
        if len(cells) > 4:
            duration_raw = cells[4].get_text(strip=True)

        # Job levels
        job_levels_raw = ""
        if len(cells) > 5:
            job_levels_raw = cells[5].get_text(strip=True)
        job_levels = [j.strip() for j in re.split(r"[,\n]+", job_levels_raw) if j.strip()]

        # Description (not on catalog index — populated later or left empty)
        description = ""

        entity_id = link.rstrip("/").split("/")[-1]

        items.append({
            "entity_id": entity_id,
            "name": name,
            "link": link,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "job_levels": job_levels,
            "job_levels_raw": job_levels_raw,
            "languages": [],
            "languages_raw": "",
            "duration": duration_raw,
            "duration_raw": f"Approximate Completion Time in minutes = {duration_raw}",
            "status": "ok",
            "remote": remote,
            "adaptive": adaptive,
            "description": description,
            "keys": keys,
            "test_type": get_test_type(keys),
        })
    return items


def count_total(html: str) -> int:
    """Try to find total count in page."""
    m = re.search(r"(\d+)\s*result", html, re.I)
    return int(m.group(1)) if m else 9999


def main():
    print("Starting SHL catalog fetch …")
    catalog = []
    seen = set()
    start = 0
    PAGE_SIZE = 12

    # First page — also get total count
    html = fetch_page(0)
    if not html:
        print("FATAL: Could not fetch catalog page.")
        return

    total = count_total(html)
    print(f"Estimated total: {total}")

    items = parse_page(html, seen)
    catalog.extend(items)
    print(f"  Page start=0: {len(items)} items")

    start = PAGE_SIZE
    consecutive_empty = 0

    while start <= total and consecutive_empty < 3:
        html = fetch_page(start)
        if not html:
            consecutive_empty += 1
            start += PAGE_SIZE
            continue

        items = parse_page(html, seen)
        if not items:
            consecutive_empty += 1
            print(f"  Page start={start}: 0 new items")
        else:
            consecutive_empty = 0
            catalog.extend(items)
            print(f"  Page start={start}: {len(items)} items (total: {len(catalog)})")

        start += PAGE_SIZE
        time.sleep(0.5)

    print(f"\nFetched {len(catalog)} assessments total.")
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
