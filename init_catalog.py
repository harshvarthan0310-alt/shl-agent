"""
init_catalog.py
---------------
The SHL catalog page is dynamically rendered (JavaScript) so cannot be
scraped with requests alone. The assessment brief provides the catalog as
a JSON dataset directly. This script fetches the catalog using the SHL
website's internal search/filter endpoint and saves it.

Run:  python init_catalog.py
"""
import json, os, re, sys, time, requests

OUTPUT = os.path.join("data", "catalog.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.shl.com/solutions/products/product-catalog/",
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

SHL_PRODUCT_API = "https://www.shl.com/wp-json/wp/v2/product"
SHL_CATALOG_API = "https://www.shl.com/solutions/products/product-catalog/"


def get_test_type(keys):
    for k in keys:
        if k in TEST_TYPE_MAP:
            return TEST_TYPE_MAP[k]
    return "K"


def try_wp_api():
    """Try fetching products via WordPress REST API."""
    items = []
    page = 1
    per_page = 100
    while True:
        try:
            r = requests.get(
                SHL_PRODUCT_API,
                params={"per_page": per_page, "page": page, "product_type": "individual"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=30
            )
            if r.status_code == 404:
                break
            if r.status_code != 200:
                print(f"  WP API: HTTP {r.status_code}")
                break
            data = r.json()
            if not data:
                break
            for post in data:
                title = post.get("title", {}).get("rendered", "")
                link = post.get("link", "")
                description = re.sub(r"<[^>]+>", "", post.get("content", {}).get("rendered", ""))
                items.append({"name": title, "link": link, "description": description[:500]})
            print(f"  WP API page {page}: {len(data)} items")
            page += 1
            time.sleep(0.3)
            if len(data) < per_page:
                break
        except Exception as e:
            print(f"  WP API error: {e}")
            break
    return items


def main():
    print("Attempting to fetch SHL catalog via API endpoints...")

    # Approach: Use the known-correct catalog data that was provided in the
    # assessment brief. The brief provides a comprehensive JSON array.
    # Since the file currently has Python code, we restore from the git history
    # or from the embedded data.

    # First, check if there's a backup
    backup_path = os.path.join("data", "catalog_backup.json")
    if os.path.exists(backup_path):
        print(f"Found backup at {backup_path}")
        with open(backup_path, encoding="utf-8") as f:
            catalog = json.load(f)
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"Restored {len(catalog)} assessments from backup.")
        return

    # Try WordPress API
    print("Trying WordPress REST API...")
    items = try_wp_api()
    if items:
        catalog = []
        for it in items:
            entity_id = it["link"].rstrip("/").split("/")[-1]
            catalog.append({
                "entity_id": entity_id,
                "name": it["name"],
                "link": it["link"],
                "job_levels": [],
                "languages": [],
                "duration": "",
                "status": "ok",
                "remote": "yes",
                "adaptive": "no",
                "description": it["description"],
                "keys": [],
                "test_type": "K",
            })
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(catalog)} assessments via WP API.")
        return

    print("ERROR: Could not fetch catalog automatically.")
    print("Please restore data/catalog.json from the assessment ZIP or git history.")
    sys.exit(1)


if __name__ == "__main__":
    main()
