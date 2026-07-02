"""
Fix catalog.json:
1. Strip non-JSON prefix ('this is the dataset')
2. Handle invalid control characters in strings
"""
import json
import re

with open("data/catalog.json", encoding="utf-8") as f:
    content = f.read()

# Strip everything before the first '['
start = content.index("[")
json_only = content[start:]

# First try strict parse
try:
    data = json.loads(json_only)
    print(f"Parsed OK (strict): {len(data)} assessments")
except json.JSONDecodeError as e:
    print(f"Strict parse failed: {e}")
    # Try with strict=False to allow control chars
    try:
        data = json.loads(json_only, strict=False)
        print(f"Parsed OK (lenient): {len(data)} assessments")
    except json.JSONDecodeError as e2:
        print(f"Lenient parse also failed: {e2}")
        # Last resort: replace literal \r\n inside the JSON string
        # (not inside string values, but that's hard — use a broad replace)
        cleaned = json_only.replace("\r\n", "\\n").replace("\r", "\\n")
        try:
            data = json.loads(cleaned)
            print(f"Parsed OK (after CR strip): {len(data)} assessments")
        except json.JSONDecodeError as e3:
            print(f"All attempts failed: {e3}")
            raise

# Re-serialise cleanly (this removes all embedded control chars safely)
clean_json = json.dumps(data, ensure_ascii=False, indent=2)

with open("data/catalog.json", "w", encoding="utf-8") as f:
    f.write(clean_json)

print(f"Saved {len(data)} assessments to data/catalog.json")
print(f"First: {data[0]['name']}")
print(f"Last:  {data[-1]['name']}")
