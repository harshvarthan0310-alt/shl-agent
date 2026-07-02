import os
from dotenv import load_dotenv

load_dotenv()

# ── API Key / Provider detection ─────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("Groq_API_KEY", os.getenv("GROQ_API_KEY", ""))

# Choose provider: prefer Gemini if both keys present
if GEMINI_API_KEY:
    LLM_PROVIDER = "gemini"
    LLM_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
elif GROQ_API_KEY:
    LLM_PROVIDER = "groq"
    LLM_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
else:
    # Gracefully handle missing keys at build time (e.g., Docker build).
    # The actual LLM call will fail at runtime, but index building doesn't need it.
    LLM_PROVIDER = "none"
    LLM_MODEL    = ""
    print("[config] WARNING: No LLM API key found. Set GEMINI_API_KEY or Groq_API_KEY.")

if LLM_PROVIDER != "none":
    print(f"[config] LLM provider: {LLM_PROVIDER} | model: {LLM_MODEL}")

# File paths
_BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
CATALOG_PATH = os.path.join(_BASE_DIR, "data", "catalog.json")
INDEX_PATH   = os.path.join(_BASE_DIR, "data", "faiss.index")
TEXTS_PATH   = os.path.join(_BASE_DIR, "data", "catalog_texts.pkl")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Scoring limits
MAX_RECOMMENDATIONS = 10
MAX_TURNS = 8

# Test type mapping from catalog keys to codes
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
