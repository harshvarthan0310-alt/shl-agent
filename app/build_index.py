"""
build_index.py
--------------
Run once (during Docker build or locally) to pre-compute the FAISS index
and save it to disk.  Subsequent startups load from disk — near-instant cold start.

Usage:
    python -m app.build_index
"""

import json
import pickle
import os
import numpy as np

def build():
    # Import here so the module can be run standalone before the rest of the
    # app package is fully initialised.
    from sentence_transformers import SentenceTransformer
    import faiss
    from app.config import CATALOG_PATH, INDEX_PATH, TEXTS_PATH, EMBEDDING_MODEL

    print(f"[build_index] Loading catalog from {CATALOG_PATH} …")
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"[build_index] {len(catalog)} assessments loaded.")

    # Build rich text representations
    texts = [_build_text(item) for item in catalog]

    print(f"[build_index] Encoding with {EMBEDDING_MODEL} …")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # FAISS IndexFlatIP (cosine-equivalent for L2-normalised vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Persist
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)

    print(f"[build_index] Index saved -> {INDEX_PATH}")
    print(f"[build_index] Texts saved -> {TEXTS_PATH}")
    return index, texts


def _build_text(item: dict) -> str:
    """Create a rich, searchable text blob for an assessment."""
    name        = item.get("name", "")
    description = item.get("description", "")
    keys        = ", ".join(item.get("keys", []))
    levels      = ", ".join(item.get("job_levels", []))
    languages   = ", ".join(item.get("languages", []))
    duration    = item.get("duration", "")
    return (
        f"{name}. "
        f"{description} "
        f"Categories: {keys}. "
        f"Job levels: {levels}. "
        f"Languages: {languages}. "
        f"Duration: {duration}."
    )


if __name__ == "__main__":
    build()
