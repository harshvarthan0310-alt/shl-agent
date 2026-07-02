"""
catalog.py
----------
Hybrid retrieval layer combining:
  1. FAISS semantic search (dense embeddings, cosine similarity)
  2. Keyword overlap scoring (TF-IDF-inspired token matching on name + description)

Results are merged with Reciprocal Rank Fusion (RRF) to balance precision and
recall.  The FAISS index is loaded from disk if available (pre-built by
build_index.py) so cold-start is near-instant.
"""

import json
import math
import os
import pickle
import re
import numpy as np

from app.config import (
    CATALOG_PATH, INDEX_PATH, TEXTS_PATH,
    EMBEDDING_MODEL, TEST_TYPE_MAP,
)


class CatalogSearch:
    def __init__(self):
        # 1. Load catalog
        with open(CATALOG_PATH, encoding="utf-8") as f:
            self.catalog: list[dict] = json.load(f)

        # 2. Attach canonical test_type and slug to every item (computed once)
        for item in self.catalog:
            item["test_type"] = self._get_test_type(item)
            item["_name_lower"] = item["name"].lower()

        # 3. Build URL → item lookup (O(1) validation)
        self._url_map: dict[str, dict] = {item["link"]: item for item in self.catalog}

        # 4. Load or build FAISS index + embedding model
        if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
            self._load_index()
        else:
            self._build_index()

        # 5. Pre-tokenise catalog texts for keyword scoring
        self._token_index: list[set[str]] = [
            _tokenise(t) for t in self._texts
        ]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def search(
        self,
        queries: list[str] | str,
        top_k: int = 25,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid semantic + keyword search.

        Parameters
        ----------
        queries : one or more search queries (strings).  Results are merged
                  across all queries before RRF.
        top_k   : number of items to return (after optional filter).
        filters : dict with optional keys:
                    job_level  – str to match against item["job_levels"]
                    remote     – "yes" | "no"
                    adaptive   – "yes" | "no"
        """
        if isinstance(queries, str):
            queries = [queries]

        # Collect candidate sets from every query
        faiss_rank:   dict[int, int] = {}   # idx → best FAISS rank
        keyword_rank: dict[int, int] = {}   # idx → best keyword rank

        for q in queries:
            if not q.strip():
                continue
            f_results = self._faiss_search(q, top_k=top_k)
            k_results = self._keyword_search(q, top_k=top_k)

            for rank, idx in enumerate(f_results):
                if idx not in faiss_rank or faiss_rank[idx] > rank:
                    faiss_rank[idx] = rank
            for rank, idx in enumerate(k_results):
                if idx not in keyword_rank or keyword_rank[idx] > rank:
                    keyword_rank[idx] = rank

        # Union of all candidates
        all_candidates = set(faiss_rank) | set(keyword_rank)

        # RRF score (k=60 is the standard constant)
        K = 60
        rrf: dict[int, float] = {}
        for idx in all_candidates:
            score = 0.0
            if idx in faiss_rank:
                score += 1.0 / (K + faiss_rank[idx] + 1)
            if idx in keyword_rank:
                score += 1.0 / (K + keyword_rank[idx] + 1)
            rrf[idx] = score

        # Sort by RRF score
        ranked = sorted(rrf, key=lambda i: rrf[i], reverse=True)

        # Apply filters
        results = []
        for idx in ranked:
            item = self.catalog[idx]
            if filters and not self._passes_filter(item, filters):
                continue
            results.append(item)
            if len(results) >= top_k:
                break

        return results

    def get_by_name(self, name: str) -> dict | None:
        """Exact then substring name lookup."""
        name_lower = name.lower().strip()
        # Exact match first
        for item in self.catalog:
            if item["_name_lower"] == name_lower:
                return item
        # Substring match
        for item in self.catalog:
            if name_lower in item["_name_lower"]:
                return item
        return None

    def get_by_names(self, names: list[str]) -> list[dict]:
        """Batch name lookup — returns items found (preserves order, skips misses)."""
        found = []
        seen = set()
        for name in names:
            item = self.get_by_name(name)
            if item and item["entity_id"] not in seen:
                found.append(item)
                seen.add(item["entity_id"])
        return found

    def fuzzy_match(self, name: str, threshold: float = 0.4) -> dict | None:
        """
        Token-overlap fuzzy matching for assessment names.
        Used when exact and substring matching both fail.
        Returns the best match above the threshold, or None.
        """
        if not name or len(name.strip()) < 3:
            return None
        query_tokens = _tokenise(name)
        if not query_tokens:
            return None

        best_score = 0.0
        best_item = None
        for item in self.catalog:
            item_tokens = _tokenise(item["name"])
            if not item_tokens:
                continue
            overlap = len(query_tokens & item_tokens)
            if overlap == 0:
                continue
            # Jaccard-like score: overlap / union
            union = len(query_tokens | item_tokens)
            score = overlap / union if union > 0 else 0
            # Bonus for containing the query as a substring
            if name.lower().strip() in item["_name_lower"]:
                score += 0.3
            if score > best_score:
                best_score = score
                best_item = item

        return best_item if best_score >= threshold else None

    def get_by_url(self, url: str) -> dict | None:
        """O(1) URL → item lookup."""
        return self._url_map.get(url)

    def is_valid_url(self, url: str) -> bool:
        return url in self._url_map

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _faiss_search(self, query: str, top_k: int) -> list[int]:
        import faiss  # noqa: F401 (loaded lazily to avoid import error at build time)
        q_emb = self._model.encode([query], normalize_embeddings=True)
        _, indices = self._index.search(q_emb.astype(np.float32), top_k)
        return [int(i) for i in indices[0] if i >= 0]

    def _keyword_search(self, query: str, top_k: int) -> list[int]:
        q_tokens = _tokenise(query)
        if not q_tokens:
            return []
        scores = []
        for idx, doc_tokens in enumerate(self._token_index):
            overlap = len(q_tokens & doc_tokens)
            if overlap:
                # Normalise by query length to prefer precise matches
                scores.append((overlap / math.sqrt(len(q_tokens)), idx))
        scores.sort(reverse=True)
        return [idx for _, idx in scores[:top_k]]

    @staticmethod
    def _get_test_type(item: dict) -> str:
        for key in item.get("keys", []):
            if key in TEST_TYPE_MAP:
                return TEST_TYPE_MAP[key]
        return "K"

    @staticmethod
    def _passes_filter(item: dict, filters: dict) -> bool:
        if "job_level" in filters and filters["job_level"]:
            levels = [l.lower() for l in item.get("job_levels", [])]
            if filters["job_level"].lower() not in levels:
                return False
        if "remote" in filters and filters["remote"]:
            if item.get("remote", "").lower() != filters["remote"].lower():
                return False
        if "adaptive" in filters and filters["adaptive"]:
            if item.get("adaptive", "").lower() != filters["adaptive"].lower():
                return False
        return True

    def _load_index(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        self._index = faiss.read_index(INDEX_PATH)
        with open(TEXTS_PATH, "rb") as f:
            self._texts: list[str] = pickle.load(f)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[catalog] Loaded FAISS index ({self._index.ntotal} vectors).")

    def _build_index(self):
        """Fallback: build index in memory (no file I/O)."""
        import faiss
        from sentence_transformers import SentenceTransformer
        from app.build_index import _build_text
        print("[catalog] Pre-built index not found — building in memory …")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._texts = [_build_text(item) for item in self.catalog]
        embeddings = self._model.encode(
            self._texts, normalize_embeddings=True, show_progress_bar=False
        )
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        print(f"[catalog] Built in-memory index ({self._index.ntotal} vectors).")


# ------------------------------------------------------------------ #
#  Utilities                                                           #
# ------------------------------------------------------------------ #

def _tokenise(text: str) -> set[str]:
    """Lowercase alpha-only tokens, min length 3."""
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    return set(tokens)
