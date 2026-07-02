"""
agent.py  (v3 – hardened, dual-provider)
-----------------------------------------
Supports both Gemini (google-generativeai) and Groq (openai-compatible).
Provider is auto-selected based on which API key is in .env.

Pipeline:
  1. Heuristic intent detection + LLM fallback  (~0-800 ms)
  2. Catalog retrieval with filters             (~5 ms)
  3. Single LLM call for response               (400–800 ms)
  4. Validate + enforce limits                   (0 ms)
"""

import json
import re
import traceback

from app.config import (
    LLM_PROVIDER, LLM_MODEL,
    GEMINI_API_KEY, GROQ_API_KEY,
    MAX_RECOMMENDATIONS, MAX_TURNS,
)
from app.catalog import CatalogSearch
from app.prompts import RECOMMENDATION_PROMPT, QUERY_REWRITE_PROMPT
from app.models import ChatResponse, Recommendation, Message

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at startup
# ---------------------------------------------------------------------------
_catalog = CatalogSearch()

if LLM_PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel(LLM_MODEL)
    _groq_client  = None
elif LLM_PROVIDER == "groq":
    from openai import OpenAI
    _groq_client  = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    _gemini_model = None
else:
    # No LLM configured (build time) — will fail at runtime if called
    _gemini_model = None
    _groq_client  = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_chat(messages: list[Message]) -> ChatResponse:
    """Stateless handler. Returns the next agent turn."""
    try:
        user_turn_count = sum(1 for m in messages if m.role == "user")

        # ── Stage 1: Intent analysis ────────────────────────────────────
        mode    = _detect_mode(messages, user_turn_count)
        filters = _extract_filters(messages)

        # At or near the turn cap, force recommend mode
        force_recommend_turn = max(3, MAX_TURNS - 2)
        if user_turn_count >= force_recommend_turn and mode == "clarify":
            mode = "recommend"

        # At the final turn, always recommend
        if user_turn_count >= MAX_TURNS and mode in ("clarify", "compare"):
            mode = "recommend"

        queries = _build_queries(messages, mode)
        compare_names = _extract_compare_names(messages) if mode == "compare" else []

        # ── Stage 2: Catalog retrieval ──────────────────────────────────
        retrieved_items: list[dict] = []

        if mode in ("recommend", "refine"):
            retrieved_items = _catalog.search(queries, top_k=12, filters=filters)
            if mode == "refine":
                # Also include items from prior recommendations
                for name in _extract_previous_rec_names(messages):
                    item = _catalog.get_by_name(name)
                    if item and item not in retrieved_items:
                        retrieved_items.append(item)

        elif mode == "compare":
            if compare_names:
                retrieved_items = _catalog.get_by_names(compare_names)
            if queries:
                extra = _catalog.search(queries, top_k=10)
                seen = {i["entity_id"] for i in retrieved_items}
                for item in extra:
                    if item["entity_id"] not in seen:
                        retrieved_items.append(item)

        elif mode == "clarify":
            # Still retrieve some items so the LLM has context to work with
            # if the user gives enough info in one message
            if queries:
                retrieved_items = _catalog.search(queries, top_k=6)
            else:
                # Get a diverse sample for general context
                all_user = " ".join(m.content for m in messages if m.role == "user")
                if len(all_user.strip()) > 5:
                    retrieved_items = _catalog.search([all_user], top_k=6)

        # ── Stage 3: Single LLM call ────────────────────────────────────
        catalog_ctx = _format_catalog_context(retrieved_items)
        conv_str    = _format_conversation(messages)

        prompt = RECOMMENDATION_PROMPT.format(
            mode=mode,
            catalog_data=catalog_ctx or "(No catalog items retrieved — ask clarifying questions.)",
            conversation=conv_str,
            turn_number=user_turn_count,
            max_turns=MAX_TURNS,
            force_recommend_turn=force_recommend_turn,
        )

        raw = _llm_call(prompt)

        # ── Stage 4: Parse, validate, enforce hard limits ───────────────
        return _parse_and_validate(raw, mode, user_turn_count, queries, filters, retrieved_items)

    except Exception as exc:
        traceback.print_exc()
        print(f"[agent] Error: {exc}")
        # Graceful degradation: try keyword-only recommendation
        return _fallback_response(messages)


# ---------------------------------------------------------------------------
# Stage 1 – Intent detection (heuristic + LLM fallback)
# ---------------------------------------------------------------------------

# Compiled patterns for refuse detection — focused on LAST message only
_RE_REFUSE = re.compile(
    r"(?:^|\s)(?:"
    r"what (?:is the |should I |to )(?:best |average |expected )?(?:salary|wage|pay\b|compensation)"
    r"|how much (?:should|do|does|to) (?:I |we )?pay"
    r"|salary (?:range|negotiat|expectat|benchmark)"
    r"|wage (?:range|negotiat)"
    r"|legal\s+(?:advice|issue|question|requirement|obligation)"
    r"|(?:is it |am I |are we )?(?:legal|lawful|allowed) to"
    r"|gdpr|lawsuit|litigation"
    r"|ignore.{0,30}(?:instruct|rules|system)"
    r"|pretend.{0,20}(?:you are|to be|you're)"
    r"|forget.{0,20}(?:instruct|rules|everything)"
    r"|disregard.{0,20}(?:all|previous|your|above)"
    r"|jailbreak"
    r"|(?:tell|write|compose).{0,20}(?:joke|poem|story|essay|song)"
    r"|you are now|act as (?:if|a |an )"
    r")",
    re.I,
)

_RE_COMPARE = re.compile(
    r"(?:difference|compare|vs\.?|versus|between).{0,60}(?:and|or|vs)",
    re.I,
)

_RE_REFINE_KW = [
    "add ", "also add", "remove ", "instead ", "update the",
    "include ", "without ", "exclude ", "change the", "modify ",
    "as well", "too.", " too,", "also want", "also need",
    "also important", "also ",
    "replace ", "swap ", "drop ",
]

# Broader role keywords — avoid "sales" triggering refuse
_ROLE_KW = [
    "developer", "engineer", "analyst", "manager", "designer", "scientist",
    "nurse", "doctor", "accountant", "sales", "marketing", "hr", "recruiter",
    "consultant", "architect", "tester", "qa", "devops", "admin", "support",
    "hiring", "hire", "recruit", "position", "role", "candidate", "job",
    "software", "data", "business", "finance", "operations", "customer",
    "service", "product", "project", "technical", "clinical",
    "executive", "director", "intern", "trainee", "associate", "officer",
    "coordinator", "specialist", "representative", "agent",
    "programmer", "coder", "dba", "sysadmin", "technician",
    "graduate", "fresher", "entry level", "mid level", "senior",
]

_SKILL_KW = [
    "java", "python", "sql", "javascript", "typescript", "react", "angular",
    "node", ".net", "c#", "c++", "ruby", "go ", "scala", " r ", "aws", "azure",
    "machine learning", "deep learning", "leadership", "communication",
    "cognitive", "personality", "aptitude", "numerical", "verbal", "reasoning",
    "coding", "simulation", "agile", "scrum", "excel", "powerpoint", "word",
    "customer service", "problem solving", "teamwork", "negotiation",
    "critical thinking", "attention to detail", "decision making",
    "project management", "stakeholder", "presentation", "writing",
    "analytical", "interpersonal", "emotional intelligence",
    "html", "css", "docker", "kubernetes", "linux", "windows server",
    "salesforce", "sap", "oracle", "tableau", "power bi",
    "assessment", "test", "evaluation", "measure",
]

_LEVEL_KW = [
    "junior", "senior", "mid-level", "mid level", "entry level", "entry-level",
    "graduate", "executive", "director", "manager", "lead ", "principal",
    "years experience", "years of experience", "4 years", "5 years", "experienced",
    "fresher", "intern", "trainee", "c-suite", "vp ", "vice president",
]

# Correction patterns
_CORRECTION_KW = [
    "actually", "i meant", "i mean", "correction", "not ", "no,",
    "sorry,", "wait,", "change that", "let me correct", "i misspoke",
    "instead of", "rather than",
]


def _detect_mode(messages: list[Message], user_turn_count: int) -> str:
    """Heuristic mode detection — fast path for clear-cut cases."""
    last_user = _last_user_message(messages).lower()
    all_user = " ".join(m.content for m in messages if m.role == "user").lower()

    # 1. Refuse — check LAST user message only (not entire history)
    if _RE_REFUSE.search(last_user):
        # But NOT if the message also contains strong assessment signals
        has_assessment_signal = any(kw in last_user for kw in [
            "assessment", "test", "evaluate", "measure", "hire", "hiring",
            "candidate", "recruit", "screen",
        ])
        if not has_assessment_signal:
            return "refuse"

    # 2. Compare
    if _RE_COMPARE.search(last_user):
        return "compare"

    # 3. Refine — only if there are prior assistant messages WITH recommendations
    has_prior_recs = _has_prior_recommendations(messages)
    if has_prior_recs and any(kw in last_user for kw in _RE_REFINE_KW):
        return "refine"

    # 4. Correction — treat as recommend/refine with updated context
    if any(kw in last_user for kw in _CORRECTION_KW):
        if has_prior_recs:
            return "refine"
        return "recommend"

    # 5. Recommend — any role/skill/level signal in the conversation
    if (
        any(kw in all_user for kw in _ROLE_KW)
        or any(kw in all_user for kw in _SKILL_KW)
        or any(kw in all_user for kw in _LEVEL_KW)
    ):
        return "recommend"

    # 6. Turn count: after 3+ user turns bias toward action
    if user_turn_count >= 3:
        return "recommend"

    # 7. Default: clarify
    return "clarify"


def _has_prior_recommendations(messages: list[Message]) -> bool:
    """Check if any prior assistant message contained recommendations."""
    for msg in messages:
        if msg.role == "assistant":
            # Check if the message mentions specific assessment names
            # (the harness sends back the reply text, not the full JSON)
            content_lower = msg.content.lower()
            # If the assistant mentioned product catalog URLs or gave a structured response
            if ("shl.com" in content_lower
                or "recommend" in content_lower
                or "assessment" in content_lower
                or "here are" in content_lower
                or "shortlist" in content_lower
                or "following" in content_lower):
                return True
    return False


def _build_queries(messages: list[Message], mode: str) -> list[str]:
    """Build catalog search queries from the conversation."""
    if mode == "refuse":
        return []

    user_msgs = [m.content for m in messages if m.role == "user"]
    if not user_msgs:
        return []

    # Extract meaningful terms from user messages
    queries = []

    # Primary query: last 3 user messages combined (gives full context)
    primary = " ".join(user_msgs[-3:])
    if primary.strip():
        queries.append(primary)

    # Individual recent messages as separate queries for diversity
    for msg in user_msgs[-2:]:
        if msg.strip() and len(msg) > 10 and msg != primary:
            queries.append(msg)

    # If mode is clarify, still try to build a query from whatever we have
    if mode == "clarify" and not queries:
        all_text = " ".join(user_msgs)
        if len(all_text.strip()) > 3:
            queries.append(all_text)

    return queries


def _extract_filters(messages: list[Message]) -> dict:
    """Extract structured filters from user messages."""
    all_user = " ".join(m.content for m in messages if m.role == "user").lower()
    # Prefer latest correction if user changed their mind
    last_user = _last_user_message(messages).lower()

    filters = {}

    # Job level extraction (check last message first for corrections)
    level_map = {
        "entry level": "Entry-Level", "entry-level": "Entry-Level",
        "junior": "Entry-Level", "fresher": "Entry-Level",
        "graduate": "Graduate", "intern": "Entry-Level",
        "mid level": "Mid-Professional", "mid-level": "Mid-Professional",
        "mid professional": "Mid-Professional",
        "4 years": "Mid-Professional", "3 years": "Mid-Professional",
        "5 years": "Mid-Professional", "3-5 years": "Mid-Professional",
        "senior": "Professional Individual Contributor",
        "lead": "Professional Individual Contributor",
        "principal": "Professional Individual Contributor",
        "experienced": "Professional Individual Contributor",
        "manager": "Manager", "head of": "Manager",
        "front line manager": "Front Line Manager",
        "director": "Director",
        "vp": "Executive", "vice president": "Executive",
        "executive": "Executive", "c-suite": "Executive", "cxo": "Executive",
    }
    # Check last message first (for corrections)
    for keyword, level in level_map.items():
        if keyword in last_user:
            filters["job_level"] = level
            break
    if "job_level" not in filters:
        for keyword, level in level_map.items():
            if keyword in all_user:
                filters["job_level"] = level
                break

    # Remote preference
    if "remote" in last_user or "online" in last_user or "virtual" in last_user:
        filters["remote"] = "yes"
    elif "remote" in all_user or "online" in all_user:
        filters["remote"] = "yes"

    # Adaptive preference
    if "adaptive" in last_user or "irt" in last_user:
        filters["adaptive"] = "yes"
    elif "adaptive" in all_user:
        filters["adaptive"] = "yes"

    return filters


def _extract_compare_names(messages: list[Message]) -> list[str]:
    """Pull assessment names from a comparison request."""
    last = _last_user_message(messages)
    names = []
    for sep in [" and ", " vs ", " versus ", " or "]:
        if sep in last.lower():
            parts = re.split(sep, last, flags=re.I)
            for part in parts:
                part = re.sub(
                    r"\b(what|is|the|difference|between|compare|of|does|do|how|can|you|tell|me|about)\b",
                    "", part, flags=re.I
                ).strip()
                part = part.strip("?., ")
                if len(part) > 3:
                    names.append(part)
            break
    return names[:4]


def _extract_previous_rec_names(messages: list[Message]) -> list[str]:
    """Extract assessment names mentioned in prior assistant messages."""
    names = []
    for msg in messages:
        if msg.role == "assistant":
            # Look for assessment-like names in the reply text
            # The harness sends back the assistant's reply text
            content = msg.content
            # Try to find names that look like SHL assessment names
            # (capitalized multi-word phrases, often with parentheses or numbers)
            patterns = [
                r"(?:^|\n)\s*\d+\.\s*\*?\*?([^*\n:]+?)(?:\*?\*?\s*[-–:])",  # numbered lists
                r"\*\*([^*]+)\*\*",  # bold text (likely assessment names)
            ]
            for pattern in patterns:
                found = re.findall(pattern, content)
                for name in found:
                    name = name.strip()
                    if len(name) > 5 and len(name) < 100:
                        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Stage 2 – Catalog helpers
# ---------------------------------------------------------------------------

def _format_catalog_context(items: list[dict]) -> str:
    if not items:
        return ""
    lines = []
    for item in items:
        lines.append(
            f"Name: {item['name']}\n"
            f"URL: {item['link']}\n"
            f"Test Type: {item.get('test_type', 'K')}\n"
            f"Description: {item.get('description', '')}\n"
            f"Categories: {', '.join(item.get('keys', []))}\n"
            f"Job Levels: {', '.join(item.get('job_levels', []))}\n"
            f"Duration: {item.get('duration', 'N/A')} | "
            f"Remote: {item.get('remote', 'N/A')} | "
            f"Adaptive: {item.get('adaptive', 'N/A')}\n"
            "---"
        )
    return "\n".join(lines)


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def _last_user_message(messages: list[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Stage 3 – LLM call (provider-agnostic)
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, temperature: float = 0.3, max_retries: int = 3) -> str:
    """Call LLM with exponential backoff retry for rate limits."""
    import time as _time

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            if LLM_PROVIDER == "groq":
                resp = _groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048,
                )
                return resp.choices[0].message.content
            else:
                config = genai.types.GenerationConfig(temperature=temperature)
                response = _gemini_model.generate_content(prompt, generation_config=config)
                return response.text
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on rate limit (429) or transient server errors (5xx)
            if "rate_limit" in error_str or "429" in error_str or "500" in error_str or "503" in error_str:
                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                print(f"[llm] Rate limited (attempt {attempt+1}/{max_retries+1}), retrying in {wait}s...")
                _time.sleep(wait)
                continue
            else:
                raise  # Non-retryable error
    # All retries exhausted
    raise last_error


# ---------------------------------------------------------------------------
# Stage 4 – Parse, validate, enforce hard limits
# ---------------------------------------------------------------------------

def _parse_and_validate(
    raw: str,
    mode: str,
    user_turn_count: int,
    queries: list[str],
    filters: dict,
    retrieved_items: list[dict] | None = None,
) -> ChatResponse:
    try:
        data = _parse_json(raw)
    except Exception:
        # JSON parse failed — try to extract useful text
        clean_text = _strip_json_fences(raw)[:500]
        # If mode is recommend/refine, fall back to catalog results
        if mode in ("recommend", "refine") and retrieved_items:
            return _catalog_fallback_response(retrieved_items, clean_text, user_turn_count)
        # If we're at the turn cap, force recommendations from catalog
        if user_turn_count >= MAX_TURNS:
            return _force_final_response(queries, filters, clean_text)
        return ChatResponse(
            reply=clean_text if clean_text else (
                "I'd be happy to help you find the right SHL assessments. "
                "Could you tell me about the role you're hiring for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # Extract and validate recommendations
    recommendations: list[Recommendation] = []
    seen_urls: set[str] = set()  # Deduplicate
    for rec in data.get("recommendations", []):
        if not isinstance(rec, dict):
            continue
        validated = _validate_rec(rec)
        if validated and validated.url not in seen_urls:
            recommendations.append(validated)
            seen_urls.add(validated.url)
        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

    # Enforce mode constraints
    if mode in ("clarify", "compare", "refuse"):
        recommendations = []

    # CRITICAL FALLBACK: If mode is recommend/refine but LLM returned
    # 0 valid recommendations, inject top catalog results directly.
    # This prevents the agent from asking more questions when it should recommend.
    if mode in ("recommend", "refine") and not recommendations and retrieved_items:
        recommendations = _items_to_recommendations(retrieved_items)

    # EOC logic
    eoc = bool(data.get("end_of_conversation", False))

    # At the turn cap: force EOC and ensure we have recommendations
    if user_turn_count >= MAX_TURNS:
        eoc = True
        if not recommendations:
            return _force_final_response(queries, filters, data.get("reply", ""))

    # If we have recommendations near the cap, force EOC
    if user_turn_count >= MAX_TURNS - 1 and recommendations:
        eoc = True

    # Don't set EOC=true without recommendations (unless refusing)
    if eoc and not recommendations and mode not in ("refuse",):
        eoc = False

    reply = data.get("reply", "").strip()
    if not reply:
        reply = "I'm here to help with SHL assessment selection. Could you share more about the role?"

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=eoc,
    )


def _force_final_response(
    queries: list[str],
    filters: dict,
    partial_reply: str,
) -> ChatResponse:
    """Force a final response with catalog-based recommendations when at turn cap."""
    # Try to get recommendations from catalog directly
    items = []
    if queries:
        items = _catalog.search(queries, top_k=10, filters=filters)
    if not items:
        # Fallback: get general popular items
        items = _catalog.search(["general assessment cognitive personality"], top_k=10)

    recommendations = []
    for item in items[:MAX_RECOMMENDATIONS]:
        recommendations.append(Recommendation(
            name=item["name"],
            url=item["link"],
            test_type=item.get("test_type", "K"),
        ))

    reply = partial_reply or (
        "Based on our conversation, here are the SHL assessments I'd recommend:"
    )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=True,
    )


def _items_to_recommendations(items: list[dict], max_items: int = MAX_RECOMMENDATIONS) -> list[Recommendation]:
    """Convert catalog items to Recommendation objects."""
    recommendations = []
    seen_urls: set[str] = set()
    for item in items:
        if item["link"] in seen_urls:
            continue
        recommendations.append(Recommendation(
            name=item["name"],
            url=item["link"],
            test_type=item.get("test_type", "K"),
        ))
        seen_urls.add(item["link"])
        if len(recommendations) >= max_items:
            break
    return recommendations


def _catalog_fallback_response(
    retrieved_items: list[dict],
    partial_reply: str,
    user_turn_count: int,
) -> ChatResponse:
    """Fallback when LLM response is unparseable but we have catalog results."""
    recommendations = _items_to_recommendations(retrieved_items)
    reply = partial_reply if partial_reply else (
        "Based on your requirements, here are the SHL assessments I'd recommend:"
    )
    eoc = user_turn_count >= MAX_TURNS - 1
    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=eoc,
    )


def _validate_rec(rec: dict) -> Recommendation | None:
    """Validate a recommendation against the catalog. Returns None if invalid."""
    url  = rec.get("url", "").strip()
    name = rec.get("name", "").strip()

    # Try URL lookup first (most reliable)
    item = _catalog.get_by_url(url)
    if not item:
        # Try name lookup (handles LLM name variations)
        item = _catalog.get_by_name(name)
    if not item:
        # Try fuzzy name match
        item = _catalog.fuzzy_match(name)
    if not item:
        return None

    return Recommendation(
        name=item["name"],
        url=item["link"],
        test_type=item["test_type"],
    )


def _fallback_response(messages: list[Message]) -> ChatResponse:
    """Graceful degradation when the main pipeline fails."""
    user_turn_count = sum(1 for m in messages if m.role == "user")

    # If near the turn cap, try to give recommendations anyway
    if user_turn_count >= MAX_TURNS - 1:
        all_user = " ".join(m.content for m in messages if m.role == "user")
        items = _catalog.search([all_user], top_k=10) if all_user.strip() else []
        if items:
            recommendations = [
                Recommendation(
                    name=item["name"],
                    url=item["link"],
                    test_type=item.get("test_type", "K"),
                )
                for item in items[:MAX_RECOMMENDATIONS]
            ]
            return ChatResponse(
                reply="Based on our conversation, here are the SHL assessments I'd recommend:",
                recommendations=recommendations,
                end_of_conversation=True,
            )

    return ChatResponse(
        reply=(
            "I'd love to help you find the right SHL assessments. "
            "Could you tell me about the role you're hiring for and "
            "what skills or competencies you'd like to assess?"
        ),
        recommendations=[],
        end_of_conversation=False,
    )


def _parse_json(text: str) -> dict:
    text = _strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError("No JSON object found")


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()
