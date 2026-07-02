1. System Architecture & Design Choices 
The recommendation agent is designed around a single-call pipeline optimized for low 
latency, high schema compliance, and robustness against deterministic conversation 
drift. 
Technology Stack Justification 
• FastAPI: Chosen for its near-zero overhead, native asynchronous support, and 
automated OpenAPI/Pydantic validation schemas. 
• FAISS (IndexFlatIP): In-memory vector database utilized for dense semantic 
retrieval. It avoids external network hops, keeping retrieval latency below 5ms. 
• Sentence-Transformers (all-MiniLM-L6-v2): A lightweight 384-dimensional 
dense embedding model pre-downloaded and baked directly into the Docker 
image, eliminating runtime cold-starts. 
• Groq / Llama 3.3 70B: Provides high-quality reasoning and schema compliance. 
The system is designed to fall back gracefully to Gemini 2.0 Flash if API limits are 
saturated.



2. Retrieval Setup & Optimization 
To maximize Recall@10, we implemented a hybrid retrieval strategy combining dense 
semantic vectors and sparse lexical search, merged via Reciprocal Rank Fusion (RRF): 
1. Lexical Matching: Standard token-overlap scoring to ensure exact matches for 
specific keywords (e.g., "Java", "SQL", "OPQ32r"). 
2. Dense Vector Search: Cosine similarity against descriptions, categories, and 
job levels using normalized embeddings. 
3. RRF Merging: Ranks items using a standard constant (k=60) to balance high
precision exact queries with conceptual queries. 
4. Structured Filters: Automatically extracts job_level, remote, 
and adaptive constraints from user messages to filter catalog candidates before 
feeding them to the LLM context. 
5. Catalog Direct Fallback: If the LLM output is corrupted or fails to produce 
recommendations when in recommendation mode, the system automatically 
injects the top-ranked candidates directly from the retriever to guarantee 
schema compliance and preserve recall.



3. Prompt Design & Intent Management 
Rather than using a complex multi-turn state machine, we manage conversation state 
inside a single stateful prompt: 
• Turn-Awareness: The system injection updates the current turn count 
(e.g., Current user turn: 6 of 8). When the conversation approaches the cap, the 
system instructions strictly force the LLM to skip clarification, output its final 
recommendations, and set end_of_conversation: true. 
• Dynamic Intent Modes: 
o clarify: Triggers when key constraints are missing. Catalog context is still 
retrieved but recommendations are withheld. 
o recommend / refine: Triggers when enough parameters are gathered or 
when the user updates/refines a previous shortlist. 
o compare: Synthesizes comparisons of named tests side-by-side using 
catalog details. 
o refuse: Blocks off-topic queries (legal advice, salary benchmarks, prompt 
injections) on the last turn without breaking the session. 
• User Correction Handling: Explicit system instructions enforce that newer 
statements (e.g., "Actually, I want senior, not junior") override previous history. 



4. What Didn't Work & Iterative Improvements 
During testing, we discovered and resolved several failure modes: 
• Heuristic Intent Misfires: Simple regex matched "sales" or "compensation 
manager" as off-topic because of keywords like "pay" or "compensation". We 
refined regex patterns to target only the last user turn and added contextual 
exclusions. 
• LLM URL/Name Hallucination: Despite prompts, LLMs sometimes fabricated 
URLs or generalized names. We implemented a post-processing validation layer 
that performs exact, substring, and token-overlap fuzzy matching 
against catalog.json to guarantee all returned assets are valid. 
• Rate-Limiting (429): Groq free-tier rate limits triggered during rapid automated 
replays. We resolved this by implementing exponential backoff retry logic (2s, 
4s, 8s delays) on the LLM client wrapper and verifying fallbacks to Gemini.
