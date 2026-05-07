import json
import re

from config.settings import GROQ_API_KEY
from core.vector_database import (
    cross_encoder,
    find_similar_chunks,
)
from groq import Groq
from utils.logger import logger

client = Groq(api_key=GROQ_API_KEY)

# Cross-encoder scores are logits (roughly -10 to +10), not cosine similarities.
# A score below this means the best chunk is a weak match for the query.
CE_LOW_CONFIDENCE_THRESHOLD = 1.0

# Number of chunks retrieved per sub-query in multi-query mode.
# Total candidates before reranking = MULTI_QUERY_N_RESULTS * number of sub-queries,
# capped at MULTI_QUERY_CANDIDATE_CAP before building the LLM context.
MULTI_QUERY_N_RESULTS = 5
MULTI_QUERY_CANDIDATE_CAP = 20

# Number of chunks retrieved for narrow single-pass queries.
SINGLE_QUERY_N_RESULTS = 6

SYSTEM_PROMPT = """You are a precise document assistant.
Answer the user's question using ONLY the numbered context passages provided.
For every claim in your answer, cite the passage number like [1] or [2].
When a passage contains a formula, equation, or algorithm, or code you MUST include and explain it — do not summarize or skip it.
Read ALL provided passages carefully before concluding anything is missing.
Only state that information is absent if you have checked every passage and it is genuinely not there.
Do not make up information or use knowledge outside the provided context."""

HYDE_PROMPT = """You are a technical document expert.
Given the question below, write a short hypothetical passage (2-3 sentences) that would appear in a technical paper or documentation and would directly answer it.
Be specific: include relevant formulas, variable names, numerical values, or technical terms if the question calls for them.
Write only the passage text. No explanation, no preamble, no meta-commentary."""

DECOMPOSE_PROMPT = """You are a search query decomposer.
Given a user question, output 3 to 4 short focused search queries that together cover all distinct sub-topics of the question.
Each query should target one specific aspect and be 3 to 6 words long.
Do not overlap or repeat the same concept across queries — each must target a different aspect.
Respond with a JSON object containing a single key 'queries' whose value is an array of strings.
No explanation, no preamble, no markdown.
Example: {"queries": ["encoder stack transformer", "decoder stack transformer", "multi-head attention mechanism", "positional encoding transformer"]}"""

# Smaller model for HyDE and decomposition — both only need short focused outputs.
# Reduces latency and avoids burning the 70b quota on retrieval-side calls.
HYDE_MODEL = "llama-3.1-8b-instant"
ANSWER_MODEL = "openai/gpt-oss-120b"

# Keywords that signal a broad question needing multi-query decomposition.
# Narrow factual lookups ("what is the learning rate?") go through single-pass retrieval.
_BROAD_QUERY_TRIGGERS = {
    "explain",
    "describe",
    "overview",
    "summarize",
    "summary",
    "components",
    "architecture",
    "how does",
    "how do",
    "what are",
    "walk me through",
    "tell me about",
    "discuss",
    "elaborate",
    "compare",
    "versus",
    "vs",
    "difference between",
    "contrast",
    "pros and cons",
    "advantages",
    "disadvantages",
    "tradeoffs",
    "trade-offs",
}


def _is_broad_query(query: str) -> bool:
    """
    Heuristic classifier: returns True when the query likely requires chunks
    from multiple distinct sections of the document.

    Broad queries use synthesis verbs or ask about multi-part concepts.
    Narrow queries ask for a single specific fact or value.
    """
    q_lower = query.lower()
    return any(trigger in q_lower for trigger in _BROAD_QUERY_TRIGGERS)


def _generate_hypothetical_answer(query: str) -> tuple[str, bool]:
    """
    Calls Groq to generate a short hypothetical passage that would answer the query.

    Returns (hypothetical_passage, hyde_succeeded).
    Falls back to the raw query string if generation fails so retrieval
    always has something to work with.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": HYDE_PROMPT},
                {"role": "user", "content": query},
            ],
            model=HYDE_MODEL,
            temperature=0.5,
            max_tokens=150,
        )
        hypothetical = response.choices[0].message.content.strip()
        logger.debug(f"HyDE hypothetical: {hypothetical[:120]}")
        return hypothetical, True
    except Exception as e:
        logger.warning(f"HyDE generation failed, falling back to raw query: {e}")
        return query, False


def _decompose_query(query: str) -> list[str]:
    """
    Decomposes a broad query into 3 to 4 focused sub-queries, each targeting
    a distinct sub-topic so per-sub-query retrieval pulls different document slices.

    Uses Groq's json_object response_format to guarantee structurally valid JSON,
    eliminating the class of failures where the model emits malformed JSON or
    markdown fences around the output.

    Falls back to [query] if the call fails or the output does not contain a valid
    'queries' list of strings, giving single-pass behaviour as a safe default.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": DECOMPOSE_PROMPT},
                {"role": "user", "content": query},
            ],
            model=HYDE_MODEL,
            temperature=0.3,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        sub_queries = parsed.get("queries", [])
        if (
            isinstance(sub_queries, list)
            and all(isinstance(q, str) for q in sub_queries)
            and sub_queries
        ):
            logger.debug(
                f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}"
            )
            return sub_queries[:4]
        logger.warning(f"Decomposition returned unexpected structure: {parsed}")
    except Exception as e:
        logger.warning(f"Query decomposition failed, using original query: {e}")
    return [query]


def _retrieve_for_sub_query(sub_q: str) -> list[dict]:
    """
    Run HyDE generation and retrieval for a single sub-query.
    """
    hypothetical, _ = _generate_hypothetical_answer(sub_q)
    return find_similar_chunks(
        hyde_query=hypothetical,
        original_query=sub_q,
        n_results=MULTI_QUERY_N_RESULTS,
    )


def _retrieve_multi_query(query: str) -> list[dict]:
    """
    Decompose the query, then run HyDE + retrieval for all sub-queries
    sequentially. Deduplicates the merged candidate pool by chunk identity,
    then reranks against the original query with the cross-encoder.

    Deduplication key: (doc_name, page_number, first 80 chars of chunk_text).
    Robust to minor whitespace differences across retrieval paths.
    """
    sub_queries = _decompose_query(query)

    seen: set[tuple] = set()
    candidates: list[dict] = []
    for sub_q in sub_queries:
        results = _retrieve_for_sub_query(sub_q)
        for chunk in results:
            key = (chunk["doc_name"], chunk["page_number"], chunk["chunk_text"][:80])
            if key not in seen:
                seen.add(key)
                candidates.append(chunk)

    logger.debug(
        f"Multi-query retrieved {len(candidates)} unique candidates "
        f"across {len(sub_queries)} sub-queries."
    )

    if not candidates:
        return candidates

    # Rerank the merged pool against the original query.
    # Per-sub-query ce_scores from find_similar_chunks reflect sub-query relevance,
    # not the full question's relevance — a fresh rerank is required here.
    pairs = [(query, chunk["chunk_text"]) for chunk in candidates]
    scores: list[float] = cross_encoder.predict(pairs).tolist()

    for chunk, score in zip(candidates, scores):
        chunk["ce_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["ce_score"], reverse=True)
    return reranked[:MULTI_QUERY_CANDIDATE_CAP]


def _retrieve_single_query(query: str) -> tuple[list[dict], bool]:
    """
    Standard single-pass HyDE retrieval for narrow factual questions.
    Returns (chunks, hyde_succeeded).
    """
    hypothetical, hyde_succeeded = _generate_hypothetical_answer(query)
    chunks = find_similar_chunks(
        hyde_query=hypothetical,
        original_query=query,
        n_results=SINGLE_QUERY_N_RESULTS,
    )
    return chunks, hyde_succeeded


def build_context_from_chunks(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        block_type = chunk.get("block_type", "paragraph")
        if block_type == "table":
            type_label = "Table"
        elif block_type == "code":
            type_label = "Code"
        else:
            type_label = "Passage"
        meta = (
            f"[{i}] {type_label} — "
            f"Source: {chunk['doc_name']}, "
            f"Page {chunk['page_number']}, "
            f"Section: {chunk['topic']}"
        )
        context_parts.append(f"{meta}\n{chunk['chunk_text']}")
    return "\n\n".join(context_parts)


def is_low_confidence(chunks: list[dict]) -> bool:
    """
    Use cross-encoder score (ce_score) as the confidence signal since
    chunks are reranked by the cross-encoder, not vector similarity.
    Falls back to vector score if ce_score is absent (defensive).
    """
    if not chunks:
        return True
    top = chunks[0]
    if "ce_score" in top:
        return top["ce_score"] < CE_LOW_CONFIDENCE_THRESHOLD
    return top.get("score", 0.0) < 0.3  # Hard fall-back if ce score is not present


def generate_answer(query: str) -> dict:
    """
    Entry point called by the route handler.

    Broad queries go through multi-query decomposition with sequential retrieval.
    Narrow queries go through single-pass HyDE retrieval.
    """
    logger.debug(f"Generating answer for query: {query[:80]}")

    broad = _is_broad_query(query)

    if broad:
        chunks = _retrieve_multi_query(query)
        hyde_succeeded = (
            None  # HyDE runs per sub-query internally; not meaningful as a single bool
        )
    else:
        chunks, hyde_succeeded = _retrieve_single_query(query)

    low_confidence = is_low_confidence(chunks)
    context = build_context_from_chunks(chunks)

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}",
            },
        ],
        model=ANSWER_MODEL,
        temperature=0.2,
        max_tokens=1000,
    )
    raw_answer = completion.choices[0].message.content

    # reasoning modelsemit <think>...</think> blocks
    # before the actual answer — strip them so they never reach the client.
    answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()

    logger.debug(
        f"Answer generated. Broad: {broad}, Low confidence: {low_confidence}, "
        f"HyDE used: {hyde_succeeded}"
    )

    return {
        "answer": answer,
        "chunks": chunks,
        "low_confidence": low_confidence,
        "hyde_used": hyde_succeeded,
    }
