import asyncio
import json
import uuid
from pathlib import Path
from typing import List

from config.settings import VECTORSTORE_DIRECTORY
from core.document_processor import load_documents_from_paths, save_uploaded_file
from fastapi import UploadFile
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from utils.logger import logger

# Clients and models — loaded once at startup
qdrant_client = QdrantClient(path=VECTORSTORE_DIRECTORY)

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

COLLECTION_NAME = "documents"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_DIM = 768  # BAAI/bge-base-en-v1.5 output dimension
# Fallback confidence threshold used by llm_chain_factory.is_low_confidence when
# ce_score is absent from a chunk. In normal operation the cross-encoder always
# sets ce_score, so this threshold is a defensive fallback, not the primary signal.
LOW_CONFIDENCE_THRESHOLD = 0.3

# Path where the global BM25 state (IDF table + vocab) is persisted to disk
# so query-time sparse vectors use the same corpus-calibrated weights as ingest time.
_BM25_STATE_PATH = Path(VECTORSTORE_DIRECTORY) / "bm25_state.json"

# In-memory cache of the persisted BM25 state. Populated at ingest time and
# loaded from disk at query time.
# Structure: {"vocab": {term: int}, "idf": {term: float}, "avgdl": float, "k1": float, "b": float}
_bm25_state: dict | None = None


def _build_embed_text(topic: str, text: str) -> str:
    """
    Prepend topic to chunk text before embedding so the vector carries
    section context. Only the raw text is stored in payload.
    """
    if topic and len(topic.split()) <= 12:
        return f"{topic}\n\n{text}"
    return text


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts, normalize_embeddings=True).tolist()


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenizer — consistent between index and query time."""
    return text.lower().split()


def _build_bm25_index(corpus_tokens: list[list[str]]) -> tuple[BM25Okapi, dict]:
    """
    Build a BM25Okapi index over all corpus token lists and derive a stable
    vocabulary mapping term -> integer index.

    The vocab is sorted so indices are deterministic across runs as long as
    the corpus is the same. Used at ingest time; the resulting IDF table and
    vocab are persisted to disk via _save_bm25_state so query time can reuse
    the same weights.
    """
    bm25 = BM25Okapi(corpus_tokens)
    vocab = {
        term: idx
        for idx, term in enumerate(
            sorted(set(t for tokens in corpus_tokens for t in tokens))
        )
    }
    return bm25, vocab


def _save_bm25_state(bm25: BM25Okapi, vocab: dict) -> None:
    """
    Persist the corpus-level BM25 parameters to disk so that query-time
    sparse vectors are calibrated against the same IDF table used during ingest.

    Serialised fields:
      vocab   — term to integer index mapping for Qdrant sparse vector indices
      idf     — per-term IDF scores from BM25Okapi.idf
      avgdl   — average document length across the corpus
      k1, b   — BM25 hyperparameters (defaults 1.5 and 0.75 in rank_bm25)
    """
    global _bm25_state
    state = {
        "vocab": vocab,
        "idf": bm25.idf,
        "avgdl": bm25.avgdl,
        "k1": bm25.k1,
        "b": bm25.b,
    }
    _BM25_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BM25_STATE_PATH.write_text(json.dumps(state))
    _bm25_state = state
    logger.info(f"BM25 state saved to {_BM25_STATE_PATH} ({len(vocab)} terms).")


def _load_bm25_state() -> dict | None:
    """
    Load BM25 state from disk into the in-memory cache.
    Returns the state dict, or None if no state has been persisted yet
    (i.e. no documents have been ingested).
    """
    global _bm25_state
    if _bm25_state is not None:
        return _bm25_state
    if not _BM25_STATE_PATH.exists():
        return None
    _bm25_state = json.loads(_BM25_STATE_PATH.read_text())
    logger.info(
        f"BM25 state loaded from {_BM25_STATE_PATH} ({len(_bm25_state['vocab'])} terms)."
    )
    return _bm25_state


def _sparse_vector_from_state(state: dict, tokens: list[str]) -> SparseVector:
    """
    Compute a BM25 sparse vector for a token list using pre-built corpus state.

    Both document vectors (ingest) and query vectors (retrieval) use this
    function with the same state, ensuring IDF weights are consistent.

    The BM25 TF component formula used here matches BM25Okapi:
        tf_score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))

    Terms absent from the corpus vocab are silently skipped; they carry no
    signal because the corpus IDF table has no entry for them.

    Sentinel handling: if no terms survive scoring (e.g. a chunk that is pure
    punctuation after tokenization), a sentinel entry is placed at index
    len(vocab) — guaranteed outside the real vocab range — so the upsert
    never fails while also avoiding a collision with a real term.
    """
    vocab: dict = state["vocab"]
    idf: dict = state["idf"]
    avgdl: float = state["avgdl"]
    k1: float = state["k1"]
    b: float = state["b"]

    tf_map: dict[str, int] = {}
    for token in tokens:
        tf_map[token] = tf_map.get(token, 0) + 1

    doc_len = len(tokens)
    indices = []
    values = []

    for term, tf in tf_map.items():
        if term not in vocab:
            continue
        term_idf = idf.get(term, 0.0)
        if term_idf <= 0:
            continue
        tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        score = term_idf * tf_score
        if score > 0:
            indices.append(vocab[term])
            values.append(float(score))

    if not indices:
        # Sentinel placed at len(vocab) — outside the real vocab index range
        # so it cannot collide with any real term's index.
        return SparseVector(indices=[len(vocab)], values=[0.0])

    return SparseVector(indices=indices, values=values)


def collection_exists() -> bool:
    return any(
        c.name == COLLECTION_NAME for c in qdrant_client.get_collections().collections
    )


def initialize_vectorstore():
    """
    Create the collection with named dense + sparse vectors.
    If the collection already exists with the old single-vector schema,
    it will be used as-is — drop manually if a schema migration is needed.
    """
    try:
        if not collection_exists():
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
            )
            logger.info(
                f"Created Qdrant collection '{COLLECTION_NAME}' with dense + sparse vectors."
            )
        count = qdrant_client.count(COLLECTION_NAME).count
        logger.info(f"Vectorstore ready. Points in store: {count}")
    except Exception as e:
        logger.error(f"Vectorstore initialization failed: {e}")
        raise


def vectorstore_exists() -> bool:
    if not collection_exists():
        return False
    return qdrant_client.count(COLLECTION_NAME).count > 0


def get_collection_count() -> int:
    if not collection_exists():
        return 0
    return qdrant_client.count(COLLECTION_NAME).count


def _fetch_all_existing_chunk_texts() -> list[str]:
    """
    Scroll through all points in the collection and return every stored
    chunk_text. Used at ingest time to include already-ingested documents
    in the global BM25 index rebuild so IDF weights are always corpus-wide,
    not just scoped to the current upload batch.

    Uses Qdrant's scroll API with a page size of 500 to avoid loading the
    entire collection into memory at once. Only the payload is fetched —
    vectors are excluded to keep memory usage low.
    """
    if not collection_exists():
        return []

    texts = []
    offset = None

    while True:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            text = point.payload.get("chunk_text", "")
            if text:
                texts.append(text)
        if next_offset is None:
            break
        offset = next_offset

    logger.debug(
        f"Fetched {len(texts)} existing chunk texts from Qdrant for BM25 rebuild."
    )
    return texts


def _delete_existing_doc(doc_name: str):
    """
    Delete all points for a given doc_name before re-ingesting.
    Prevents duplicates if the same file is uploaded more than once.
    """
    existing_count = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_name))]
        ),
    ).count

    if existing_count > 0:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="doc_name", match=MatchValue(value=doc_name))
                    ]
                )
            ),
        )
        logger.warning(
            f"Replaced existing '{doc_name}' — deleted {existing_count} stale points before re-ingesting."
        )


def _ingest_sync(docs: list[dict], docs_by_source: dict[str, list]) -> None:
    """
    Synchronous core of the ingest pipeline. Runs in a thread pool via
    asyncio.to_thread so CPU-bound embedding and BM25 work never blocks
    the event loop.

    Steps:
      1. Delete stale points for each incoming file.
      2. Fetch all existing chunk texts from Qdrant (for corpus-wide BM25).
      3. Build a global BM25 index over existing + new chunks combined.
      4. Persist the BM25 state to disk.
      5. Compute dense embeddings and sparse BM25 vectors for new chunks.
      6. Upsert all new points into Qdrant.

    Existing points are NOT re-embedded — only their chunk texts are used
    to calibrate the IDF table. This keeps ingest time proportional to the
    size of the new upload, not the entire collection.
    """
    # Step 1: delete stale points
    for source in docs_by_source:
        _delete_existing_doc(source)

    # Step 2: fetch existing chunk texts for corpus-wide IDF calibration.
    # These are the texts already in Qdrant (after stale deletions above).
    existing_texts = _fetch_all_existing_chunk_texts()
    existing_tokens = [_tokenize(t) for t in existing_texts]

    # Step 3: build global BM25 over existing + new chunks combined.
    new_tokens = [_tokenize(d["text"]) for d in docs]
    all_corpus_tokens = existing_tokens + new_tokens
    bm25_index, vocab = _build_bm25_index(all_corpus_tokens)

    # Step 4: persist BM25 state for query-time use.
    _save_bm25_state(bm25_index, vocab)
    state = _load_bm25_state()

    # Step 5: dense embeddings + sparse vectors for new chunks only.
    all_points = []
    for source, source_docs in docs_by_source.items():
        embed_inputs = [
            _build_embed_text(d["metadata"]["topic"], d["text"]) for d in source_docs
        ]
        dense_embeddings = embed_texts(embed_inputs)

        for doc, dense_vec in zip(source_docs, dense_embeddings):
            doc_tokens = _tokenize(doc["text"])
            sparse_vec = _sparse_vector_from_state(state, doc_tokens)
            meta = doc["metadata"]
            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
                    payload={
                        "doc_name": meta["source"],
                        "page_number": meta["page"],
                        "topic": meta["topic"],
                        "block_type": meta["block_type"],
                        "chunk_text": doc["text"],
                    },
                )
            )

    # Step 6: upsert new points.
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    logger.info(
        f"Upserted {len(all_points)} points from {len(docs_by_source)} file(s). "
        f"BM25 index built over {len(all_corpus_tokens)} total chunks "
        f"({len(existing_tokens)} existing + {len(new_tokens)} new)."
    )


async def upsert_vectorstore_from_pdfs(uploaded_files: List[UploadFile]):
    """
    Async entry point for the ingest pipeline. File saving is awaited
    directly since it is already async. PDF parsing and all downstream
    CPU/IO work is offloaded to a thread pool via asyncio.to_thread.
    """
    file_paths = await save_uploaded_file(uploaded_files)

    # load_documents_from_paths is CPU-bound (PDF parsing + chunking).
    # Run it in a thread pool so it does not block the event loop.
    docs = await asyncio.to_thread(load_documents_from_paths, file_paths)

    if not docs:
        logger.warning("No blocks extracted from uploaded files.")
        return

    docs_by_source: dict[str, list] = {}
    for doc in docs:
        source = doc["metadata"]["source"]
        docs_by_source.setdefault(source, []).append(doc)

    # All remaining ingest work (BM25, embedding, Qdrant upsert) is blocking.
    # Offload the entire pipeline to a thread pool as a single unit to avoid
    # repeated thread-pool hand-offs between steps.
    await asyncio.to_thread(_ingest_sync, docs, docs_by_source)


def _cross_encoder_rerank(
    original_query: str, chunks: list[dict], top_k: int
) -> list[dict]:
    """
    Rerank candidates using a cross-encoder model.
    Takes the original user query (not HyDE hypothetical) so scores
    reflect true query-chunk relevance, not hypothetical-chunk similarity.
    """
    if not chunks:
        return chunks

    pairs = [(original_query, chunk["chunk_text"]) for chunk in chunks]
    scores = cross_encoder.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["ce_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["ce_score"], reverse=True)
    return reranked[:top_k]


def find_similar_chunks(
    hyde_query: str, original_query: str, n_results: int = 8
) -> list[dict]:
    """
    Hybrid retrieval: dense BGE search + sparse BM25 search fused via RRF,
    followed by cross-encoder reranking on the original user query.

    hyde_query     — HyDE hypothetical passage used for dense embedding search
    original_query — raw user query used for BM25 sparse search and cross-encoder reranking
    n_results      — final number of chunks returned after reranking

    This function is synchronous and must be called via asyncio.to_thread
    from async contexts (as done in llm_chain_factory._find_similar_chunks_sync).
    """
    if not vectorstore_exists():
        raise ValueError("No documents have been uploaded yet.")

    candidate_limit = (
        n_results * 3
    )  # fetch more candidates to give the reranker headroom

    # Dense query vector — use BGE instruction prefix with the HyDE hypothetical
    prefixed_query = (
        f"Represent this sentence for searching relevant passages: {hyde_query}"
    )
    dense_query_vec = embed_texts([prefixed_query])[0]

    # Sparse query vector — built from corpus IDF weights, not a toy single-doc index.
    # _load_bm25_state returns None only if no documents have ever been ingested,
    # which is already guarded above by vectorstore_exists().
    state = _load_bm25_state()
    query_tokens = _tokenize(original_query)
    if state and query_tokens:
        sparse_query_vec = _sparse_vector_from_state(state, query_tokens)
    else:
        # Fallback: empty query or missing state — use zero sentinel.
        # This path should not be reached in normal operation.
        vocab_size = len(state["vocab"]) if state else 0
        sparse_query_vec = SparseVector(indices=[vocab_size], values=[0.0])

    # Hybrid search with RRF fusion across dense and sparse legs
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_query_vec, using=DENSE_VECTOR_NAME, limit=candidate_limit
            ),
            Prefetch(
                query=sparse_query_vec, using=SPARSE_VECTOR_NAME, limit=candidate_limit
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=candidate_limit,
    ).points

    chunks = []
    for hit in results:
        payload = hit.payload
        chunks.append(
            {
                "doc_name": payload["doc_name"],
                "page_number": payload["page_number"],
                "topic": payload["topic"],
                "block_type": payload["block_type"],
                "chunk_text": payload["chunk_text"],
                "score": round(hit.score, 4),
            }
        )

    reranked = _cross_encoder_rerank(original_query, chunks, top_k=n_results)

    logger.debug(
        f"Hybrid search returned {len(chunks)} candidates, cross-encoder reranked to {len(reranked)}."
    )
    return reranked
