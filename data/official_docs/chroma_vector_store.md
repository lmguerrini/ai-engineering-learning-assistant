# Chroma Vector Store

- **Official source**: https://docs.trychroma.com/
- **Last refreshed**: 2026-05-13
- **source_type**: official_docs
- **Versions**: `chromadb>=0.5`

## When to Use

- Storing and querying document embeddings for similarity search.
- Building RAG pipelines that need fast local vector retrieval.
- Prototyping vector search without external infrastructure.

## Key Concepts

### Client Types

```python
import chromadb

# Ephemeral — in-memory, for testing
client = chromadb.Client()

# Persistent — local storage, for development/production
client = chromadb.PersistentClient(path="./chroma_db")

# Remote — Chroma server
client = chromadb.HttpClient(host="localhost", port=8000)
```

Persistent client stores data in SQLite + HNSW index files at the specified path.

- `PersistentClient` creates `chroma.sqlite3` and `*.bin` index files in the specified directory.
- The path must be writable; concurrent writes from multiple processes are **not safe** with `PersistentClient`.
- `HttpClient` connects to a standalone Chroma server (supports multi-client access).
- Client instances are heavyweight — create once and reuse throughout the application lifecycle.

### Collections

```python
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},  # distance metric: cosine, l2, ip
)

collection.count()  # number of documents in the collection
```

- `get_or_create_collection` is idempotent — safe for repeated calls.
- Collections hold documents, embeddings, metadata, and IDs.
- **Always specify `hnsw:space`** — default is `l2`, not `cosine`.
- Distance metric cannot be changed after collection creation — recreate the collection to change it.

**HNSW index tuning**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hnsw:space` | `l2` | Distance function: `cosine`, `l2`, `ip` |
| `hnsw:construction_ef` | `100` | Build-time accuracy (higher = better index, slower build) |
| `hnsw:search_ef` | `10` | Query-time accuracy (higher = better recall, slower query) |
| `hnsw:M` | `16` | Max connections per node (higher = better recall, more memory) |

```python
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:search_ef": 50,   # increase for better recall at query time
        "hnsw:M": 32,           # increase for larger collections
    },
)
```

> **Note**: Increasing `hnsw:M` and `hnsw:construction_ef` improves recall but increases memory usage and build time. For collections under 10K documents, defaults are generally sufficient.

### Adding Documents

```python
collection.add(
    ids=["doc_001", "doc_002"],
    documents=["RAG combines retrieval with generation...", "Agents use tools..."],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],  # optional if embedding_function set
    metadatas=[
        {"source": "rag_basics.md", "chunk_index": 0},
        {"source": "ai_agents_intro.md", "chunk_index": 0},
    ],
)
```

- IDs must be unique strings; duplicates are silently ignored on `add`.
- Metadata values must be flat: `str`, `int`, `float`, or `bool` only.
- Use `collection.upsert(...)` to update existing documents by ID.
- `collection.update(...)` modifies existing documents; raises error if ID doesn't exist.
- Batch operations are significantly faster than single-document calls — group inserts into batches of 100–1000.

**ID generation strategies**:

```python
import hashlib

# Deterministic IDs based on content (enables deduplication)
def make_chunk_id(source: str, chunk_index: int) -> str:
    return hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()

# Or use source + index directly
ids = [f"{doc.metadata['source']}__chunk_{i}" for i, doc in enumerate(chunks)]
```

- Deterministic IDs enable idempotent re-ingestion — `upsert` with the same ID updates in place.
- Random UUIDs prevent deduplication; prefer content-based IDs for reproducible pipelines.

### Querying

```python
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)

# Results are nested lists (one per query)
documents = results["documents"][0]    # list[str]
metadatas = results["metadatas"][0]    # list[dict]
distances = results["distances"][0]    # list[float] — lower = more similar (cosine)
```

- Distances for cosine space: `0.0` = identical, `2.0` = maximally dissimilar.
- Results are ordered by ascending distance (most similar first).
- `n_results` must be ≤ `collection.count()` — requesting more raises `NotEnoughElementsException`.
- `include` parameter controls what’s returned; omitting fields saves memory for large result sets.
- Query accepts multiple `query_embeddings` for batch similarity search.

**Distance interpretation guide**:

| Cosine Distance | Interpretation | Typical Action |
|----------------|----------------|----------------|
| 0.0 – 0.3 | High similarity | Strong match — include in context |
| 0.3 – 0.5 | Moderate similarity | May be relevant — include with lower priority |
| 0.5 – 1.0 | Low similarity | Likely irrelevant — consider filtering out |
| > 1.0 | Dissimilar | Not relevant — discard |

### Filtering

```python
# Metadata filter
results = collection.query(
    query_embeddings=[embedding],
    n_results=5,
    where={"source": "rag_basics.md"},
)

# Document content filter
results = collection.query(
    query_embeddings=[embedding],
    n_results=5,
    where_document={"$contains": "retrieval"},
)

# Compound filters
results = collection.query(
    query_embeddings=[embedding],
    n_results=5,
    where={"$and": [{"source": "rag_basics.md"}, {"chunk_index": {"$gte": 0}}]},
)
```

Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.

- `where` filters apply **before** similarity ranking — they reduce the candidate set.
- Compound filters with `$and`/`$or` support up to 10 conditions.
- `where_document` filters are substring-based — not suited for semantic filtering.
- Filtering on non-existent metadata keys returns no results (no error).

### Collection Management

```python
# List all collections
collections = client.list_collections()

# Delete a collection (irreversible)
client.delete_collection(name="old_knowledge_base")

# Get collection by name (raises if not found)
collection = client.get_collection(name="knowledge_base")

# Peek at first N items (useful for debugging)
sample = collection.peek(limit=5)
print(sample["documents"], sample["metadatas"])

# Delete specific documents by ID
collection.delete(ids=["doc_001", "doc_002"])

# Delete by metadata filter
collection.delete(where={"source": "deprecated_file.md"})
```

- `delete_collection` removes all data and indexes — cannot be undone.
- `collection.peek()` returns items in insertion order; useful for sanity checks during development.

## Advanced Patterns

### Re-ingestion & Collection Reset

```python
def reingest_collection(client, name: str, documents, embeddings, metadatas, ids):
    """Drop and recreate collection for clean re-ingestion."""
    try:
        client.delete_collection(name=name)
    except ValueError:
        pass  # collection doesn't exist yet
    
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Batch insert in chunks of 500
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
    return collection
```

- Re-ingestion is preferred over incremental updates when KB content changes significantly.
- Batch size of 500 balances memory usage and insertion speed.

### Persistence & Backup

- `PersistentClient` data lives in `chroma.sqlite3` + `*.bin` files in the persist directory.
- To back up: copy the entire persist directory while the client is not actively writing.
- To migrate: copy the persist directory to a new location; update `path` in the client constructor.
- Chroma does not support concurrent writes from multiple processes to the same persist directory.

### Embedding Functions

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_fn = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

collection = client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,  # auto-embed on add/query
)

# Now documents are auto-embedded — no need to pass embeddings explicitly
collection.add(ids=["doc_001"], documents=["RAG combines retrieval with generation..."])
results = collection.query(query_texts=["What is RAG?"], n_results=5)
```

- Using `embedding_function` on the collection auto-embeds documents on `add` and queries on `query`.
- Ensure the same embedding function is used for both ingestion and querying — mismatched embeddings produce meaningless distances.
- External embedding (pre-computed) gives more control over batching and error handling.

## Practical Implementation Notes

- Use `PersistentClient` for production; `Client()` for unit tests.
- Always specify `hnsw:space` at collection creation — cannot be changed later.
- Batch inserts for efficiency; avoid adding one document at a time.
- Store source filename and chunk index in metadata for traceability.
- Check `collection.count()` before querying to handle empty-collection edge cases.
- Use `collection.get(ids=[...])` to verify specific documents exist after ingestion.
- Monitor persist directory size — HNSW indexes grow with document count and `M` parameter.
- For collections > 100K documents, consider increasing `hnsw:search_ef` for maintained recall quality.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `NotEnoughElementsException` | `n_results` > `collection.count()` | Check count before querying; use `min(n_results, count)` |
| Poor retrieval quality | Wrong distance metric (L2 vs cosine) | Verify `hnsw:space` matches embedding model expectations |
| Duplicate documents in collection | Non-deterministic IDs on re-ingestion | Use content-based deterministic IDs; use `upsert` instead of `add` |
| Slow queries on large collections | Low `hnsw:search_ef` | Increase `search_ef` (e.g., 50–100); trade speed for recall |
| `sqlite3.OperationalError: database is locked` | Concurrent writes from multiple processes | Use `HttpClient` with Chroma server for multi-process access |
| Metadata filter returns empty results | Key doesn’t exist in stored metadata | Verify metadata keys with `collection.peek()`; keys are case-sensitive |
| Persist directory grows large | Many documents with high `M` parameter | Lower `M`; or archive old collections |

## Common Mistakes

- Using ephemeral `Client()` in production — data is lost on restart.
- Not specifying `hnsw:space` — default is L2, not cosine.
- Passing duplicate IDs to `add()` — silently skips the insert.
- Querying with `n_results` larger than `collection.count()` — raises an error.
- Storing nested dicts or lists in metadata — only flat scalar values allowed.
- Mixing embedding models between ingestion and querying — distances become meaningless.
- Not batching inserts — single-document `add()` calls are orders of magnitude slower.
- Assuming `add()` updates existing documents — it silently skips duplicates; use `upsert()`.

## Related Project Usage

- `src/kb/vector_store.py`: Chroma persistent client, collection management, similarity search.
- `src/kb/ingestion.py`: Batch document addition to Chroma collections.
- `src/kb/retrieval.py`: Query interface over the Chroma vector store.
- `src/config.py`: `chroma_persist_dir` setting for persistence path.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://docs.trychroma.com/docs/overview/introduction

```
Skip to main content Chroma Docs home page Search... ⌘ K Ask AI 27k 11k 29k Dashboard Dashboard Search... Navigation Overview Introduction Docs Chroma Cloud Guides Integrations Reference Overview Introduction Getting Started Run Chroma Chroma Clients Client-Server Mode Collections Manage Collections Add Data Update Data Delete Data Configure Collections Querying Collections Query and Get Metadata Filtering Full Text Search Embeddings Embedding Functions Multimodal Embeddings CLI Installing the CLI Run a Chroma Server Data Management Cloud Other Other Open Source Migration Troubleshooting Overview Introduction Copy page Chroma is the open-source data infrastructure for AI. It comes with everything you need to get started built-in. Copy page Documentation Index Fetch the complete documentation index at: https://docs.trychroma.com/llms.txt Use this file to discover all available pages be...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
