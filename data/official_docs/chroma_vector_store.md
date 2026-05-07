# Chroma Vector Store

- **Official source**: https://docs.trychroma.com/
- **Last refreshed**: 2025-05-05
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

## Practical Implementation Notes

- Use `PersistentClient` for production; `Client()` for unit tests.
- Always specify `hnsw:space` at collection creation — cannot be changed later.
- Batch inserts for efficiency; avoid adding one document at a time.
- Store source filename and chunk index in metadata for traceability.
- Check `collection.count()` before querying to handle empty-collection edge cases.

## Common Mistakes

- Using ephemeral `Client()` in production — data is lost on restart.
- Not specifying `hnsw:space` — default is L2, not cosine.
- Passing duplicate IDs to `add()` — silently skips the insert.
- Querying with `n_results` larger than `collection.count()` — raises an error.
- Storing nested dicts or lists in metadata — only flat scalar values allowed.

## Related Project Usage

- `src/kb/vector_store.py`: Chroma persistent client, collection management, similarity search.
- `src/kb/ingestion.py`: Batch document addition to Chroma collections.
- `src/kb/retrieval.py`: Query interface over the Chroma vector store.
- `src/config.py`: `chroma_persist_dir` setting for persistence path.
