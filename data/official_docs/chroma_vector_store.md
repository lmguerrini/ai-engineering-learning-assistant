# Chroma Vector Store

- **Official source**: https://docs.trychroma.com/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Storing and querying document embeddings for similarity search.
- Building RAG pipelines that need fast local vector retrieval.
- Prototyping vector search without external infrastructure.

## Key Concepts

### Client Types

- `chromadb.Client()` — ephemeral in-memory client (testing/prototyping).
- `chromadb.PersistentClient(path="./data")` — persistent local storage.
- `chromadb.HttpClient(host, port)` — remote Chroma server client.
- Persistent client stores data in SQLite + HNSW index files.

### Collections

- `client.get_or_create_collection(name, metadata)` — idempotent creation.
- Collections hold documents, embeddings, metadata, and IDs.
- `metadata={"hnsw:space": "cosine"}` sets distance metric (cosine, l2, ip).
- `collection.count()` returns number of documents in the collection.

### Adding Documents

- `collection.add(ids, documents, embeddings, metadatas)` — batch insert.
- IDs must be unique strings; duplicates are silently ignored.
- Embeddings are float arrays matching the model dimension.
- Metadata is a flat dict of string/number/bool values per document.

### Querying

- `collection.query(query_embeddings, n_results, include)` — similarity search.
- `include=["documents", "metadatas", "distances"]` controls returned fields.
- Results are nested lists: `results["documents"][0]` for first query.
- Distances are cosine distances (lower = more similar for cosine space).

### Filtering

- `where={"field": "value"}` filters on metadata before similarity search.
- `where_document={"$contains": "keyword"}` filters on document content.
- Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`.
- Combine filters: `{"$and": [{...}, {...}]}` or `{"$or": [{...}, {...}]}`.

## Practical Implementation Notes

- Use `PersistentClient` for production; `Client()` for unit tests.
- Always specify distance metric at collection creation time.
- Batch inserts for efficiency; avoid adding one document at a time.
- Store source filename and chunk index in metadata for traceability.
- Check `collection.count()` before querying to avoid empty-collection errors.

## Common Mistakes

- Using ephemeral client in production and losing data on restart.
- Not specifying `hnsw:space` — default is L2, not cosine.
- Passing duplicate IDs, which silently skips the insert.
- Querying with `n_results` larger than `collection.count()`.
- Storing complex nested objects in metadata (only flat dicts allowed).

## Related Project Usage

- `src/kb/vector_store.py`: Chroma persistent client, collection management, similarity search.
- `src/kb/ingestion.py`: Batch document addition to Chroma collections.
- `src/kb/retrieval.py`: Query interface over the Chroma vector store.
- `src/config.py`: `chroma_persist_dir` setting for persistence path.
