# Topic: Retrieval-Augmented Generation (RAG)
# Sprint: 2
# Part: 2
# Tags: rag, retrieval, vector-search, embeddings, knowledge-base, augmented-generation, chromadb, bm25, hybrid-search, semantic-search

## Overview
Retrieval-Augmented Generation (RAG) improves LLM accuracy by retrieving relevant external information at query time and using it to ground generation.

Instead of relying only on model training data, RAG connects an LLM to a knowledge base containing documents, notes, databases, or other trusted sources.

RAG is especially useful when answers must depend on private, recent, domain-specific, or verifiable information.

## Key Concepts

- **Retrieval-Augmented Generation (RAG)**  
  A pattern that combines retrieval from external sources with LLM generation.

- **Knowledge Base**  
  A searchable collection of external documents prepared for retrieval.

- **Chunking**  
  Splitting large documents into smaller text segments so retrieval can return focused context.

- **Embeddings**  
  Vector representations of text that capture semantic meaning.

- **Embedding Models**  
  Models that convert text into embeddings for semantic search.

- **Keyword Search**  
  Exact or term-based retrieval using methods such as BM25 or TF-IDF.

- **Vector Search**  
  Semantic retrieval based on similarity between query embeddings and document embeddings.

- **Hybrid Search**  
  Combines keyword search and vector search.

- **Augmented Generation**  
  The step where retrieved chunks are inserted into the prompt before generation.

- **Vector Database**  
  A storage system optimized for embeddings and similarity search.

- **ChromaDB**  
  A lightweight open-source vector database used for storing and retrieving embeddings.

- **Cosine Similarity**  
  A common way to measure similarity between vectors.

## How It Works

### 1. Build the Knowledge Base
A RAG system starts by preparing documents.

Steps:
1. collect documents
2. clean and preprocess text
3. split documents into chunks
4. create embeddings for chunks
5. store chunks and embeddings in a vector database

### 2. Receive the User Query
The user asks a question in natural language.

Example:
"What are the main limitations of LLMs?"

### 3. Retrieve Relevant Context
The system searches the knowledge base.

Retrieval methods:
- keyword search
- vector search
- hybrid search

### 4. Rank by Similarity
For vector search:
- embed the query
- compare query embedding with stored document embeddings
- return nearest matching chunks

Common similarity metrics:
- cosine similarity
- Euclidean distance
- dot product

### 5. Augment the Prompt
The retrieved chunks are added to the prompt as context.

The LLM receives:
- user query
- retrieved context
- instructions

### 6. Generate Grounded Answer
The LLM produces an answer based on retrieved information.

This improves:
- factuality
- domain relevance
- access to private or recent knowledge
- answer traceability

## Example

### Journalism Analogy
A journalist gathers sources before writing an article.

RAG follows the same structure:
1. retrieve evidence
2. use evidence to generate an answer

### Quiz-from-Notes Example
A chatbot creates quizzes from personal notes.

Flow:
1. notes are stored in a knowledge base
2. relevant notes are retrieved
3. retrieved notes are passed to the LLM
4. LLM generates quiz questions grounded in the notes

### Medical Side-Effects Example
User asks:
"What are the side effects of medication X?"

Retrieved context:
"Medication X may cause nausea, dizziness, headaches, and rare blurred vision."

The LLM uses this context to produce a grounded answer.

### Retrieval Method Examples
- Error code `E-4012` → keyword search works well
- "How can I make my app faster?" → vector search works well
- Medical/legal queries → hybrid search is often better

## When to Use

- **RAG**
  - private documents
  - recent information
  - domain-specific knowledge
  - grounded answers

- **Keyword Search**
  - exact identifiers
  - product codes
  - error messages
  - legal or medical terms

- **Vector Search**
  - conceptual questions
  - semantic similarity
  - synonyms
  - meaning-based search

- **Hybrid Search**
  - production systems with mixed query types
  - cases requiring exact terms and semantic meaning

- **Vector Databases**
  - large embedding collections
  - fast semantic retrieval

- **ChromaDB**
  - lightweight or mid-sized RAG projects
  - prototyping
  - semantic search
  - knowledge retrieval

## Common Mistakes

- **Assuming LLMs know everything**
  - LLMs do not know private documents or recent data unless retrieved.

- **Skipping preprocessing**
  - Poor document cleaning and chunking lead to weak retrieval.

- **Using the wrong retrieval method**
  - Keyword search and vector search solve different problems.

- **Ignoring hybrid retrieval**
  - Many real systems need both exact and semantic matching.

- **Misunderstanding embeddings**
  - Embeddings are vector representations, not raw text.

- **Ignoring indexing and similarity metrics**
  - Efficient retrieval depends on proper vector indexing and similarity search.

- **Treating vector databases like normal SQL databases**
  - Vector databases are designed for similarity search and metadata filtering.

## Related Concepts

- LangChain and Chains  
- Embeddings  
- Knowledge Base Ingestion  
- BM25 and TF-IDF  
- Hybrid Retrieval  
- Cosine Similarity  
- Vector Databases  
- ChromaDB  
- Augmented Generation  
- Agentic RAG  