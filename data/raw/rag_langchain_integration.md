# Topic: Integrating RAG with LangChain
# Sprint: 2
# Part: 3
# Tags: rag, langchain, document-loaders, text-splitting, embeddings, vector-database, semantic-search, ragas, faithfulness, retrieval-evaluation

## Overview
Integrating RAG with LangChain means building a complete retrieval pipeline that loads documents, splits them into chunks, embeds them, stores them in a vector database, retrieves relevant context, and generates grounded answers with an LLM.

This section also introduces RAG evaluation with RAGAs, focusing on whether answers are faithful, relevant, and supported by retrieved context.

## Key Concepts

- **RAG Pipeline**  
  A workflow that prepares a knowledge base, retrieves relevant context, and uses the context for grounded generation.

- **Document Loader**  
  A component that extracts and standardizes content from sources such as PDFs, websites, databases, CSV files, JSON files, directories, or text files.

- **Text Splitter**  
  A component that divides large documents into smaller chunks suitable for retrieval and context windows.

- **Embeddings and Vector Storage**  
  The process of turning chunks into vectors and storing them in a vector database.

- **Semantic Search Engine**  
  A retrieval system that matches queries to documents by meaning rather than exact keywords.

- **RAGAs**  
  A framework for evaluating RAG systems using metrics and LLM-as-a-judge patterns.

- **Faithfulness**  
  Measures whether the generated answer is supported by the retrieved context.

- **Answer Relevancy**  
  Measures whether the answer addresses the original user question.

- **Context Precision**  
  Measures whether retrieved chunks are relevant and well ranked.

- **Context Recall**  
  Measures whether retrieval captured the information needed to answer fully.

- **Atomic Facts**  
  Small factual claims extracted from an answer for verification against retrieved context.

## How It Works

### 1. Load Source Documents
Raw knowledge sources are loaded from files, websites, databases, or directories.

LangChain document loaders convert different input formats into a standard document representation.

### 2. Split Documents into Chunks
Large documents are split into smaller chunks.

Good chunking helps:
- fit content into model context windows
- improve retrieval precision
- avoid returning entire irrelevant documents

Chunking can be based on:
- paragraphs
- sentences
- tokens
- characters
- document structure

### 3. Embed and Store Chunks
Each chunk is converted into an embedding.

The embeddings and chunk metadata are stored in a vector database.

### 4. Retrieve Relevant Chunks
When a user asks a question, the retriever searches for chunks that are semantically close to the query.

The retrieved chunks become context for generation.

### 5. Augment the Prompt
The prompt includes:
- user question
- retrieved context
- instructions for grounded answering

### 6. Generate the Answer
The LLM generates a response using the retrieved context.

A good RAG answer should:
- answer the question
- stay grounded in context
- avoid unsupported claims
- cite or expose sources when possible

### 7. Evaluate with RAGAs
RAGAs evaluates the RAG pipeline using:

- faithfulness
- answer relevancy
- context precision
- context recall

These metrics help identify whether failures come from retrieval, generation, or both.

### 8. Improve the Pipeline
Metric interpretation:

- Low faithfulness → answer contains unsupported claims
- Low answer relevancy → answer does not address the question
- Low context precision → retrieved chunks are noisy
- Low context recall → retriever missed needed information

## Example

### Chunking Example
A user manual can be split into sections:

- Setup Instructions
- Troubleshooting Steps
- Error Codes
- Maintenance

When a user asks about troubleshooting, the retriever can return only the relevant section.

### Embedding Example
Text chunk:
"To reset the router, press the back button for 10 seconds."

The embedding model converts it into a vector so it can be compared semantically with a query like:
"How do I restart my router?"

### RAGAs Hallucination Example
If the retrieved context says only:
"Sun Tzu wrote The Art of War."

But the answer says:
"Sun Tzu served as a general for 30 years."

The extra claim lowers faithfulness because it is not supported by the retrieved context.

### Faithfulness Nuance Example
Retrieved context:
"Albert Einstein was born on 14 March 1879 in Germany."

High-faithfulness answer:
"Einstein was born in Germany on 14 March 1879."

Lower-faithfulness answer:
"Einstein, who developed relativity, was born in Germany on 14 March 1879."

The second answer may be true, but the relativity claim is not supported by the provided context.

## When to Use

- **LangChain RAG**
  - Document Q&A
  - semantic search
  - knowledge assistants
  - grounded educational tools

- **Document Loaders**
  - When knowledge is stored across files, websites, or structured sources.

- **Text Splitters**
  - When raw documents are too large or unfocused for retrieval.

- **Vector Storage**
  - When semantic search is required.

- **RAGAs**
  - When you need to evaluate grounding, retrieval quality, and answer relevance.

- **Faithfulness Evaluation**
  - When hallucination risk matters.

## Common Mistakes

- **Skipping evaluation**
  - Without evaluation, hallucinations and retriever failures are hard to detect.

- **Poor chunking**
  - Bad chunk boundaries reduce retrieval quality.

- **Assuming retrieval guarantees truth**
  - The LLM can still hallucinate even with good context.

- **Confusing faithfulness with global truth**
  - Faithfulness means supported by retrieved context, not true in the outside world.

- **Using only one metric**
  - RAG quality depends on both retrieval and generation.

- **Ignoring examples behind metrics**
  - Metrics should be interpreted together with concrete failure cases.

- **Assuming RAGAs only works for Python apps**
  - RAGAs can evaluate outputs from other systems if data is exported.

## Related Concepts

- Retrieval-Augmented Generation (RAG)  
- LangChain  
- Document Loaders  
- Text Splitting  
- Embeddings  
- Vector Databases  
- Semantic Search  
- LLM-as-a-Judge  
- Faithfulness  
- Context Precision and Context Recall  
- Answer Relevancy  
- RAGAs  