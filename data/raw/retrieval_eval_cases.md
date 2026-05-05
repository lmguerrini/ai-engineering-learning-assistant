# Retrieval Evaluation Cases

## Overview

This document defines evaluation queries to test the quality of retrieval in the knowledge base.

The goal is to verify:

- relevance of retrieved documents
- coverage of concepts
- robustness to ambiguity
- ability to handle complex queries

---

## Test Categories

### 1. Basic Concept Retrieval

Query:
"What is RAG?"

Expected:
- rag_basics.md

---

Query:
"What is an AI agent?"

Expected:
- ai_agents_intro.md

---

Query:
"What is prompt engineering?"

Expected:
- prompt_engineering.md

---

## 2. Multi-File Retrieval

Query:
"Difference between LangChain chains and LangGraph"

Expected:
- langchain_chains.md
- langgraph_advanced_agents.md

---

Query:
"How do agents use tools and function calling?"

Expected:
- function_calling_tools_mcp.md
- production_agents_chains.md

---

Query:
"Difference between short-term and long-term memory in agents"

Expected:
- long_term_memory_hitl.md

---

## 3. Agent Systems

Query:
"What is the ReAct pattern and how does it work?"

Expected:
- ai_agents_intro.md

---

Query:
"How to build a production-ready agent"

Expected:
- production_agents_chains.md

---

Query:
"What is agent state and why is it important?"

Expected:
- production_agents_chains.md
- langgraph_advanced_agents.md

---

## 4. Advanced Orchestration

Query:
"What are nodes and edges in LangGraph?"

Expected:
- langgraph_advanced_agents.md

---

Query:
"How does conditional routing work in LangGraph?"

Expected:
- langgraph_advanced_agents.md

---

Query:
"When should I use LangGraph instead of create_agent?"

Expected:
- langgraph_advanced_agents.md

---

## 5. Memory and Personalization

Query:
"How to implement long-term memory in an agent?"

Expected:
- long_term_memory_hitl.md

---

Query:
"What is human-in-the-loop in AI systems?"

Expected:
- long_term_memory_hitl.md

---

## 6. Tooling and APIs

Query:
"How does function calling work in AI agents?"

Expected:
- function_calling_tools_mcp.md

---

Query:
"What is MCP and how is it used?"

Expected:
- function_calling_tools_mcp.md

---

## 7. Modern Architectures

Query:
"What is an agent harness?"

Expected:
- agent_harness_frontier_technologies.md

---

Query:
"Difference between model and harness in AI systems"

Expected:
- agent_harness_frontier_technologies.md

---

Query:
"What are OpenClaw and OpenCode?"

Expected:
- agent_harness_frontier_technologies.md

---

## 8. Complex Queries (Multi-Hop)

Query:
"How to build a scalable AI agent with memory and tools?"

Expected:
- ai_agents_intro.md
- production_agents_chains.md
- long_term_memory_hitl.md

---

Query:
"How to design a robust agent system with fallback and state management?"

Expected:
- production_agents_chains.md
- langgraph_advanced_agents.md

---

Query:
"How to prevent context drift in long-running agents?"

Expected:
- agent_harness_frontier_technologies.md
- long_term_memory_hitl.md

---

## 9. Edge Cases

Query:
"Explain quantum computing"

Expected:
- No relevant documents (out-of-domain)

---

Query:
"What is the capital of France?"

Expected:
- No relevant documents (general knowledge)

---

## Evaluation Criteria

A good retrieval system should:

- return relevant documents only
- avoid irrelevant documents
- retrieve multiple documents when needed
- handle ambiguous queries correctly
- support multi-hop reasoning

---

## Notes

- These queries can be used manually or programmatically
- Extend this list as the KB grows
- Use them to debug retrieval issues and improve chunking strategy