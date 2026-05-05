# Topic: Introduction to AI Agents
# Sprint: 3
# Part: 1
# Tags: ai-agents, react, reason-act, agentic-rag, langsmith, state, tools, pipelines-vs-agents, agent-architectures, multi-agent-systems

## Overview
AI agents are systems that can observe state, choose tools, act, and adapt based on intermediate results.

Unlike fixed pipelines, agents dynamically decide the next step based on current context, observations, errors, retrieved information, and available tools.

This section introduces the ReAct pattern, compares pipelines with agents, explains agentic RAG, and surveys common agent architectures.

## Key Concepts

- **AI Agent**  
  A software system that interacts with its environment and performs self-directed actions to achieve predefined goals.

- **State**  
  The current context used for decision-making, including history, retrieved documents, tool results, errors, and constraints.

- **Tools**  
  External capabilities such as APIs, databases, monitoring tools, calculators, applications, or other models.

- **ReAct Pattern**  
  A Reason + Act loop where the agent reasons, acts, observes, and updates state.

- **Pipeline**  
  A fixed sequence of predefined steps chosen by the developer.

- **Agent Loop**  
  A dynamic loop where the next step is selected based on observations.

- **Normal RAG**  
  A retrieve-once-then-generate pipeline.

- **Agentic RAG**  
  A RAG system where retrieval is treated as a tool the agent can invoke, refine, repeat, combine with other tools, or skip.

- **Goal-Based Agent**  
  An agent that selects actions based on an explicit objective.

- **Multi-Agent System**  
  A system where multiple specialized agents coordinate to solve a task.

- **LangSmith Trace**  
  An inspectable record of an agent workflow, useful for debugging Thought → Action → Observation cycles.

## How It Works

### 1. Recognize Pipeline Limits
Fixed pipelines work when the process is known in advance.

They become weak when tasks require:
- live data
- adaptive strategy changes
- dynamic tool selection
- retries
- escalation
- multi-step investigation

### 2. Define Agent Ingredients
An agent combines:

- LLM → reasoning and decision-making
- State → current context
- Tools → external actions and data access

### 3. Run the ReAct Loop
Typical ReAct flow:

1. observe current state
2. reason about the next step
3. act by calling a tool or stopping
4. observe tool result
5. update state
6. repeat until completion

### 4. Differentiate Tool Calls from Agents
A simple tool call follows:

LLM → Tool → LLM response

An agent can follow:

LLM → Tool → Observation → LLM → Tool → Observation → Final answer

The difference is dynamic control and iterative decision-making.

### 5. Compare Normal RAG and Agentic RAG
Normal RAG:
1. retrieve documents once
2. insert context into prompt
3. generate answer

Agentic RAG:
1. decide whether retrieval is needed
2. retrieve documents
3. evaluate retrieval quality
4. reformulate query if needed
5. retrieve again
6. combine retrieval with other tools
7. generate final answer when enough evidence exists

### 6. Understand Agent Architecture Spectrum
Agent architectures include:

- simple reflex agents
- model-based reflex agents
- goal-based agents
- learning agents
- utility-based agents
- multi-agent systems

In this sprint, the most relevant patterns are goal-based agents and multi-step agent workflows.

## Example

### Critical Support Ticket Example
A payment API fails in production.

A simple RAG bot can:
- retrieve troubleshooting documents
- generate a generic answer

An AI agent can:
- inspect previous diagnostics
- check live API status
- decide whether escalation is needed
- create a priority support ticket
- adapt if the first solution fails

### Normal RAG vs Agentic RAG
Normal RAG:
- query → retrieve once → answer

Agentic RAG:
- query → retrieve → evaluate → refine → retrieve again → answer

### Architecture Spectrum Examples
- Stateless chatbot → simple reflex
- Chatbot with history → model-based reflex
- ReAct workflow with objective → goal-based agent
- Multiple specialized workflows → multi-agent-like architecture

## When to Use

- **Pipeline**
  - predictable workflow
  - fixed steps
  - low uncertainty
  - simple automation

- **AI Agent**
  - unpredictable tasks
  - iterative refinement
  - dynamic tool choice
  - live system interaction

- **Agentic RAG**
  - retrieval may need refinement
  - sources may be insufficient
  - multiple retrieval passes may be needed
  - retrieval combines with other tools

- **Goal-Based Agents**
  - explicit objective
  - action selection required

- **Multi-Agent Systems**
  - specialized roles must coordinate

## Common Mistakes

- **Using agents for simple problems**
  - If steps are always known, a pipeline is simpler and safer.

- **Confusing tool use with agency**
  - A single hardcoded tool call is not an agent.

- **No iteration limits**
  - Agents can loop, repeat calls, or exhaust token budgets.

- **Weak tool reliability**
  - Agent quality depends on tool quality.

- **Ignoring security and privacy**
  - Agents with live tools can create real operational risks.

- **Assuming agents are transparent**
  - Multi-step workflows need logs, traces, and observability.

- **Expecting human judgment**
  - Agents may miss ethical nuance or hidden constraints.

## Related Concepts

- Prompt Engineering  
- Function Calling  
- Tool Use  
- Checkpointers and `thread_id`  
- LangSmith  
- LangGraph  
- Human-in-the-Loop  
- Long-Term Memory  
- Agent Architectures  
- Agentic RAG  