# Topic: Introduction to Prompt Engineering
# Sprint: 1
# Part: 3
# Tags: prompt-engineering, system-user-assistant-roles, zero-shot, few-shot, chain-of-thought, structured-output, function-calling, llm-settings, temperature, top-p

## Overview
Prompt engineering is the core skill for effectively working with large language models (LLMs).

It involves designing clear, structured, and constrained prompts to guide model behavior, improve reliability, and produce useful outputs. Prompts act similarly to code: better inputs lead to better outputs.

This section covers prompting techniques, structured outputs, function calling, and model settings that influence behavior.

## Key Concepts

- **Prompt Engineering**  
  The practice of designing prompts to control LLM behavior and output quality.

- **Message Roles (System, User, Assistant)**  
  - System: defines behavior, tone, and rules  
  - User: provides input or task  
  - Assistant: generates output  

- **Stateless LLMs**  
  LLMs do not retain memory across calls; conversation history must be explicitly passed.

- **Zero-Shot Prompting**  
  Task is performed using instructions only, without examples.

- **Few-Shot Prompting**  
  Task is demonstrated through input-output examples.

- **Chain-of-Thought Prompting**  
  The model is guided to reason step by step.

- **Structured Output**  
  Output constrained to a defined schema (e.g., JSON).

- **Function Calling**  
  The model produces structured tool calls instead of free text.

- **Sampling Controls (Temperature, Top-p)**  
  Parameters controlling randomness and diversity.

- **Reasoning Effort**  
  Controls how much internal reasoning the model performs.

## How It Works

### 1. Message Structure
A prompt consists of role-based messages:

- System → defines behavior and constraints  
- User → defines the task  
- Assistant → returns the response  

Because LLMs are stateless, the full conversation must be included in each request.

### 2. Prompt Design Principles

Effective prompts include:

- Clear instructions  
- Context and constraints  
- Defined output format  
- Explicit steps when needed  
- Examples (if required)  

Prompting is iterative:
→ write → test → refine

### 3. Zero-Shot Prompting

- No examples provided  
- Works for simple or common tasks  

Good prompt includes:
- goal
- tone
- format
- constraints

### 4. Few-Shot Prompting

- Includes examples of input → output  
- Helps model learn patterns  

Best practices:
- include edge cases  
- keep examples consistent  
- avoid conflicting patterns  

### 5. Chain-of-Thought Prompting

- Encourages step-by-step reasoning  

Used for:
- math
- logic
- debugging
- planning  

Improves accuracy on complex tasks.

### 6. Output Types

- **Free Text**  
  Default natural language output  

- **Structured Output**  
  JSON-based schema (e.g., Pydantic, JSON Schema)  

- **Function Calling**  
  Model returns tool name + arguments  
  → system executes  
  → result returned to model  

### 7. LLM Settings

- **Temperature**  
  Lower → deterministic  
  Higher → creative  

- **Top-p**  
  Limits token selection to a probability mass  

- **Max Tokens**  
  Limits output length  

- **Frequency Penalty**  
  Reduces repetition  

- **Presence Penalty**  
  Encourages new topics  

- **Reasoning Effort**  
  Controls internal reasoning depth  

### 8. Prompting Workflow

1. Start with a simple prompt  
2. Evaluate output  
3. Add constraints or examples  
4. Adjust reasoning (if needed)  
5. Tune settings only if necessary  

## Example

### Zero-Shot vs Specific Prompt

Bad:
"Write a response to this email"

Better:
"Write a professional email declining a meeting politely, under 50 words, and suggest email updates instead."

### Few-Shot Example

Provide labeled examples:

Input → Output  
Pattern → New input  

Model learns format and applies it.

### Chain-of-Thought Example

Prompt:
"Analyze step by step and identify the bug."

Model:
- identifies intent  
- traces logic  
- finds error  

### Function Calling Example

User:
"What is 8 + 12?"

Model:
→ calls calculator function  
→ returns 20  

### Structured Output Example

Prompt:
"Extract the product name, price, and category from this review. Return JSON."

Output:
`{"product": "Wireless Mouse", "price": 29.99, "category": "Electronics"}`

This is reliable for downstream processing in pipelines and APIs.

## When to Use

- **Zero-Shot**
  - simple tasks
  - quick responses  

- **Few-Shot**
  - structured formats
  - classification  

- **Chain-of-Thought**
  - reasoning-heavy tasks  

- **Structured Output**
  - APIs and pipelines  

- **Function Calling**
  - tool integration  
  - agent systems  

- **Settings Tuning**
  - creativity vs precision trade-offs  

## Common Mistakes

- **Vague prompts**
  → unclear outputs  

- **Ignoring system message**
  → unstable behavior  

- **No structure**
  → model confusion  

- **Poor few-shot examples**
  → wrong patterns  

- **Skipping reasoning**
  → incorrect answers  

- **Over-tuning settings**
  → hard to debug  

- **Using sampling instead of fixing prompt**
  → unstable results  

## Best Practices

- Start with a clear system message that defines the assistant's role, tone, and constraints.
- Use structured output (JSON mode or Pydantic schemas) whenever the output must be parsed programmatically.
- Include few-shot examples for tasks where format consistency matters.
- Test prompts systematically with an evaluation dataset, not just a few manual examples.
- Keep prompts modular: separate instructions, context, and output format for easier iteration.
- Use chain-of-thought prompting for reasoning-heavy tasks; skip it for simple lookups.
- Set temperature to 0 for deterministic tasks (classification, extraction) and 0.7–1.0 for creative tasks.
- Version-control prompts alongside code to track changes and regressions.

## Related Concepts

- Message roles (system/user/assistant)  
- Few-shot, zero-shot, chain-of-thought  
- Structured outputs and schemas  
- Function calling and tools  
- LLM settings and controls  
- Prompt evaluation and benchmarking  
- Retrieval-Augmented Generation (RAG)  
- Prompt injection and security  