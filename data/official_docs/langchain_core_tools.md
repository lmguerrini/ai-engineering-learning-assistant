# LangChain Core & Tools

- **Official source**: https://docs.langchain.com/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Building LLM-powered chains for sequential reasoning.
- Integrating external tools (search, calculators, APIs) with LLMs.
- Creating reusable prompt templates and output parsers.

## Key Concepts

### LCEL (LangChain Expression Language)

- Compose chains using the pipe (`|`) operator: `prompt | llm | parser`.
- Each component implements `Runnable` interface with `invoke`, `batch`, `stream`.
- Supports async via `ainvoke`, `abatch`, `astream`.
- Built-in retry and fallback support: `chain.with_retry()`, `chain.with_fallback()`.

### Prompt Templates

- `ChatPromptTemplate.from_messages([...])` for chat-style prompts.
- Use `{variable}` placeholders filled at invocation time.
- `SystemMessage`, `HumanMessage`, `AIMessage` for typed message construction.
- `MessagesPlaceholder` for injecting dynamic message history.

### Output Parsers

- `StrOutputParser()` extracts raw text from LLM response.
- `JsonOutputParser(pydantic_object=MyModel)` parses into Pydantic models.
- `PydanticOutputParser` generates format instructions automatically.
- Parsers can be chained after the LLM in LCEL pipelines.

### Tools

- Annotate functions with `@tool` decorator to make them LLM-callable.
- Tools have `name`, `description`, and `args_schema` (Pydantic model).
- `ToolMessage` carries tool execution results back to the LLM.
- Tools integrate with LangGraph nodes for agentic tool use.

### Document Loaders & Text Splitters

- `TextLoader`, `DirectoryLoader`, `UnstructuredMarkdownLoader` for various formats.
- `RecursiveCharacterTextSplitter` for chunking with configurable size and overlap.
- `CharacterTextSplitter` for simpler delimiter-based splitting.
- Metadata is preserved through splitting operations.

## Practical Implementation Notes

- Prefer LCEL pipe syntax over legacy `LLMChain` for new code.
- Always add output parsing to avoid raw LLM string handling.
- Use `RunnablePassthrough` to forward inputs through chain steps.
- `RunnableLambda` wraps arbitrary Python functions into chain components.
- Test chains with small inputs before scaling.

## Common Mistakes

- Mixing legacy chain classes with LCEL — pick one style.
- Not handling `OutputParserException` when LLM output doesn't match expected format.
- Forgetting to pass required variables to prompt templates.
- Creating overly complex chains when simple function calls suffice.

## Related Project Usage

- `src/kb/chunker.py`: Text splitting inspired by LangChain splitter patterns.
- `src/kb/loader.py`: Document loading following LangChain document model.
- `src/schemas.py`: Pydantic models for structured LLM output parsing.
