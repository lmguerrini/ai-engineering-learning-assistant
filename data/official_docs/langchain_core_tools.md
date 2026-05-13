# LangChain Core & Tools

- **Official source**: https://docs.langchain.com/
- **Last refreshed**: 2026-05-13
- **source_type**: official_docs
- **Versions**: `langchain-core>=0.3`, `langchain>=0.3`

## When to Use

- Building LLM-powered chains for sequential reasoning.
- Integrating external tools (search, calculators, APIs) with LLMs.
- Creating reusable prompt templates and output parsers.

## Key Concepts

### LCEL (LangChain Expression Language)

Compose chains using the pipe (`|`) operator. Each component implements the `Runnable` interface.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor."),
    ("human", "Explain {topic} in {style} style."),
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "RAG", "style": "concise"})
```

`Runnable` interface methods: `invoke`, `batch`, `stream`, `ainvoke`, `abatch`, `astream`.

Built-in resilience: `chain.with_retry(stop_after_attempt=3)`, `chain.with_fallback([fallback_chain])`.

**Advanced LCEL patterns**:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

# Parallel execution — run retriever and question processing simultaneously
rag_chain = RunnableParallel(
    context=retriever | format_docs,
    question=RunnablePassthrough(),
) | prompt | llm | StrOutputParser()

# Conditional branching
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["type"] == "quiz", quiz_chain),
    (lambda x: x["type"] == "summary", summary_chain),
    default_chain,  # fallback
)
```

- `RunnableParallel` executes branches concurrently and merges results into a dict.
- `RunnableBranch` evaluates conditions in order; first match wins.
- `RunnableLambda(fn)` wraps any Python function; use `RunnableLambda(async_fn)` for async.
- All `Runnable` methods accept `config` dict with `callbacks`, `tags`, `metadata`, and `run_name` for observability.
- `chain.with_config(run_name="my_step")` labels steps in LangSmith traces.

### Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI engineering tutor."),
    MessagesPlaceholder("chat_history"),  # inject dynamic message list
    ("human", "{question}"),
])
```

- Use `{variable}` placeholders filled at invocation time.
- `MessagesPlaceholder` injects dynamic message history into the prompt.
- Message types: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.
- `ChatPromptTemplate.from_template("...")` creates a single human-message template (shorthand).
- Partial prompts: `prompt.partial(style="concise")` pre-fills variables, useful for shared templates.
- `FewShotChatMessagePromptTemplate` injects dynamic example sets for in-context learning.

### Output Parsers

| Parser | Use Case |
|--------|----------|
| `StrOutputParser()` | Extract raw text from LLM response |
| `JsonOutputParser(pydantic_object=Model)` | Parse into Pydantic model |
| `PydanticOutputParser(pydantic_object=Model)` | Parse + auto-generate format instructions |

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class StudyGuide(BaseModel):
    topic: str
    sections: list[str]

parser = JsonOutputParser(pydantic_object=StudyGuide)
chain = prompt | llm | parser  # returns StudyGuide dict
```

**Error recovery for output parsing**:

```python
from langchain_core.output_parsers import OutputParserException
from langchain.output_parsers import OutputFixingParser

# Auto-fix malformed output by re-prompting the LLM
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

# Or use retry parser with the original prompt
from langchain.output_parsers import RetryOutputParser
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=llm)
```

> **Caveat**: `OutputFixingParser` makes an additional LLM call on failure — monitor cost. Prefer `JsonOutputParser` with clear format instructions to minimize parse failures.

### Tools

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant documents."""
    results = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in results)
```

- `@tool` decorator creates an LLM-callable tool with `name`, `description`, and `args_schema`.
- `ToolMessage` carries execution results back to the LLM.
- Tools integrate with LangGraph nodes for agentic tool use.

**Advanced tool patterns**:

```python
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

# Tool with structured input schema
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=20)

@tool(args_schema=SearchInput)
def search_kb(query: str, max_results: int = 5) -> str:
    """Search knowledge base with configurable result count."""
    return perform_search(query, max_results)

# Programmatic tool creation
def my_func(x: str) -> str:
    return x.upper()

tool_instance = StructuredTool.from_function(
    func=my_func,
    name="uppercase",
    description="Convert text to uppercase",
)

# Bind tools to chat model
llm_with_tools = llm.bind_tools([search_kb, tool_instance])
```

- `llm.bind_tools(tools)` attaches tools to a chat model; the model decides when to call them.
- Tool descriptions are critical for model selection — vague descriptions lead to incorrect tool choices.
- Return strings from tools; the LLM processes the string result. Return errors as descriptive strings, not exceptions.

### Document Loaders & Text Splitters

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("./data/raw", glob="*.md", loader_cls=TextLoader)
docs = loader.load()  # list[Document] with page_content + metadata

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

- `TextLoader`, `DirectoryLoader`, `UnstructuredMarkdownLoader` for various formats.
- Metadata is preserved through splitting operations.
- `RecursiveCharacterTextSplitter` tries separators in order: `\n\n`, `\n`, ` `, `""` — prefers paragraph boundaries.
- For markdown: `MarkdownHeaderTextSplitter` splits on heading hierarchy and includes headers in metadata.
- Set `chunk_overlap` to 10-20% of `chunk_size` for context continuity across chunk boundaries.
- Add custom metadata during loading for downstream filtering: `doc.metadata["category"] = "tutorial"`.

## Advanced Patterns

### Async Execution

```python
# All LCEL chains support async natively
result = await chain.ainvoke({"topic": "RAG", "style": "concise"})

# Batch with concurrency control
results = await chain.abatch(
    [{"topic": t} for t in topics],
    config={"max_concurrency": 5},
)
```

- Use `ainvoke`/`abatch`/`astream` in async contexts (FastAPI, Streamlit async handlers).
- `max_concurrency` in batch config limits parallel LLM calls — critical for rate limit compliance.

### Caching

```python
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # caches identical prompt → response pairs
```

- Cache reduces cost and latency for repeated queries during development.
- Disable caching in evaluation pipelines to ensure fresh responses.

### Callbacks & Observability

```python
from langchain_core.callbacks import StdOutCallbackHandler

result = chain.invoke(
    {"topic": "RAG"},
    config={"callbacks": [StdOutCallbackHandler()], "tags": ["learn-flow"]},
)
```

- Callbacks fire on chain/LLM/tool start, end, and error events.
- Tags and metadata propagate to LangSmith traces for filtering.

## Practical Implementation Notes

- Prefer LCEL pipe syntax over legacy `LLMChain` for new code.
- Always add output parsing to avoid raw LLM string handling.
- Use `RunnablePassthrough` to forward inputs through chain steps unchanged.
- `RunnableLambda(fn)` wraps arbitrary Python functions into chain components.
- Handle `OutputParserException` when LLM output doesn't match expected format.
- Use `.with_retry(stop_after_attempt=3)` on LLM steps for transient failures.
- Add `.with_fallback([simpler_chain])` for graceful degradation when primary chain fails.
- Profile chain latency with `%timeit chain.invoke(...)` during development; each `|` adds minimal overhead.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `OutputParserException` | LLM returned malformed JSON/text | Add format instructions to prompt; use `OutputFixingParser` |
| Missing variable error on invoke | Prompt expects variable not in input dict | Check `prompt.input_variables`; use `.partial()` for defaults |
| Chain returns `AIMessage` instead of string | Missing output parser at chain end | Append `\| StrOutputParser()` |
| `RunnableParallel` timeout | One branch hangs (e.g., slow retriever) | Add timeout: `chain.with_config({"timeout": 30})` |
| Duplicate callbacks firing | Callbacks attached at multiple levels | Pass callbacks only at top-level `invoke` call |

## Common Mistakes

- Mixing legacy chain classes (`LLMChain`, `SequentialChain`) with LCEL — use one style.
- Not handling `OutputParserException` when LLM output doesn't match expected format.
- Forgetting to pass required variables to prompt templates at invocation.
- Creating overly complex chains when simple function calls suffice.
- Using `RunnableLambda` without error handling — exceptions crash the entire chain.
- Not setting `max_concurrency` on `batch` calls — overwhelming the LLM API with parallel requests.
- Passing mutable state through `RunnablePassthrough` — mutations in one branch affect others.

## Related Project Usage

- `src/kb/chunker.py`: Text splitting inspired by LangChain splitter patterns.
- `src/kb/loader.py`: Document loading following LangChain document model.
- `src/schemas.py`: Pydantic models for structured LLM output parsing.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://docs.langchain.com/

```
Skip to main content Join us May 13th & May 14th at Interrupt, the Agent Conference by LangChain. Buy tickets > Docs by LangChain home page Home Search... ⌘ K Ask AI GitHub Try LangSmith Try LangSmith Search... Navigation Documentation Index Fetch the complete documentation index at: https://docs.langchain.com/llms.txt Use this file to discover all available pages before exploring further. Documentation LangChain is the platform for agent engineering. AI teams at Clay, Rippling, Cloudflare, Workday, and more trust LangChain’s products to engineer reliable agents. LangSmith LangSmith is a platform that helps AI teams use live production data for continuous testing and improvement. LangSmith provides: Observability See exactly how your agent thinks and acts with detailed tracing and aggregate trend metrics. Learn more Evaluation Test and score agent behavior on production data or offlin...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
