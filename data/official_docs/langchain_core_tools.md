# LangChain Core & Tools

- **Official source**: https://docs.langchain.com/
- **Last refreshed**: 2025-05-05
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

## Practical Implementation Notes

- Prefer LCEL pipe syntax over legacy `LLMChain` for new code.
- Always add output parsing to avoid raw LLM string handling.
- Use `RunnablePassthrough` to forward inputs through chain steps unchanged.
- `RunnableLambda(fn)` wraps arbitrary Python functions into chain components.
- Handle `OutputParserException` when LLM output doesn't match expected format.

## Common Mistakes

- Mixing legacy chain classes (`LLMChain`, `SequentialChain`) with LCEL — use one style.
- Not handling `OutputParserException` when LLM output doesn't match expected format.
- Forgetting to pass required variables to prompt templates at invocation.
- Creating overly complex chains when simple function calls suffice.

## Related Project Usage

- `src/kb/chunker.py`: Text splitting inspired by LangChain splitter patterns.
- `src/kb/loader.py`: Document loading following LangChain document model.
- `src/schemas.py`: Pydantic models for structured LLM output parsing.
