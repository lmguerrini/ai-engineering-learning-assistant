# Pydantic Validation & Settings

- **Official source**: https://docs.pydantic.dev/
- **Last refreshed**: 2026-05-11
- **source_type**: official_docs
- **Versions**: `pydantic>=2.0`, `pydantic-settings>=2.0`

## When to Use

- Validating data structures with type-safe models.
- Loading and validating application configuration from environment variables.
- Parsing and validating LLM outputs into structured objects.

## Key Concepts

### BaseModel

```python
from pydantic import BaseModel, Field

class StudyGuide(BaseModel):
    topic: str = Field(description="The learning topic")
    sections: list[str] = Field(default_factory=list, description="Guide sections")
    difficulty: int = Field(default=1, ge=1, le=5, description="Difficulty 1-5")
```

- Fields declared as class attributes with type annotations.
- Validation runs automatically on instantiation — invalid data raises `ValidationError`.
- Access validated data via attribute access: `guide.topic`.
- Models are immutable by default in v2; use `model_config = ConfigDict(frozen=True)` to enforce.

**Model configuration**:

```python
from pydantic import ConfigDict

class StrictGuide(BaseModel):
    model_config = ConfigDict(
        frozen=True,           # immutable instances
        extra="forbid",        # reject unexpected fields
        str_strip_whitespace=True,  # strip whitespace from strings
        validate_assignment=True,   # validate on attribute assignment
    )
    topic: str
    difficulty: int = Field(ge=1, le=5)
```

- `extra="forbid"` raises `ValidationError` on unexpected fields — useful for strict API contracts.
- `extra="ignore"` silently drops unknown fields — useful for forward-compatible parsing.
- `extra="allow"` stores unknown fields in `model.__pydantic_extra__`.

### Field Types & Validators

```python
from pydantic import BaseModel, field_validator, model_validator

class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int

    @field_validator("options")
    @classmethod
    def validate_options(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 options required")
        return v

    @model_validator(mode="after")
    def validate_correct_index(self):
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index out of range")
        return self
```

Standard types: `str`, `int`, `float`, `bool`, `list`, `dict`, `Optional`.

`Field()` parameters: `default`, `default_factory`, `description`, `ge`, `le`, `min_length`, `max_length`, `pattern`.

**Validator execution order**: `field_validator` (per-field) runs before `model_validator` (cross-field). Within the same level, validators run in definition order.

**Discriminated unions** (for polymorphic models):

```python
from typing import Literal, Union
from pydantic import BaseModel

class MultipleChoice(BaseModel):
    type: Literal["multiple_choice"]
    options: list[str]
    correct_index: int

class FreeText(BaseModel):
    type: Literal["free_text"]
    expected_keywords: list[str]

class Quiz(BaseModel):
    questions: list[Union[MultipleChoice, FreeText]] = Field(discriminator="type")
```

- Discriminated unions use a `Literal` field to determine which model to parse into.
- Significantly faster than undiscriminated unions for complex nested structures.
- Useful for LLM outputs that return different question types.

### Serialization

```python
guide = StudyGuide(topic="RAG", sections=["Overview", "Implementation"], difficulty=3)

guide.model_dump()          # → {"topic": "RAG", "sections": [...], "difficulty": 3}
guide.model_dump_json()     # → '{"topic":"RAG",...}'

# Deserialization
StudyGuide.model_validate({"topic": "RAG", "sections": ["Overview"], "difficulty": 2})
StudyGuide.model_validate_json('{"topic":"RAG","sections":["Overview"],"difficulty":2}')

# JSON Schema (for OpenAI function calling / structured outputs)
StudyGuide.model_json_schema()  # → {"type": "object", "properties": {...}}
```

> **Note**: Pydantic v2 renamed `.dict()` → `.model_dump()`, `.json()` → `.model_dump_json()`, `.parse_obj()` → `.model_validate()`.

**Selective serialization**:

```python
# Include only specific fields
guide.model_dump(include={"topic", "difficulty"})  # → {"topic": "RAG", "difficulty": 3}

# Exclude fields
guide.model_dump(exclude={"sections"})  # → {"topic": "RAG", "difficulty": 3}

# Exclude unset fields (only include explicitly set values)
guide.model_dump(exclude_unset=True)

# Alias-based serialization (for API response formatting)
class APIResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    study_topic: str = Field(alias="studyTopic")
```

- `exclude_unset=True` is useful for PATCH-style updates — only serialize fields the user provided.
- `by_alias=True` serializes using field aliases (e.g., camelCase for JSON APIs).

### Pydantic Settings

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini")
    chroma_persist_dir: str = Field(default="./chroma_db")
    log_level: str = Field(default="INFO")

settings = Settings()  # reads from env vars + .env file
```

- Automatically reads from environment variables (case-insensitive).
- `env_file=".env"` loads from dotenv files.
- `extra="ignore"` skips unknown env vars without errors.
- Nested settings with `env_prefix` for grouped configuration.

**Settings priority order** (highest to lowest):
1. Constructor arguments (`Settings(openai_api_key="sk-...")`)
2. Environment variables
3. `.env` file values
4. Field defaults

**Nested settings and prefixes**:

```python
class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")
    
    db: DatabaseSettings = DatabaseSettings()
    # Set via: DB__HOST=mydb.example.com DB__PORT=5433
```

- `env_nested_delimiter` enables flat env vars to populate nested models.
- Useful for Docker/Kubernetes environments where env vars are the primary config source.

### Enums with Pydantic

```python
from enum import StrEnum

class Difficulty(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class Config(BaseModel):
    difficulty: Difficulty = Difficulty.BEGINNER
```

- Pydantic validates enum values automatically on instantiation.
- Serialize to string: `model.model_dump(mode="json")`.
- `StrEnum` values serialize as strings; `IntEnum` values serialize as integers.

## Advanced Patterns

### Handling LLM Output Parsing Errors

```python
from pydantic import ValidationError
import json

def parse_llm_output(raw: str, model_class: type[BaseModel]):
    """Parse LLM JSON output with structured error handling."""
    try:
        data = json.loads(raw)
        return model_class.model_validate(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except ValidationError as e:
        # Log specific field errors for debugging
        for error in e.errors():
            print(f"Field '{'.'.join(str(l) for l in error['loc'])}': {error['msg']}")
        raise
```

- `ValidationError.errors()` returns a list of dicts with `loc` (field path), `msg`, and `type` (error code).
- Common LLM parsing failures: missing required fields, wrong types, values outside constraints.
- Consider using `model_validate(data, strict=False)` for lenient coercion (e.g., `"3"` → `3`).

### Generic Models

```python
from typing import Generic, TypeVar

T = TypeVar("T")

class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: T
    error: str | None = None

# Usage
response = APIResponse[StudyGuide](success=True, data=guide, error=None)
```

- Generic models enable type-safe wrapper patterns for API responses, pagination, etc.

### Computed Fields

```python
from pydantic import computed_field

class QuizResult(BaseModel):
    correct: int
    total: int
    
    @computed_field
    @property
    def score_pct(self) -> float:
        return (self.correct / self.total * 100) if self.total > 0 else 0.0
```

- `@computed_field` includes the property in serialization output without storing it.
- Computed fields are read-only and recalculated on access.

## Practical Implementation Notes

- Use `Field(default_factory=list)` for mutable defaults, never `Field(default=[])`.
- Use `model_json_schema()` to generate JSON schemas for OpenAI function calling.
- Combine `BaseSettings` with `.env` files for twelve-factor app configuration.
- Keep models focused — one model per logical data structure.
- Handle `ValidationError` explicitly when parsing untrusted input (e.g., LLM output).
- Use `model_config = ConfigDict(extra="forbid")` for strict API contracts; `extra="ignore"` for forward-compatible parsing.
- Prefer `model_validate()` over direct constructor for parsing external data — same behavior but signals intent.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ValidationError` on valid-looking data | Strict type checking (e.g., `"3"` not accepted as `int`) | Use `strict=False` in `model_validate()` or add coercion |
| Settings not reading from `.env` | `env_file` path wrong or `pydantic-settings` not installed | Check path relative to CWD; `pip install pydantic-settings` |
| Nested model not validating | Dict passed instead of model instance | Pydantic auto-coerces dicts to models; check field types |
| `model_json_schema()` produces invalid OpenAI schema | Optional fields or unions not supported | Simplify schema; ensure all fields have explicit types |
| Extra fields silently dropped | `extra="ignore"` is set (default for `BaseSettings`) | Use `extra="forbid"` if unexpected fields should error |
| `AttributeError: 'dict' has no attribute 'topic'` | Forgot to call `model_validate()` on raw dict | Parse with `Model.model_validate(data)` before attribute access |

## Common Mistakes

- Using mutable default values (`default=[]`) instead of `default_factory`.
- Forgetting that Pydantic v2 renamed `.dict()` to `.model_dump()`.
- Not handling `ValidationError` when parsing untrusted input.
- Defining overly complex nested models when a simple dict suffices.
- Ignoring `extra="forbid"` vs `extra="ignore"` behavior differences.
- Not installing `pydantic-settings` separately — it’s a separate package in v2.
- Using `Optional[str]` without a default — field is still required; use `Optional[str] = None`.

## Related Project Usage

- `src/config.py`: `Settings(BaseSettings)` for app configuration from env vars.
- `src/schemas.py`: `StudyGuide`, `QuizQuestion`, `QuizResult`, `UserProgress` models.
- `src/kb/loader.py`: `Document` model for loaded documents.
- `src/graphs/learn_nodes.py`: Parses LLM JSON output into `StudyGuide` model.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://pydantic.dev/docs/

```
Skip to content Pydantic Docs Pydantic Validation Pydantic AI Pydantic Logfire Search Ctrl K Pydantic Docs Documentation for the Pydantic stack. Build and validate data with Pydantic Validation, create agents with Pydantic AI, and observe and improve agents in production with Pydantic Logfire. Pydantic Validation Data validation using Python type annotations. Parse and validate complex data, generate JSON schemas, and ensure data integrity. Pydantic AI Agent framework for building production AI applications. Type-safe, structured outputs, tool use, multi-agent orchestration with native Logfire integration. Pydantic Logfire General and AI observability to monitor LLM calls, agent behavior, costs, and service performance across your entire stack. © Pydantic Services Inc. 2025 to present
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
