# Pydantic Validation & Settings

- **Official source**: https://docs.pydantic.dev/
- **Last refreshed**: 2025-05-05
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

## Practical Implementation Notes

- Use `Field(default_factory=list)` for mutable defaults, never `Field(default=[])`.
- Use `model_json_schema()` to generate JSON schemas for OpenAI function calling.
- Combine `BaseSettings` with `.env` files for twelve-factor app configuration.
- Keep models focused — one model per logical data structure.
- Handle `ValidationError` explicitly when parsing untrusted input (e.g., LLM output).

## Common Mistakes

- Using mutable default values (`default=[]`) instead of `default_factory`.
- Forgetting that Pydantic v2 renamed `.dict()` to `.model_dump()`.
- Not handling `ValidationError` when parsing untrusted input.
- Defining overly complex nested models when a simple dict suffices.
- Ignoring `extra="forbid"` vs `extra="ignore"` behavior differences.

## Related Project Usage

- `src/config.py`: `Settings(BaseSettings)` for app configuration from env vars.
- `src/schemas.py`: `StudyGuide`, `QuizQuestion`, `QuizResult`, `UserProgress` models.
- `src/kb/loader.py`: `Document` model for loaded documents.
- `src/graphs/learn_nodes.py`: Parses LLM JSON output into `StudyGuide` model.
