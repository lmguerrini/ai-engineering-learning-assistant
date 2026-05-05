# Pydantic Validation & Settings

- **Official source**: https://docs.pydantic.dev/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Validating data structures with type-safe models.
- Loading and validating application configuration from environment variables.
- Parsing and validating LLM outputs into structured objects.

## Key Concepts

### BaseModel

- Inherit from `BaseModel` to define validated data classes.
- Fields are declared as class attributes with type annotations.
- Validation runs automatically on instantiation.
- Access validated data via attribute access: `model.field_name`.

### Field Types & Validators

- Standard types: `str`, `int`, `float`, `bool`, `list`, `dict`, `Optional`.
- `Field(default=..., description=..., ge=0, le=100)` for constraints.
- `@field_validator("field_name")` for custom validation logic.
- `@model_validator(mode="after")` for cross-field validation.

### Serialization

- `model.model_dump()` converts to dict (replaces `.dict()` in v2).
- `model.model_dump_json()` converts to JSON string.
- `Model.model_validate(data)` creates instance from dict.
- `Model.model_validate_json(json_str)` parses from JSON string.

### Pydantic Settings

- `from pydantic_settings import BaseSettings` for config classes.
- Automatically reads from environment variables (case-insensitive).
- `SettingsConfigDict(env_file=".env")` loads from .env files.
- `extra="ignore"` skips unknown env vars without errors.
- Nested settings with `env_prefix` for grouped configuration.

### Enums with Pydantic

- Use Python `enum.Enum` or `StrEnum` for constrained choices.
- Pydantic validates enum values automatically.
- Serialize to string value with `model_dump(mode="json")`.

## Practical Implementation Notes

- Use `total=False` on TypedDict when fields are optional (common with LangGraph state).
- Use `Field(default_factory=list)` for mutable defaults, never `Field(default=[])`.
- Keep models focused — one model per logical data structure.
- Use `model_json_schema()` to generate JSON schemas for LLM function calling.
- Combine `BaseSettings` with `.env` files for twelve-factor config.

## Common Mistakes

- Using mutable default values (`default=[]`) instead of `default_factory`.
- Forgetting that Pydantic v2 renamed `.dict()` to `.model_dump()`.
- Not handling `ValidationError` when parsing untrusted input.
- Defining overly complex nested models when a simple dict suffices.
- Ignoring `extra="forbid"` vs `extra="ignore"` behavior differences.

## Related Project Usage

- `src/config.py`: `Settings(BaseSettings)` for app configuration.
- `src/schemas.py`: `StudyGuide`, `QuizQuestion`, `QuizResult`, `UserProgress` models.
- `src/kb/loader.py`: `Document` model for loaded documents.
- `src/graphs/learn_nodes.py`: Parses LLM JSON output into `StudyGuide` model.
