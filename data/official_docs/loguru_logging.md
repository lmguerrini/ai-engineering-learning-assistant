# Loguru Logging

- **Official source**: https://loguru.readthedocs.io/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs
- **Versions**: `loguru>=0.7`

## When to Use

- Adding structured logging to Python applications.
- Replacing the standard `logging` module with a simpler API.
- Configuring log rotation, retention, and formatting.

## Key Concepts

### Logger Interface

Single global logger — no factory or configuration boilerplate:

```python
from loguru import logger

logger.info("Processing topic: {}", topic)
logger.debug("Retrieved {} documents", len(docs))
logger.warning("No context found for topic: {}", topic)
logger.error("LLM call failed: {}", error_msg)
```

- Methods: `debug()`, `info()`, `warning()`, `error()`, `critical()`.
- Uses `{}` placeholders (not `%s`) — arguments are formatted lazily only if the level is active.

### Sinks (Output Destinations)

```python
import sys

# Remove default stderr sink
logger.remove()

# Console output
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")

# File with rotation and retention
logger.add(
    "logs/app.log",
    rotation="10 MB",    # rotate when file reaches 10 MB
    retention="7 days",  # delete logs older than 7 days
    compression="zip",   # compress rotated files
    level="DEBUG",
)

# Custom callable sink
logger.add(my_monitoring_function, level="ERROR")
```

- `logger.remove()` with no arguments removes **all** previously added sinks.
- `logger.remove(sink_id)` removes a specific sink by the ID returned from `add()`.

### Formatting

Default format includes timestamp, level, module, function, line, and message.

```python
# Custom format string
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}")
```

Available fields: `{time}`, `{level}`, `{module}`, `{function}`, `{line}`, `{message}`, `{name}`, `{file}`.

Color markup for console: `<green>{time}</green> <level>{message}</level>`.

### Exception Handling

```python
# In except block — includes full traceback with local variables
try:
    result = process_query(topic)
except Exception:
    logger.exception("Failed to process query")

# Decorator — catches and logs exceptions automatically
@logger.catch
def generate_study_guide(topic: str) -> dict:
    ...

# Include exception info in any log call
logger.opt(exception=True).error("Something failed")
```

Stack traces include local variable values for faster debugging.

### Context & Filtering

```python
# Bind context to logger
contextualized = logger.bind(user_id="user-123", session="abc")
contextualized.info("Starting workflow")  # includes user_id and session in record

# Context manager
with logger.contextualize(request_id="req-456"):
    logger.info("Processing request")  # includes request_id

# Filter by module
logger.disable("noisy_module")
logger.enable("noisy_module")

# Custom filter function
logger.add(sink, filter=lambda record: "kb" in record["name"])
```

## Practical Implementation Notes

- Call `logger.remove()` before adding custom sinks to avoid duplicate output.
- Use `rotation` + `retention` together for production log management.
- Add structured context with `logger.bind()` for request-level tracing.
- Use `logger.opt(lazy=True)` with lambda for expensive log message construction.
- Configure logging once at application startup, not per-module.

## Common Mistakes

- Adding sinks without removing the default — causes duplicate console output.
- Not using `logger.exception()` in except blocks — loses traceback information.
- Using f-strings instead of `{}` placeholders — loses lazy evaluation benefit.
- Calling `logger.remove()` with no arguments when intending to remove a specific sink.
- Setting log level too low (`DEBUG`) in production — degrades performance.

## Related Project Usage

- `src/logging_config.py`: Loguru configuration with level from app settings.
- All `src/` modules: Use `from loguru import logger` for consistent logging.
- `src/kb/loader.py`, `src/kb/retrieval.py`: Log document loading and retrieval operations.
- `src/graphs/learn_nodes.py`, `src/graphs/quiz_nodes.py`: Log workflow node execution.
