# Loguru Logging

- **Official source**: https://loguru.readthedocs.io/
- **Last refreshed**: 2026-05-13
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
- `trace()` is available below `debug()` for ultra-verbose output.
- All methods accept keyword arguments that become part of the log record's `extra` dict.

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
- Multiple sinks can be active simultaneously (e.g., console + file + monitoring).

**Sink types**:

| Sink | Type | Use Case |
|------|------|----------|
| `sys.stderr` | Stream | Console output (default) |
| `"logs/app.log"` | File path | Persistent log files with rotation |
| Callable | Function | Custom processing (monitoring, alerting) |
| `io.StringIO()` | Buffer | Testing (capture log output) |

**Rotation strategies**:

```python
logger.add("logs/app.log", rotation="10 MB")       # by file size
logger.add("logs/app.log", rotation="00:00")        # daily at midnight
logger.add("logs/app.log", rotation="1 week")       # weekly
logger.add("logs/app.log", rotation=lambda msg, _: len(msg) > 500)  # custom
```

- Rotated files are renamed with timestamps (e.g., `app.2025-05-07_18-00-00.log`).
- Combine `rotation` + `retention` + `compression` for production-ready log management.

### Formatting

Default format includes timestamp, level, module, function, line, and message.

```python
# Custom format string
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}")
```

Available fields: `{time}`, `{level}`, `{module}`, `{function}`, `{line}`, `{message}`, `{name}`, `{file}`.

Color markup for console: `<green>{time}</green> <level>{message}</level>`.

- Color markup is auto-disabled when output is not a terminal (e.g., piped to file or CI).
- Use `colorize=True/False` in `logger.add()` to override auto-detection.

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

- `logger.exception()` is equivalent to `logger.opt(exception=True).error()` but more readable.
- `@logger.catch` also works as a context manager: `with logger.catch(): ...`.
- Exception logging captures the full traceback including chained exceptions (`__cause__`, `__context__`).
- In production, sensitive local variables may appear in traces — consider filtering or using `diagnose=False`:

```python
logger.add("logs/prod.log", diagnose=False)  # omit local variable values in tracebacks
```

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

**Structured logging (JSON output)**:

```python
# JSON-formatted logs for log aggregation systems (ELK, Datadog, etc.)
logger.add(
    "logs/structured.log",
    format="{message}",
    serialize=True,  # outputs each log record as a JSON object
)

# Custom structured fields
logger.bind(request_id="req-456", user_id="u-123").info("Study guide generated")
# JSON output includes: {"text": "Study guide generated", "record": {"extra": {"request_id": "req-456", ...}}}
```

- `serialize=True` produces machine-readable JSON logs — ideal for production log pipelines.
- Bound context (`logger.bind()`) persists for the lifetime of the returned logger instance.
- `logger.contextualize()` is thread-safe and async-safe — context is scoped to the current execution context.

## Advanced Patterns

### Integration with Standard Library Logging

```python
import logging
from loguru import logger

# Intercept standard library logging and route to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        level = logger.level(record.levelname).name
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
```

- Libraries using `logging.getLogger()` (e.g., `uvicorn`, `httpx`) route through loguru.
- `depth=6` corrects the caller frame information in intercepted logs.
- This pattern centralizes all logging through a single loguru configuration.

### Async-Safe Logging

```python
# Enqueue=True makes the sink async-safe and thread-safe
logger.add("logs/app.log", enqueue=True, rotation="10 MB")

# Safe to call from async handlers, threads, and multiprocessing
async def handle_request():
    logger.info("Processing async request")  # non-blocking
```

- `enqueue=True` sends log records to a background queue — logging calls never block the event loop.
- Essential for async applications (FastAPI, aiohttp) and multi-threaded programs.
- Queue is flushed on interpreter shutdown; call `logger.complete()` for explicit flush.

### Testing with Loguru

```python
import io

def test_logging_output():
    output = io.StringIO()
    logger.remove()
    logger.add(output, format="{level}: {message}")
    
    my_function()  # function under test
    
    log_content = output.getvalue()
    assert "Processing" in log_content
    assert "ERROR" not in log_content
```

- Use `io.StringIO()` sink to capture and assert on log output in tests.
- Always `logger.remove()` before adding test sinks to avoid interference.

## Practical Implementation Notes

- Call `logger.remove()` before adding custom sinks to avoid duplicate output.
- Use `rotation` + `retention` together for production log management.
- Add structured context with `logger.bind()` for request-level tracing.
- Use `logger.opt(lazy=True)` with lambda for expensive log message construction.
- Configure logging once at application startup, not per-module.
- Use `enqueue=True` in production for thread-safe, non-blocking logging.
- Use `diagnose=False` in production sinks to prevent sensitive data in tracebacks.
- Set `serialize=True` for log aggregation pipelines (ELK, CloudWatch, Datadog).

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Duplicate log lines | Default sink not removed before adding custom sinks | Call `logger.remove()` first |
| Missing log output | Sink level higher than log level | Check `level` parameter on `logger.add()` |
| No color in terminal | Output redirected or `colorize=False` | Set `colorize=True` explicitly; verify terminal supports ANSI |
| Log file not rotating | Rotation condition not met | Check rotation value format (e.g., `"10 MB"` not `"10MB"`) |
| Traceback missing variable values | `diagnose=False` on sink | Set `diagnose=True` for development sinks |
| Slow logging in async app | Synchronous sink blocking event loop | Add `enqueue=True` to sink configuration |
| `logger.bind()` context leaking | Bound logger reused across requests | Use `logger.contextualize()` context manager for request-scoped context |

## Common Mistakes

- Adding sinks without removing the default — causes duplicate console output.
- Not using `logger.exception()` in except blocks — loses traceback information.
- Using f-strings instead of `{}` placeholders — loses lazy evaluation benefit.
- Calling `logger.remove()` with no arguments when intending to remove a specific sink.
- Setting log level too low (`DEBUG`) in production — degrades performance.
- Not using `enqueue=True` in async/threaded applications — risks data races.
- Forgetting `diagnose=False` in production — local variables appear in tracebacks (security risk).

## Related Project Usage

- `src/logging_config.py`: Loguru configuration with level from app settings.
- All `src/` modules: Use `from loguru import logger` for consistent logging.
- `src/kb/loader.py`, `src/kb/retrieval.py`: Log document loading and retrieval operations.
- `src/graphs/learn_nodes.py`, `src/graphs/quiz_nodes.py`: Log workflow node execution.

<!-- AUTO-GENERATED SOURCE PREVIEW START -->
## Latest Official Preview

This machine-generated section is refreshed from the configured external documentation URLs.

### https://loguru.readthedocs.io/en/stable/

```
loguru Overview API Reference Help & Guides Project Information loguru Table of Contents Loguru is a library which aims to bring enjoyable logging in Python. Did you ever feel lazy about configuring a logger and used print() instead?… I did, yet logging is fundamental to every application and eases the process of debugging. Using Loguru you have no excuse not to use logging from the start, this is as simple as from loguru import logger. Also, this library is intended to make Python logging less painful by adding a bunch of useful functionalities that solve caveats of the standard loggers. Using logs in your application should be an automatism, Loguru tries to make it both pleasant and powerful. Table of Contents  Overview Installation Features Take the tour API Reference loguru.logger Type Hints Help & Guides Switching from Standard Logging to Loguru Frequently Asked Questions and Tr...
```

*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*
<!-- AUTO-GENERATED SOURCE PREVIEW END -->
