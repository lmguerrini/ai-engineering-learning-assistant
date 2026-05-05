# Loguru Logging

- **Official source**: https://loguru.readthedocs.io/
- **Last refreshed**: 2025-05-05
- **source_type**: official_docs

## When to Use

- Adding structured logging to Python applications.
- Replacing the standard `logging` module with a simpler API.
- Configuring log rotation, retention, and formatting.

## Key Concepts

### Logger Interface

- Single global `logger` object: `from loguru import logger`.
- Methods: `logger.debug()`, `logger.info()`, `logger.warning()`, `logger.error()`, `logger.critical()`.
- String formatting with `{}` placeholders: `logger.info("User {}", username)`.
- Lazy evaluation — format arguments only if the level is active.

### Sinks (Output Destinations)

- `logger.add(sys.stderr, level="INFO")` — console output.
- `logger.add("file.log", rotation="10 MB")` — file with rotation.
- `logger.add("file.log", retention="7 days")` — automatic cleanup.
- `logger.add(custom_function)` — any callable as a log sink.
- `logger.remove()` removes all previously added sinks.

### Formatting

- Default format includes timestamp, level, module, function, and message.
- Custom format: `logger.add(sink, format="{time} {level} {message}")`.
- Available fields: `{time}`, `{level}`, `{module}`, `{function}`, `{line}`, `{message}`.
- Color markup: `<green>{time}</green>` for colored console output.

### Exception Handling

- `logger.exception("msg")` logs the current exception with full traceback.
- `@logger.catch` decorator catches and logs exceptions from functions.
- `logger.opt(exception=True)` includes exception info in any log call.
- Stack traces are formatted with local variable values.

### Filtering & Configuration

- `logger.add(sink, filter=lambda record: "keyword" in record["message"])`.
- `logger.bind(key=value)` adds context to subsequent log calls.
- `logger.contextualize(key=value)` for context managers.
- `logger.disable("module_name")` suppresses logs from specific modules.
- `logger.enable("module_name")` re-enables suppressed logs.

## Practical Implementation Notes

- Remove the default stderr sink before adding custom sinks: `logger.remove()`.
- Use `rotation` and `retention` together for production log management.
- Add structured context with `logger.bind()` for request tracing.
- Use `logger.opt(lazy=True)` with lambda for expensive log message construction.
- Configure logging once at application startup, not per-module.

## Common Mistakes

- Adding multiple sinks without removing the default, causing duplicate output.
- Not using `logger.exception()` in except blocks, losing traceback info.
- Using f-strings instead of `{}` placeholders, losing lazy evaluation benefit.
- Forgetting that `logger.remove()` with no arguments removes all sinks.
- Setting log level too low in production, causing performance issues.

## Related Project Usage

- `src/logging_config.py`: Loguru configuration with level from app settings.
- All `src/` modules: Use `from loguru import logger` for consistent logging.
- `src/kb/loader.py`, `src/kb/retrieval.py`: Log document loading and retrieval operations.
- `src/graphs/learn_nodes.py`, `src/graphs/quiz_nodes.py`: Log workflow node execution.
