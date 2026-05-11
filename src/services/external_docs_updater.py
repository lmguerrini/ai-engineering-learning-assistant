"""Manual updater for external official docs stored under data/official_docs."""

from __future__ import annotations

import re
import ssl
import tempfile
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from src.config import get_settings

AUTO_PREVIEW_START = "<!-- AUTO-GENERATED SOURCE PREVIEW START -->"
AUTO_PREVIEW_END = "<!-- AUTO-GENERATED SOURCE PREVIEW END -->"

EXTERNAL_DOC_SOURCES: list[dict[str, object]] = [
    {
        "name": "Agent Harness & Production Deployment",
        "filename": "agent_harness_production.md",
        "urls": [
            "https://langchain-ai.github.io/langgraph/concepts/deployment/",
            "https://python.langchain.com/docs/concepts/architecture/",
        ],
    },
    {
        "name": "AI Agents & ReAct Pattern",
        "filename": "ai_agents_react_pattern.md",
        "urls": [
            "https://arxiv.org/abs/2210.03629",
            "https://python.langchain.com/docs/concepts/agents/",
        ],
    },
    {
        "name": "Chroma Vector Store",
        "filename": "chroma_vector_store.md",
        "urls": ["https://docs.trychroma.com/"],
    },
    {
        "name": "LangChain Core & Tools",
        "filename": "langchain_core_tools.md",
        "urls": ["https://docs.langchain.com/"],
    },
    {
        "name": "LangGraph State & Orchestration",
        "filename": "langgraph_state_orchestration.md",
        "urls": ["https://langchain-ai.github.io/langgraph/"],
    },
    {
        "name": "LangSmith Observability",
        "filename": "langsmith_observability.md",
        "urls": ["https://docs.smith.langchain.com/"],
    },
    {
        "name": "Loguru Logging",
        "filename": "loguru_logging.md",
        "urls": ["https://loguru.readthedocs.io/"],
    },
    {
        "name": "Long-Term Memory & Human-in-the-Loop",
        "filename": "memory_human_in_the_loop.md",
        "urls": [
            "https://langchain-ai.github.io/langgraph/concepts/memory/",
            "https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/",
        ],
    },
    {
        "name": "OpenAI API & Structured Outputs",
        "filename": "openai_api_structured_outputs.md",
        "urls": ["https://platform.openai.com/docs/"],
    },
    {
        "name": "Pydantic Validation & Settings",
        "filename": "pydantic_validation_settings.md",
        "urls": ["https://docs.pydantic.dev/"],
    },
    {
        "name": "RAGAs Evaluation",
        "filename": "ragas_evaluation.md",
        "urls": ["https://docs.ragas.io/"],
    },
    {
        "name": "State Management & Agentic RAG",
        "filename": "state_management_agentic_rag.md",
        "urls": [
            "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/",
            "https://python.langchain.com/docs/concepts/rag/",
        ],
    },
    {
        "name": "Streamlit App Patterns",
        "filename": "streamlit_app_patterns.md",
        "urls": ["https://docs.streamlit.io/"],
    },
    {
        "name": "Tool Calling & Function Calling",
        "filename": "tool_calling_function_calling.md",
        "urls": [
            "https://platform.openai.com/docs/guides/function-calling",
            "https://python.langchain.com/docs/concepts/tool_calling/",
        ],
    },
]


def get_external_doc_source_registry() -> list[dict[str, object]]:
    """Return the configured external docs registry."""
    return [
        {
            "name": str(source["name"]),
            "filename": str(source["filename"]),
            "urls": list(source["urls"]),
        }
        for source in EXTERNAL_DOC_SOURCES
    ]


def get_external_docs_dir(directory: str | Path | None = None) -> Path:
    """Return the configured official docs directory."""
    if directory is not None:
        return Path(directory)
    return Path(get_settings().official_docs_dir)


def fetch_external_doc_url(url: str) -> str | None:
    """Fetch one configured external docs URL with structured result metadata."""
    try:
        import requests
    except Exception:
        requests = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    if requests is not None:
        return _fetch_with_requests(url, headers)
    return _fetch_with_urllib(url, headers)


def get_external_docs_source_status(
    directory: str | Path | None = None,
    sources: list[dict[str, object]] | None = None,
) -> list[dict[str, str]]:
    """Return reviewer-facing status rows for configured external docs files."""
    docs_dir = get_external_docs_dir(directory)
    registry = sources or get_external_doc_source_registry()
    rows: list[dict[str, str]] = []
    for source in registry:
        filepath = docs_dir / str(source["filename"])
        rows.append(
            {
                "Source": str(source["name"]),
                "File": str(source["filename"]),
                "Primary Domain": _primary_domain(source),
                "Local File": "Present" if filepath.exists() else "Missing",
                "Last Refreshed": _read_last_refreshed(filepath) if filepath.exists() else "—",
                "URLs": str(len(_source_urls(source))),
            }
        )
    return rows


def get_latest_external_docs_refresh_date(
    directory: str | Path | None = None,
    sources: list[dict[str, object]] | None = None,
) -> str | None:
    """Return the most recent official-doc refresh date found in local metadata."""
    docs_dir = get_external_docs_dir(directory)
    registry = sources or get_external_doc_source_registry()
    latest_value: str | None = None
    latest_sort_key: datetime | None = None

    for source in registry:
        filepath = docs_dir / str(source["filename"])
        if not filepath.exists():
            continue
        refreshed = _read_last_refreshed(filepath)
        if refreshed == "—":
            continue

        sort_key = _parse_refresh_value(refreshed)
        if sort_key is None:
            if latest_value is None:
                latest_value = refreshed
            continue

        if latest_sort_key is None or sort_key > latest_sort_key:
            latest_sort_key = sort_key
            latest_value = refreshed

    return latest_value


def update_external_docs(
    directory: str | Path | None = None,
    sources: list[dict[str, object]] | None = None,
    fetcher=None,
) -> dict:
    """Manually refresh configured external docs files in a safe, non-destructive way."""
    docs_dir = get_external_docs_dir(directory)
    registry = sources or get_external_doc_source_registry()
    fetch = fetcher or fetch_external_doc_url
    refreshed_on = datetime.now(timezone.utc).date().isoformat()
    results: list[dict[str, object]] = []
    updated_files = 0
    partial_files = 0
    skipped_files = 0
    error_count = 0

    for source in registry:
        urls = _source_urls(source)
        filepath = docs_dir / str(source["filename"])
        existing_text = filepath.read_text(encoding="utf-8") if filepath.exists() else None
        fetched_pages: list[dict[str, str]] = []
        failed_urls: list[str] = []

        for url in urls:
            fetch_result = _normalize_fetch_result(fetch(url), source_url=url)
            if fetch_result["ok"] and str(fetch_result["content"]).strip():
                fetched_pages.append(
                    {
                        "url": str(fetch_result.get("final_url") or url),
                        "content": str(fetch_result["content"]),
                    }
                )
            else:
                failed_urls.append(_format_failed_url(fetch_result))

        if not fetched_pages:
            reason = _summarize_failed_urls(failed_urls)
            if filepath.exists():
                skipped_files += 1
                results.append(
                    {
                        "Source": str(source["name"]),
                        "File": str(source["filename"]),
                        "Status": "Skipped",
                        "URLs Checked": len(urls),
                        "URLs Fetched": 0,
                        "Result": f"Fetch failed; kept existing local file unchanged. {reason}",
                    }
                )
            else:
                error_count += 1
                results.append(
                    {
                        "Source": str(source["name"]),
                        "File": str(source["filename"]),
                        "Status": "Error",
                        "URLs Checked": len(urls),
                        "URLs Fetched": 0,
                        "Result": f"Fetch failed and no local file was available to preserve. {reason}",
                    }
                )
            continue

        updated_text = _build_updated_doc_text(
            source=source,
            existing_text=existing_text,
            fetched_pages=fetched_pages,
            refreshed_on=refreshed_on,
        )
        _safe_write_text(filepath, updated_text)
        status = "Updated"
        note = f"Fetched {len(fetched_pages)}/{len(urls)} URL(s) and refreshed the local Markdown file."
        if failed_urls:
            status = "Partial"
            partial_files += 1
            note += f" Partial success. {_summarize_failed_urls(failed_urls)}"
        else:
            updated_files += 1
        results.append(
            {
                "Source": str(source["name"]),
                "File": str(source["filename"]),
                "Status": status,
                "URLs Checked": len(urls),
                "URLs Fetched": len(fetched_pages),
                "Result": note,
            }
        )

    return {
        "checked_sources": len(registry),
        "updated_files": updated_files,
        "partial_files": partial_files,
        "skipped_files": skipped_files,
        "error_count": error_count,
        "results": results,
    }


def _source_urls(source: dict[str, object]) -> list[str]:
    """Normalize configured source URLs into a concrete list."""
    return [str(url) for url in list(source.get("urls", []))]


def _primary_domain(source: dict[str, object]) -> str:
    """Return a compact reviewer-facing domain label for the primary source URL."""
    urls = _source_urls(source)
    if not urls:
        return "—"

    netloc = urlparse(urls[0]).netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or "—"


def _fetch_with_requests(url: str, headers: dict[str, str]) -> dict[str, object]:
    """Fetch one URL with requests and return structured status information."""
    import requests

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(10, 20),
            allow_redirects=True,
        )
    except requests.exceptions.SSLError as exc:
        return _fetch_failure(url, "SSL error", exc)
    except requests.exceptions.Timeout as exc:
        return _fetch_failure(url, "timeout", exc)
    except requests.exceptions.TooManyRedirects as exc:
        return _fetch_failure(url, "too many redirects", exc)
    except requests.exceptions.ConnectionError as exc:
        return _fetch_failure(url, "connection error", exc)
    except requests.exceptions.RequestException as exc:
        return _fetch_failure(url, "request error", exc)

    if response.status_code >= 400:
        return {
            "ok": False,
            "content": None,
            "error": f"HTTP {response.status_code} {response.reason}",
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("content-type", ""),
            "source_url": url,
        }

    content_type = response.headers.get("content-type", "")
    if not _is_supported_content_type(content_type):
        return {
            "ok": False,
            "content": None,
            "error": f"unsupported content type: {content_type or 'unknown'}",
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": content_type,
            "source_url": url,
        }

    if not response.text or not response.text.strip():
        return {
            "ok": False,
            "content": None,
            "error": "empty response",
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": content_type,
            "source_url": url,
        }

    return {
        "ok": True,
        "content": response.text,
        "error": None,
        "status_code": response.status_code,
        "final_url": response.url,
        "content_type": content_type,
        "source_url": url,
    }


def _fetch_with_urllib(url: str, headers: dict[str, str]) -> dict[str, object]:
    """Fallback fetch path when requests is unavailable."""
    try:
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = ssl.create_default_context()
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20, context=context) as response:
            content_type = response.headers.get("content-type", "")
            body = response.read().decode("utf-8", errors="replace")
            if not _is_supported_content_type(content_type):
                return {
                    "ok": False,
                    "content": None,
                    "error": f"unsupported content type: {content_type or 'unknown'}",
                    "status_code": getattr(response, "status", None),
                    "final_url": response.geturl(),
                    "content_type": content_type,
                    "source_url": url,
                }
            if not body.strip():
                return {
                    "ok": False,
                    "content": None,
                    "error": "empty response",
                    "status_code": getattr(response, "status", None),
                    "final_url": response.geturl(),
                    "content_type": content_type,
                    "source_url": url,
                }
            return {
                "ok": True,
                "content": body,
                "error": None,
                "status_code": getattr(response, "status", None),
                "final_url": response.geturl(),
                "content_type": content_type,
                "source_url": url,
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "content": None,
            "error": f"HTTP {exc.code} {exc.reason}",
            "status_code": exc.code,
            "final_url": url,
            "content_type": "",
            "source_url": url,
        }
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, ssl.SSLError):
            return _fetch_failure(url, "SSL error", reason)
        return _fetch_failure(url, "connection error", reason)
    except TimeoutError as exc:
        return _fetch_failure(url, "timeout", exc)
    except Exception as exc:
        return _fetch_failure(url, "unexpected error", exc)


def _fetch_failure(url: str, label: str, exc) -> dict[str, object]:
    """Build one structured failure payload."""
    detail = str(exc).strip() or label
    if detail.lower().startswith(label.lower()):
        error = detail
    else:
        error = f"{label}: {detail}"
    return {
        "ok": False,
        "content": None,
        "error": error,
        "status_code": None,
        "final_url": url,
        "content_type": "",
        "source_url": url,
    }


def _normalize_fetch_result(result, *, source_url: str) -> dict[str, object]:
    """Accept either the structured fetch payload or the older test helper values."""
    if isinstance(result, dict):
        normalized = dict(result)
        normalized.setdefault("ok", bool(normalized.get("content")))
        normalized.setdefault("content", None)
        normalized.setdefault("error", None if normalized.get("ok") else "no response")
        normalized.setdefault("final_url", source_url)
        normalized.setdefault("source_url", source_url)
        normalized.setdefault("status_code", None)
        normalized.setdefault("content_type", "")
        return normalized
    if isinstance(result, str):
        return {
            "ok": True,
            "content": result,
            "error": None,
            "status_code": 200,
            "final_url": source_url,
            "content_type": "text/html",
            "source_url": source_url,
        }
    return {
        "ok": False,
        "content": None,
        "error": "no response",
        "status_code": None,
        "final_url": source_url,
        "content_type": "",
        "source_url": source_url,
    }


def _is_supported_content_type(content_type: str) -> bool:
    """Return whether the fetched response can be safely treated as text."""
    lower = (content_type or "").lower()
    if not lower:
        return True
    return (
        lower.startswith("text/")
        or "json" in lower
        or "xml" in lower
        or "javascript" in lower
    )


def _format_failed_url(fetch_result: dict[str, object]) -> str:
    """Format one failed URL with a concise reviewer-facing reason."""
    source_url = str(fetch_result.get("source_url") or fetch_result.get("final_url") or "URL")
    error = str(fetch_result.get("error") or "unknown error").strip()
    domain = urlparse(source_url).netloc or source_url
    return f"{domain} ({error})"


def _summarize_failed_urls(failed_urls: list[str]) -> str:
    """Summarize one or more failed URL fetches for the dashboard result table."""
    if not failed_urls:
        return ""
    if len(failed_urls) == 1:
        return f"Failed: {failed_urls[0]}."
    sample = "; ".join(failed_urls[:2])
    extra = len(failed_urls) - 2
    if extra > 0:
        sample += f"; +{extra} more"
    return f"Failed URLs: {sample}."


def _read_last_refreshed(filepath: Path) -> str:
    """Extract the last refreshed date from one local docs file."""
    try:
        content = filepath.read_text(encoding="utf-8")
    except OSError:
        return "—"

    match = re.search(r"^- \*\*Last refreshed\*\*: (.+)$", content, flags=re.MULTILINE)
    if not match:
        return "—"
    return match.group(1).strip()


def _parse_refresh_value(value: str) -> datetime | None:
    """Parse a stored refresh date/timestamp into a comparable datetime."""
    normalized = value.strip()
    if not normalized:
        return None

    try:
        if len(normalized) == 10:
            parsed_date = date.fromisoformat(normalized)
            return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)
        parsed_dt = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if parsed_dt.tzinfo is None:
            return parsed_dt.replace(tzinfo=timezone.utc)
        return parsed_dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _build_updated_doc_text(
    *,
    source: dict[str, object],
    existing_text: str | None,
    fetched_pages: list[dict[str, str]],
    refreshed_on: str,
) -> str:
    """Preserve existing notes while refreshing machine-managed preview content."""
    base_text = existing_text or _new_doc_skeleton(source)
    with_metadata = _upsert_metadata_block(base_text, source, refreshed_on)
    preview_block = _build_preview_block(fetched_pages)
    return _replace_or_append_preview_block(with_metadata, preview_block)


def _new_doc_skeleton(source: dict[str, object]) -> str:
    """Create a safe starter document if the file does not yet exist."""
    urls = ", ".join(_source_urls(source))
    return (
        f"# {source['name']}\n\n"
        f"- **Official source**: {urls}\n"
        "- **Last refreshed**: —\n"
        "- **source_type**: official_docs\n\n"
        "## Overview\n\n"
        "External documentation snapshot created by the manual Dashboard updater.\n"
    )


def _upsert_metadata_block(text: str, source: dict[str, object], refreshed_on: str) -> str:
    """Update key metadata lines while preserving the rest of the document."""
    urls = ", ".join(_source_urls(source))
    updated = _upsert_metadata_line(text, "**Official source**", urls)
    updated = _upsert_metadata_line(updated, "**Last refreshed**", refreshed_on)
    if "- **source_type**: official_docs" not in updated:
        updated = _upsert_metadata_line(updated, "**source_type**", "official_docs")
    return updated


def _upsert_metadata_line(text: str, label: str, value: str) -> str:
    """Replace one metadata line when present, or insert it after the title block."""
    line = f"- {label}: {value}"
    pattern = re.compile(rf"^- {re.escape(label)}: .*$", flags=re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(line, text, count=1)

    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0] + "\n" + line + "\n\n" + parts[1]
    return text.rstrip() + "\n" + line + "\n"


def _build_preview_block(fetched_pages: list[dict[str, str]]) -> str:
    """Build the machine-managed preview block from fetched source content."""
    sections = []
    for page in fetched_pages:
        preview = _preview_text(page["content"])
        sections.append(
            f"### {page['url']}\n\n"
            f"```\n{preview}\n```"
        )

    sections_text = "\n\n".join(sections)
    return (
        f"{AUTO_PREVIEW_START}\n"
        "## Latest Official Preview\n\n"
        "This machine-generated section is refreshed from the configured external "
        "documentation URLs.\n\n"
        f"{sections_text}\n\n"
        "*Run Rebuild KB Index in the Dashboard to make refreshed docs available to retrieval.*\n"
        f"{AUTO_PREVIEW_END}"
    )


def _replace_or_append_preview_block(text: str, preview_block: str) -> str:
    """Replace the machine-managed preview block, or append it if not present."""
    pattern = re.compile(
        re.escape(AUTO_PREVIEW_START) + r".*?" + re.escape(AUTO_PREVIEW_END),
        flags=re.DOTALL,
    )
    if pattern.search(text):
        updated = pattern.sub(preview_block, text, count=1)
    else:
        updated = text.rstrip() + "\n\n" + preview_block
    return updated.rstrip() + "\n"


def _preview_text(content: str, max_chars: int = 900) -> str:
    """Convert fetched page content into a compact preview for the Markdown file."""
    text = _extract_readable_text(content)
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = cleaned.replace("```", "'''")
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 3].rstrip() + "..."
    return cleaned


def _extract_readable_text(content: str) -> str:
    """Convert fetched HTML or text into compact readable preview text."""
    if not content:
        return ""

    if not _looks_like_html(content):
        return unescape(content)

    parser = _PreviewHTMLTextExtractor()
    try:
        parser.feed(content)
        parser.close()
    except Exception:
        # Fall back to a simple tag strip if parsing is interrupted by malformed HTML.
        fallback = re.sub(r"(?is)<(script|style|head|title|noscript).*?>.*?</\1>", " ", content)
        fallback = re.sub(r"(?is)<!--.*?-->", " ", fallback)
        fallback = re.sub(r"(?is)<[^>]+>", " ", fallback)
        return unescape(fallback)

    return parser.get_text()


def _looks_like_html(content: str) -> bool:
    """Return whether the fetched content is likely HTML markup."""
    snippet = content[:2000].lower()
    return bool(
        re.search(
            r"<!doctype html|<html\b|<head\b|<body\b|<script\b|<style\b|<meta\b|<link\b|<div\b|<p\b",
            snippet,
        )
    )


class _PreviewHTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML while skipping non-content elements."""

    _IGNORE_TAGS = {"head", "script", "style", "noscript", "svg", "iframe", "template"}
    _BLOCK_TAGS = {
        "article", "aside", "blockquote", "br", "div", "dl", "dt", "dd", "figcaption",
        "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
        "li", "main", "nav", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lower = tag.lower()
        if lower in self._IGNORE_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth == 0 and lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in self._IGNORE_TAGS:
            if self._ignore_depth > 0:
                self._ignore_depth -= 1
            return
        if self._ignore_depth == 0 and lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if data and data.strip():
            self._parts.append(data)

    def handle_comment(self, data: str) -> None:  # noqa: ARG002
        return

    def handle_decl(self, decl: str) -> None:  # noqa: ARG002
        return

    def get_text(self) -> str:
        """Return normalized visible text extracted from the HTML."""
        return unescape(" ".join(self._parts))


def _safe_write_text(filepath: Path, content: str) -> None:
    """Write one updated doc file atomically to avoid partial truncation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=filepath.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            tmp_path = Path(handle.name)
        tmp_path.replace(filepath)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
