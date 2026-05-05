"""Tests for retrieval validation module."""

from pathlib import Path

import pytest

from src.eval.retrieval_validation import (
    CaseResult,
    EvalCase,
    ValidationSummary,
    format_validation_report,
    parse_eval_cases,
    validate_retrieval,
)
from src.kb.loader import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_EVAL_MD = """\
# Retrieval Evaluation Cases

## 1. Basic Concept Retrieval

Query:
"What is RAG?"

Expected:
- rag_basics.md

---

Query:
"What is an AI agent?"

Expected:
- ai_agents_intro.md

---

## 2. Multi-File Retrieval

Query:
"Difference between chains and graphs"

Expected:
- langchain_chains.md
- langgraph_advanced_agents.md

---

## 3. Edge Cases

Query:
"Explain quantum computing"

Expected:
- No relevant documents (out-of-domain)

---
"""


@pytest.fixture()
def eval_file(tmp_path: Path) -> Path:
    """Create a temporary eval cases file."""
    fp = tmp_path / "eval_cases.md"
    fp.write_text(SAMPLE_EVAL_MD, encoding="utf-8")
    return fp


def _make_doc(filename: str, content: str = "chunk") -> Document:
    """Helper to create a Document with filename metadata."""
    return Document(content=content, metadata={"filename": filename})


# ---------------------------------------------------------------------------
# Tests — Parsing
# ---------------------------------------------------------------------------

class TestParseEvalCases:
    """Tests for parse_eval_cases."""

    def test_parse_basic_cases(self, eval_file: Path) -> None:
        cases = parse_eval_cases(eval_file)
        assert len(cases) == 4

    def test_parse_single_expected_file(self, eval_file: Path) -> None:
        cases = parse_eval_cases(eval_file)
        rag_case = cases[0]
        assert rag_case.query == "What is RAG?"
        assert rag_case.expected_files == ["rag_basics.md"]

    def test_parse_multi_expected_files(self, eval_file: Path) -> None:
        cases = parse_eval_cases(eval_file)
        multi_case = cases[2]
        assert multi_case.query == "Difference between chains and graphs"
        assert "langchain_chains.md" in multi_case.expected_files
        assert "langgraph_advanced_agents.md" in multi_case.expected_files

    def test_parse_out_of_domain_case(self, eval_file: Path) -> None:
        cases = parse_eval_cases(eval_file)
        ood_case = cases[3]
        assert ood_case.query == "Explain quantum computing"
        assert ood_case.expected_files == []

    def test_parse_categories(self, eval_file: Path) -> None:
        cases = parse_eval_cases(eval_file)
        assert "Basic Concept Retrieval" in cases[0].category
        assert "Multi-File Retrieval" in cases[2].category
        assert "Edge Cases" in cases[3].category

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        cases = parse_eval_cases(tmp_path / "nonexistent.md")
        assert cases == []

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        fp = tmp_path / "empty.md"
        fp.write_text("", encoding="utf-8")
        cases = parse_eval_cases(fp)
        assert cases == []


# ---------------------------------------------------------------------------
# Tests — Parsing the real eval file
# ---------------------------------------------------------------------------

class TestParseRealEvalFile:
    """Ensure the real data/eval/retrieval_eval_cases.md parses correctly."""

    EVAL_FILE = Path("data/eval/retrieval_eval_cases.md")

    @pytest.mark.skipif(
        not Path("data/eval/retrieval_eval_cases.md").exists(),
        reason="Real eval file not present",
    )
    def test_real_eval_file_parses(self) -> None:
        cases = parse_eval_cases(self.EVAL_FILE)
        assert len(cases) >= 10
        queries = [c.query for c in cases]
        assert "What is RAG?" in queries

    @pytest.mark.skipif(
        not Path("data/eval/retrieval_eval_cases.md").exists(),
        reason="Real eval file not present",
    )
    def test_real_eval_file_has_out_of_domain(self) -> None:
        cases = parse_eval_cases(self.EVAL_FILE)
        ood_cases = [c for c in cases if not c.expected_files]
        assert len(ood_cases) >= 1


# ---------------------------------------------------------------------------
# Tests — data/eval and data/meta not loaded as KB content
# ---------------------------------------------------------------------------

class TestDataIsolation:
    """Ensure eval/meta files are not loaded as normal KB content."""

    def test_data_eval_not_in_raw(self) -> None:
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            raw_files = {f.name for f in raw_dir.iterdir() if f.is_file()}
            assert "retrieval_eval_cases.md" not in raw_files
            assert "kb_index.md" not in raw_files

    def test_loader_only_reads_raw(self) -> None:
        from src.kb.loader import load_documents

        raw_dir = Path("data/raw")
        if raw_dir.exists():
            docs = load_documents(raw_dir)
            filenames = {d.metadata["filename"] for d in docs}
            assert "retrieval_eval_cases.md" not in filenames
            assert "kb_index.md" not in filenames


# ---------------------------------------------------------------------------
# Tests — Validation logic
# ---------------------------------------------------------------------------

class TestValidateRetrieval:
    """Tests for validate_retrieval."""

    def test_all_pass(self) -> None:
        cases = [
            EvalCase(query="q1", expected_files=["a.md"]),
            EvalCase(query="q2", expected_files=["b.md"]),
        ]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            mapping = {"q1": ["a.md"], "q2": ["b.md"]}
            return [_make_doc(f) for f in mapping.get(query, [])]

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.pass_rate == 100.0

    def test_partial_pass(self) -> None:
        cases = [
            EvalCase(query="q1", expected_files=["a.md"]),
            EvalCase(query="q2", expected_files=["b.md"]),
        ]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            if query == "q1":
                return [_make_doc("a.md")]
            return [_make_doc("wrong.md")]

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.pass_rate == 50.0

    def test_out_of_domain_always_passes(self) -> None:
        cases = [EvalCase(query="quantum", expected_files=[])]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            return []

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.passed == 1

    def test_multi_expected_files(self) -> None:
        cases = [
            EvalCase(query="multi", expected_files=["a.md", "b.md"]),
        ]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            return [_make_doc("a.md"), _make_doc("b.md"), _make_doc("c.md")]

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.passed == 1
        assert summary.results[0].missing_files == []

    def test_missing_files_tracked(self) -> None:
        cases = [
            EvalCase(query="q", expected_files=["a.md", "b.md"]),
        ]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            return [_make_doc("a.md")]

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.failed == 1
        assert "b.md" in summary.results[0].missing_files

    def test_retrieval_error_handled(self) -> None:
        cases = [EvalCase(query="boom", expected_files=["a.md"])]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            raise RuntimeError("API error")

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.failed == 1
        assert summary.results[0].retrieved_files == []

    def test_empty_cases(self) -> None:
        summary = validate_retrieval([], lambda q, top_k=5: [])
        assert summary.total == 0
        assert summary.pass_rate == 0.0

    def test_duplicate_filenames_deduped(self) -> None:
        cases = [EvalCase(query="q", expected_files=["a.md"])]

        def mock_retrieval(query: str, top_k: int = 5) -> list[Document]:
            return [_make_doc("a.md"), _make_doc("a.md"), _make_doc("a.md")]

        summary = validate_retrieval(cases, mock_retrieval)
        assert summary.passed == 1
        assert summary.results[0].retrieved_files == ["a.md"]


# ---------------------------------------------------------------------------
# Tests — Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    """Tests for result dataclass structure."""

    def test_case_result_fields(self) -> None:
        r = CaseResult(
            query="test",
            expected_files=["a.md"],
            retrieved_files=["a.md", "b.md"],
            passed=True,
            missing_files=[],
        )
        assert r.query == "test"
        assert r.passed is True

    def test_summary_fields(self) -> None:
        s = ValidationSummary(
            total=10, passed=8, failed=2, pass_rate=80.0,
        )
        assert s.total == 10
        assert s.results == []


# ---------------------------------------------------------------------------
# Tests — Report formatting
# ---------------------------------------------------------------------------

class TestFormatReport:
    """Tests for format_validation_report."""

    def test_all_pass_report(self) -> None:
        summary = ValidationSummary(
            total=3, passed=3, failed=0, pass_rate=100.0,
        )
        report = format_validation_report(summary)
        assert "Pass rate:   100.0%" in report
        assert "Failed Cases" not in report

    def test_failed_report_includes_details(self) -> None:
        summary = ValidationSummary(
            total=2, passed=1, failed=1, pass_rate=50.0,
            results=[
                CaseResult(
                    query="bad query",
                    expected_files=["x.md"],
                    retrieved_files=["y.md"],
                    passed=False,
                    missing_files=["x.md"],
                ),
            ],
        )
        report = format_validation_report(summary)
        assert "Failed Cases" in report
        assert "bad query" in report
        assert "x.md" in report
