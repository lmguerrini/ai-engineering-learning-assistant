"""Tests for the scoped Help Assistant service."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.kb.loader import Document


def _make_llm_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=80, total_tokens=200),
    )


class TestHelpAssistantScope:
    def test_personality_mode_defaults_to_technical(self):
        from src.services.help_assistant import (
            classify_technical_help_question,
            get_help_assistant_personality_profiles,
            get_help_assistant_personality_modes,
            get_help_assistant_runtime_defaults,
            normalize_help_assistant_personality_mode,
        )

        assert normalize_help_assistant_personality_mode(None) == "Technical"
        assert get_help_assistant_personality_modes() == [
            "Technical",
            "Concise",
            "Friendly",
            "Formal",
        ]
        defaults = get_help_assistant_runtime_defaults(None)
        assert defaults["temperature"] == 0.15
        assert defaults["top_p"] == 0.85
        assert classify_technical_help_question("How does this app work?") == "product_overview"
        assert (
            classify_technical_help_question("Why does app-workflow context skip retrieval?")
            == "runtime_architecture"
        )
        assert (
            classify_technical_help_question("How does tool calling work in LangChain?")
            == "runtime_architecture"
        )
        profiles = get_help_assistant_personality_profiles()
        assert profiles["Technical"]["tone"] == "Pragmatic senior AI engineer"
        assert "Adaptive technical mode" in profiles["Technical"]["output_style"]
        assert profiles["Concise"]["verbosity"] == "Low"
        assert profiles["Friendly"]["best_use_case"] == "Explanations for newer users or softer onboarding"
        assert profiles["Formal"]["output_style"] == "Polished documentation-style answer with explicit sectioning and restrained wording"

    def test_in_domain_queries_are_allowed(self):
        from src.services.help_assistant import is_help_query_in_domain

        assert is_help_query_in_domain("Explain LangGraph state management")
        assert is_help_query_in_domain("How should I evaluate a RAG pipeline with RAGAs?")

    def test_app_workflow_queries_are_allowed(self):
        from src.services.help_assistant import is_help_query_in_domain

        assert is_help_query_in_domain("How does this app work?")
        assert is_help_query_in_domain("What does the dashboard show?")
        assert is_help_query_in_domain("How do Learn and Quiz work?")
        assert is_help_query_in_domain("What is Official Docs Sync?")
        assert is_help_query_in_domain("How does KB index work?")
        assert is_help_query_in_domain("How does the app use official docs alongside curated notes?")
        assert is_help_query_in_domain("What is the difference between official snapshots and live docs enrichment?")
        assert is_help_query_in_domain("When does Help Assistant use live official docs?")
        assert is_help_query_in_domain("Did you use live official docs enrichment to answer that question?")

    def test_out_of_domain_queries_are_rejected(self):
        from src.services.help_assistant import is_help_query_in_domain

        assert is_help_query_in_domain("What is the weather in Rome?") is False
        assert is_help_query_in_domain("Who won the election?") is False

    def test_follow_up_queries_are_allowed_when_history_exists(self):
        from src.services.help_assistant import is_help_query_in_domain

        history = [{"question": "Difference between RAG and Agentic RAG"}]
        assert is_help_query_in_domain(
            "What was my last question?",
            conversation_history=history,
        )
        assert is_help_query_in_domain(
            "Can you explain that more simply?",
            conversation_history=history,
        )
        assert is_help_query_in_domain(
            "Did you use live official docs enrichment to answer that question?",
            conversation_history=history,
        )
        assert is_help_query_in_domain(
            "qhat was my lasts question?",
            conversation_history=history,
        )

    def test_follow_up_queries_without_history_remain_out_of_domain(self):
        from src.services.help_assistant import is_help_query_in_domain

        assert is_help_query_in_domain("What was my last question?") is False

    def test_scope_configuration_exposes_examples_and_domains(self):
        from src.services.help_assistant import get_help_assistant_scope

        scope = get_help_assistant_scope()

        assert "out_of_domain_message" in scope
        assert "Explain LangGraph state management" in scope["example_prompts"]
        assert "langgraph" in scope["approved_domains"]
        assert "openai" in scope["approved_domains"]

    def test_live_source_selection_stays_within_approved_registry(self):
        from src.services.help_assistant import (
            get_help_assistant_source_registry,
            select_live_help_sources,
        )

        selected = select_live_help_sources("How does tool calling work in LangChain?")
        approved_filenames = {row["filename"] for row in get_help_assistant_source_registry()}

        assert selected
        assert all(row["filename"] in approved_filenames for row in selected)
        assert any(row["domain"] in {"langchain", "tools", "openai"} for row in selected)

    def test_live_source_registry_uses_only_official_docs_domains(self):
        from src.services.help_assistant import get_help_assistant_source_registry

        registry = get_help_assistant_source_registry()
        urls = [url for row in registry for url in row["urls"]]
        assert urls
        assert all("arxiv.org" not in url for url in urls)


class TestHelpAssistantBehavior:
    def test_out_of_domain_refusal_skips_openai(self):
        from src.services.help_assistant import OUT_OF_DOMAIN_MESSAGE, answer_help_question

        with patch("src.services.help_assistant.OpenAI") as mock_openai:
            result = answer_help_question("Plan a weekend trip to Venice.")

        assert result["status"] == "refused"
        assert result["message"] == OUT_OF_DOMAIN_MESSAGE
        assert result["sources"] == []
        assert result["live_enrichment_used"] is False
        mock_openai.assert_not_called()

    def test_answer_uses_local_and_live_context_when_available(self):
        from src.services.help_assistant import answer_help_question

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '{"answer_markdown":"## Answer\\nLangGraph state management keeps graph state explicit and inspectable.\\n\\nSources used\\n- Curated KB\\n- Official Snapshot\\n- Live Official Docs"}'
        )

        curated = [
            Document(
                content="LangGraph state keeps node outputs in one evolving graph object.",
                metadata={"topic": "LangGraph State", "filename": "langgraph_state.md"},
            )
        ]
        official = [
            Document(
                content="Official snapshot about reducers, nodes, and edges.",
                metadata={
                    "topic": "LangGraph Official Docs",
                    "filename": "langgraph_state_orchestration.md",
                    "domain": "langgraph",
                },
            )
        ]

        with patch(
            "src.services.help_assistant.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key", app_default_model="gpt-4o-mini"),
        ), patch(
            "src.services.help_assistant.retrieve_documents",
            return_value=curated,
        ), patch(
            "src.services.help_assistant.retrieve_official_docs",
            return_value=official,
        ), patch(
            "src.services.help_assistant.OpenAI",
            return_value=mock_client,
        ), patch(
            "src.services.help_assistant.wrap_openai",
            side_effect=lambda client: client,
        ):
            result = answer_help_question(
                "Explain LangGraph state management",
                live_fetcher=lambda _url: "<html><body><h1>StateGraph</h1><p>Reducers merge node outputs into shared state.</p></body></html>",
            )

        assert result["status"] == "answered"
        assert result["live_enrichment_used"] is True
        assert "LangGraph state management" in result["question"]
        assert any(row["Kind"] == "Curated KB" for row in result["sources"])
        assert any(row["Kind"] == "Official Snapshot" for row in result["sources"])
        assert any(row["Kind"] == "Live Official Docs" for row in result["sources"])
        assert result["usage_records"][0]["operation"] == "help_assistant_answer"
        assert any("select_live_sources" in step for step in result["trace"])

    def test_last_question_follow_up_can_be_answered_from_history(self):
        from src.services.help_assistant import answer_help_question

        history = [{"question": "How does this app work?", "answer_markdown": "It has Learn and Quiz workflows."}]
        with patch("src.services.help_assistant.OpenAI") as mock_openai:
            result = answer_help_question(
                "What was my last question?",
                conversation_history=history,
            )

        assert result["status"] == "answered"
        assert "How does this app work?" in result["answer_markdown"]
        assert result["sources"] == []
        assert result["usage_records"] == []
        mock_openai.assert_not_called()

    def test_live_enrichment_follow_up_can_be_answered_from_history(self):
        from src.services.help_assistant import answer_help_question

        history = [
            {
                "question": "How does tool calling work in LangChain?",
                "answer_markdown": "It can call structured tools.",
                "live_enrichment_used": True,
                "sources": [
                    {"Kind": "Curated KB", "Title": "KB"},
                    {"Kind": "Live Official Docs", "Title": "Live"},
                ],
            }
        ]
        with patch("src.services.help_assistant.OpenAI") as mock_openai:
            result = answer_help_question(
                "Did you use live official docs enrichment to answer that question?",
                conversation_history=history,
            )

        assert result["status"] == "answered"
        assert "Yes." in result["answer_markdown"]
        assert result["live_enrichment_used"] is True
        assert any(row["Kind"] == "Live Official Docs" for row in result["sources"])
        mock_openai.assert_not_called()

    def test_technical_overview_questions_use_product_overview_routing(self):
        from src.services.help_assistant import answer_help_question

        captured: dict[str, str] = {}

        def _fake_call_help_llm(**kwargs):
            captured["prompt"] = str(kwargs["prompt"])
            return _make_llm_response(
                '{"answer_markdown":"App-workflow questions skip retrieval because built-in workflow context is already deterministic."}'
            )

        mock_client = MagicMock()
        with patch(
            "src.services.help_assistant.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key", app_default_model="gpt-4o-mini"),
        ), patch(
            "src.services.help_assistant.retrieve_documents",
        ) as mock_curated, patch(
            "src.services.help_assistant.retrieve_official_docs",
        ) as mock_official, patch(
            "src.services.help_assistant.OpenAI",
            return_value=mock_client,
        ), patch(
            "src.services.help_assistant.wrap_openai",
            side_effect=lambda client: client,
        ), patch(
            "src.services.help_assistant._call_help_llm",
            side_effect=_fake_call_help_llm,
        ):
            result = answer_help_question(
                "How does this app work?",
                personality_mode="Technical",
            )

        assert result["status"] == "answered"
        assert result["live_enrichment_used"] is False
        assert result["sources"] == []
        assert "App Workflow Context:" in captured["prompt"]
        assert "Response style: Technical." in captured["prompt"]
        assert "Technical question category: product_overview." in captured["prompt"]
        assert "compact technical overview of how the system is composed" in captured["prompt"]
        assert "Start from the architectural split, runtime boundary, or concrete mechanism directly." in captured["prompt"]
        assert "Avoid generic definitional openings." in captured["prompt"]
        assert "Mention major flows or surfaces only when they help explain the architecture." in captured["prompt"]
        assert "Runtime Info:" in captured["prompt"]
        assert "RAGAs Evaluation:" in captured["prompt"]
        assert "KB Index:" in captured["prompt"]
        mock_curated.assert_not_called()
        mock_official.assert_not_called()

    def test_technical_runtime_questions_use_runtime_architecture_routing(self):
        from src.services.help_assistant import answer_help_question

        captured: dict[str, str] = {}

        def _fake_call_help_llm(**kwargs):
            captured["prompt"] = str(kwargs["prompt"])
            return _make_llm_response(
                '{"answer_markdown":"App-workflow questions skip retrieval because built-in workflow context is already deterministic."}'
            )

        with patch(
            "src.services.help_assistant.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key", app_default_model="gpt-4o-mini"),
        ), patch(
            "src.services.help_assistant.retrieve_documents",
        ) as mock_curated, patch(
            "src.services.help_assistant.retrieve_official_docs",
        ) as mock_official, patch(
            "src.services.help_assistant.OpenAI",
            return_value=MagicMock(),
        ), patch(
            "src.services.help_assistant.wrap_openai",
            side_effect=lambda client: client,
        ), patch(
            "src.services.help_assistant._call_help_llm",
            side_effect=_fake_call_help_llm,
        ):
            result = answer_help_question(
                "Why does app-workflow context skip retrieval in this app?",
                personality_mode="Technical",
            )

        assert result["status"] == "answered"
        assert result["sources"] == []
        assert "Technical question category: runtime_architecture." in captured["prompt"]
        assert "Start from the concrete mechanism, execution boundary, or orchestration behavior directly." in captured["prompt"]
        assert "Avoid generic definitional openings." in captured["prompt"]
        assert "Lead with routing, orchestration, state flow, retrieval boundaries" in captured["prompt"]
        assert "what to inspect when it goes wrong" in captured["prompt"]
        mock_curated.assert_not_called()
        mock_official.assert_not_called()

    def test_technical_runtime_questions_use_runtime_architecture_for_langchain_tool_calling(self):
        from src.services.help_assistant import _build_help_assistant_prompt

        prompt = _build_help_assistant_prompt(
            question="How does tool calling work in LangChain?",
            context_question="How does tool calling work in LangChain?",
            curated_docs=[],
            official_docs=[],
            live_sources=[],
            conversation_history=[],
            personality_mode="Technical",
        )

        assert "Technical question category: runtime_architecture." in prompt
        assert "Start from the concrete mechanism, execution boundary, or orchestration behavior directly." in prompt
        assert "Lead with routing, orchestration, state flow, retrieval boundaries" in prompt

    def test_prompt_mentions_grounded_scope_and_live_docs(self):
        from src.services.help_assistant import _build_help_assistant_prompt

        prompt = _build_help_assistant_prompt(
            question="How does tool calling work?",
            curated_docs=[],
            official_docs=[],
            live_sources=[],
            conversation_history=[{"question": "How does this app work?", "answer_markdown": "It has Learn and Quiz workflows."}],
            personality_mode="Concise",
        )

        assert "Use ONLY the grounded context provided below." in prompt
        assert "Live Official Docs" in prompt
        assert "Do NOT include a `Sources`, `Sources used`, or bibliography section" in prompt
        assert "NEVER answer as JSON, a dict, a schema, or a key/value object" in prompt
        assert "ALWAYS answer as natural markdown prose" in prompt
        assert "Recent conversation:" in prompt
        assert "App Workflow Context:" in prompt
        assert "Response style: Concise." in prompt
        assert "Make the answer clearly shorter than the other personalities would." in prompt
        assert "Return valid JSON" in prompt

    def test_personality_contracts_are_visibly_different(self):
        from src.services.help_assistant import _format_personality_mode_instruction

        technical = _format_personality_mode_instruction("Technical")
        concise = _format_personality_mode_instruction("Concise")
        friendly = _format_personality_mode_instruction("Friendly")
        formal = _format_personality_mode_instruction("Formal")

        assert "senior ai engineer" in technical.lower()
        assert "implementation review" in technical.lower()
        assert "engineer handoff" in technical.lower()
        assert "reader already sees the ui" in technical.lower()
        assert "question type control the framing" in technical.lower()
        assert "compact system overview" in technical.lower()
        assert "runtime reasoning" in technical.lower()
        assert "engineer-to-engineer explanation" in technical.lower()
        assert "clearly shorter" in concise.lower()
        assert "natural conversational pacing" in friendly.lower()
        assert "short paragraphs over bullets" in friendly.lower()
        assert "documentation-like" in formal.lower()
        assert "explicit headings" in formal.lower()

    def test_help_turn_caption_reflects_app_context_vs_retrieval_sources(self):
        from src.ui.help_page import _get_help_answer_caption

        assert (
            _get_help_answer_caption(
                {
                    "sources": [],
                    "trace": ["retrieve_local_context: skipped — app workflow context preferred"],
                    "live_enrichment_used": False,
                    "personality_label": "Technical",
                }
            )
            == "Answered from app workflow context. | Agent Personality: Technical"
        )
        assert (
            _get_help_answer_caption(
                {
                    "question": "When does Help Assistant use live official docs?",
                    "sources": [],
                    "trace": ["retrieve_local_context: skipped — app workflow context preferred"],
                    "live_enrichment_used": False,
                    "personality_label": "Technical",
                }
            )
            == "Answered from app workflow / live-docs policy context. | Agent Personality: Technical"
        )
        assert (
            _get_help_answer_caption(
                {
                    "sources": [{"Kind": "Curated KB", "Title": "A"}],
                    "trace": [],
                    "live_enrichment_used": False,
                    "personality_label": "Custom",
                }
            )
            == "Answered from local curated and official-doc snapshot context. | Agent Personality: Custom"
        )

    def test_help_turn_caption_reflects_live_enrichment_and_personality(self):
        from src.ui.help_page import _get_help_answer_caption

        assert (
            _get_help_answer_caption(
                {
                    "sources": [{"Kind": "Live Official Docs", "Title": "A"}],
                    "trace": [],
                    "live_enrichment_used": True,
                    "personality_label": "Friendly",
                }
            )
            == "Live official docs enrichment was used for this answer. | Agent Personality: Friendly"
        )

    def test_help_trace_entries_are_human_readable(self):
        from src.ui.help_page import _format_help_trace_entries

        entries = _format_help_trace_entries(
            [
                "validate_scope: started",
                "retrieve_local_context: skipped — app workflow context preferred",
                "select_live_sources: started",
                "select_live_sources: selected=2 from approved registry",
                "select_live_sources: skipped — app workflow question",
                "answer_generation: started — model=gpt-4o-mini",
                "answer_generation: completed — 120 chars",
            ]
        )

        assert "Scope check: passed" in entries
        assert "Local retrieval: skipped because this was an app-workflow question" in entries
        assert (
            "Live docs: skipped because no live source was needed" in entries
            or "Live docs: selected approved official-doc sources" in entries
        )
        assert "Generation: started with gpt-4o-mini" in entries
        assert "Generation: completed" in entries
        assert entries.count("Live docs: selected approved official-doc sources") <= 1

    def test_execution_trace_formats_app_workflow_skip_path(self):
        from src.ui.help_page import _format_help_execution_trace

        entries = _format_help_execution_trace(
            [
                "validate_scope: started",
                "retrieve_local_context: skipped — app workflow context preferred",
                "select_live_sources: skipped — app workflow question",
                "answer_generation: started — model=gpt-4o-mini",
                "answer_generation: completed — 120 chars",
            ]
        )

        assert entries[0] == "✓ Scope validation passed"
        assert any("✓ App workflow context selected" in entry for entry in entries)
        assert any("• Local KB retrieval skipped" in entry for entry in entries)
        assert any("• Live docs skipped" in entry for entry in entries)
        assert any("✓ Response generated" in entry and "• Model: gpt-4o-mini" in entry for entry in entries)

    def test_execution_trace_formats_retrieval_and_live_enrichment_path(self):
        from src.ui.help_page import _format_help_execution_trace

        entries = _format_help_execution_trace(
            [
                "validate_scope: started",
                "retrieve_local_context: curated=4, official_snapshot=4",
                "select_live_sources: selected=2 from approved registry",
                "live_fetch: LangGraph Official Docs fetched",
                "live_fetch: OpenAI Docs fetched",
                "answer_generation: started — model=gpt-4o-mini",
                "answer_generation: completed — 300 chars",
            ]
        )

        assert any("✓ Local KB retrieval completed" in entry for entry in entries)
        assert any("• Curated KB sources: 4" in entry for entry in entries)
        assert any("• Official snapshot sources: 4" in entry for entry in entries)
        assert any("✓ Live official-doc enrichment completed" in entry for entry in entries)
        assert any("• Approved live sources selected: 2" in entry for entry in entries)
        assert any("• Live sources fetched: 2" in entry for entry in entries)

    def test_execution_trace_formats_partial_live_fetch_failure_path(self):
        from src.ui.help_page import _format_help_execution_trace

        entries = _format_help_execution_trace(
            [
                "validate_scope: started",
                "retrieve_local_context: curated=2, official_snapshot=3",
                "select_live_sources: selected=3 from approved registry",
                "live_fetch: LangGraph Official Docs fetched",
                "live_fetch: OpenAI Docs fetched",
                "live_fetch: LangSmith Docs failed",
                "live_enrichment: 1 fetch failure(s) preserved as non-fatal",
                "answer_generation: started — model=gpt-4o-mini",
                "answer_generation: completed — 300 chars",
            ]
        )

        assert any("⚠ Live official-doc enrichment partially completed" in entry for entry in entries)
        assert any("• Live sources fetched: 2" in entry for entry in entries)
        assert any("• Live source fetch failures: 1" in entry for entry in entries)

    def test_clean_live_preview_text_replaces_redirect_noise(self):
        from src.services.help_assistant import _clean_live_preview_text

        assert _clean_live_preview_text("Redirecting...") == "Live source fetched; readable preview unavailable."
        assert (
            _clean_live_preview_text("Skip to content Toggle navigation Table of contents Redirecting...")
            == "Live source fetched; readable preview unavailable."
        )

    def test_conversation_context_uses_last_five_turns(self):
        from src.services.help_assistant import _format_conversation_context

        history = [
            {"question": f"Q{i}", "answer_markdown": f"A{i}"}
            for i in range(1, 8)
        ]
        context = _format_conversation_context(history)

        assert "Q1" not in context
        assert "Q2" not in context
        assert "Q3" in context
        assert "Q7" in context

    def test_personality_metadata_marks_custom_runtime(self):
        from src.services.help_assistant import _build_help_personality_metadata

        metadata = _build_help_personality_metadata(
            "Technical",
            {
                "temperature": 0.4,
                "top_p": 0.85,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "max_tokens": 1100,
            },
        )

        assert metadata["personality_label"] == "Custom"
        assert metadata["runtime_is_custom"] is True

    def test_openai_call_receives_runtime_sampling_parameters(self):
        from src.services.help_assistant import _call_help_llm

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _make_llm_response(
            '{"answer_markdown":"ok"}'
        )

        _call_help_llm(
            client=mock_client,
            model="gpt-4o-mini",
            prompt="Prompt",
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.3,
            presence_penalty=0.2,
            max_tokens=900,
        )

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["frequency_penalty"] == 0.3
        assert kwargs["presence_penalty"] == 0.2
        assert kwargs["max_tokens"] == 900

    def test_answer_help_question_passes_runtime_sampling_to_llm_call(self):
        from src.services.help_assistant import answer_help_question

        captured: dict[str, object] = {}

        def _fake_call_help_llm(**kwargs):
            captured.update(kwargs)
            return _make_llm_response('{"answer_markdown":"ok"}')

        with patch(
            "src.services.help_assistant.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key", app_default_model="gpt-4o-mini"),
        ), patch(
            "src.services.help_assistant.retrieve_documents",
            return_value=[],
        ), patch(
            "src.services.help_assistant.retrieve_official_docs",
            return_value=[],
        ), patch(
            "src.services.help_assistant.OpenAI",
            return_value=MagicMock(),
        ), patch(
            "src.services.help_assistant.wrap_openai",
            side_effect=lambda client: client,
        ), patch(
            "src.services.help_assistant._call_help_llm",
            side_effect=_fake_call_help_llm,
        ):
            answer_help_question(
                "Explain LangGraph state management",
                personality_mode="Technical",
                runtime_config={
                    "temperature": 0.55,
                    "top_p": 0.95,
                    "frequency_penalty": 0.15,
                    "presence_penalty": 0.05,
                    "max_tokens": 850,
                },
            )

        assert captured["temperature"] == 0.55
        assert captured["top_p"] == 0.95
        assert captured["frequency_penalty"] == 0.15
        assert captured["presence_penalty"] == 0.05
        assert captured["max_tokens"] == 850

    def test_sources_used_block_is_removed_from_answer(self):
        from src.services.help_assistant import _parse_help_answer

        answer = _parse_help_answer(
            '{"answer_markdown":"## Answer\\nGrounded response.\\n\\n## Sources used\\n- A\\n- B"}'
        )

        assert "Grounded response." in answer
        assert "Sources used" not in answer

    def test_structured_answer_payload_is_coerced_to_markdown(self):
        from src.services.help_assistant import _parse_help_answer

        answer = _parse_help_answer(
            '{"answer_markdown":"{\\"Learn\\": \\"Generates study guides.\\", \\"Quiz\\": \\"Evaluates quiz answers.\\"}"}'
        )

        assert "### Learn" in answer
        assert "Generates study guides." in answer
        assert "### Quiz" in answer
        assert "{" not in answer

    def test_app_workflow_answer_never_leaks_raw_dict_body(self):
        from src.services.help_assistant import answer_help_question

        def _fake_call_help_llm(**_kwargs):
            return _make_llm_response(
                '{"answer_markdown":"{\\"Learn\\": \\"Generates grounded study guides.\\", \\"Dashboard\\": \\"Shows runtime diagnostics.\\"}"}'
            )

        with patch(
            "src.services.help_assistant.get_settings",
            return_value=SimpleNamespace(openai_api_key="test-key", app_default_model="gpt-4o-mini"),
        ), patch(
            "src.services.help_assistant.retrieve_documents",
            return_value=[],
        ), patch(
            "src.services.help_assistant.retrieve_official_docs",
            return_value=[],
        ), patch(
            "src.services.help_assistant.OpenAI",
            return_value=MagicMock(),
        ), patch(
            "src.services.help_assistant.wrap_openai",
            side_effect=lambda client: client,
        ), patch(
            "src.services.help_assistant._call_help_llm",
            side_effect=_fake_call_help_llm,
        ):
            result = answer_help_question("How does this app work?", personality_mode="Technical")

        assert result["status"] == "answered"
        assert "### Learn" in result["answer_markdown"]
        assert "### Dashboard" in result["answer_markdown"]
        assert '{"Learn":' not in result["answer_markdown"]


class TestHelpAssistantChatState:
    @patch("src.ui.help_page.st")
    def test_queue_help_assistant_question_sets_prefill_and_section(self, mock_st):
        from src.ui.help_page import queue_help_assistant_question

        mock_st.session_state = {}
        queue_help_assistant_question("How does this app work?")

        assert mock_st.session_state["help_assistant_question"] == "How does this app work?"
        assert mock_st.session_state["active_section"] == "Help Assistant"
        mock_st.rerun.assert_called_once()

    @patch("src.ui.help_page.st")
    def test_sync_help_assistant_draft_uses_separate_widget_key(self, mock_st):
        from src.ui.help_page import _sync_help_assistant_draft

        mock_st.session_state = {"help_assistant_question": "What does the dashboard show?"}
        _sync_help_assistant_draft()

        assert "help_assistant_question" not in mock_st.session_state
        assert (
            mock_st.session_state["help_assistant_draft_question"]
            == "What does the dashboard show?"
        )

    @patch("src.ui.help_page.st")
    def test_sync_help_assistant_draft_defaults_to_empty_string(self, mock_st):
        from src.ui.help_page import _sync_help_assistant_draft

        mock_st.session_state = {}
        _sync_help_assistant_draft()

        assert mock_st.session_state["help_assistant_draft_question"] == ""

    @patch("src.ui.help_page.st")
    def test_sync_help_assistant_draft_applies_reset_flag_before_widget(self, mock_st):
        from src.ui.help_page import _sync_help_assistant_draft

        mock_st.session_state = {
            "help_assistant_reset_draft": True,
            "help_assistant_draft_question": "Old draft",
        }
        _sync_help_assistant_draft()

        assert "help_assistant_reset_draft" not in mock_st.session_state
        assert mock_st.session_state["help_assistant_draft_question"] == ""

    @patch("src.ui.help_page.st")
    def test_apply_help_style_preset_updates_runtime_defaults(self, mock_st):
        from src.ui.help_page import _apply_help_style_preset

        mock_st.session_state = {}
        _apply_help_style_preset("Friendly")

        assert mock_st.session_state["help_assistant_personality_mode"] == "Friendly"
        assert mock_st.session_state["help_assistant_temperature"] == 0.7
        assert mock_st.session_state["help_assistant_top_p"] == 1.0
        assert mock_st.session_state["help_assistant_frequency_penalty"] == 0.0
        assert mock_st.session_state["help_assistant_presence_penalty"] == 0.3
        assert mock_st.session_state["help_assistant_max_tokens"] == 1100

    @patch("src.ui.help_page.st")
    def test_help_runtime_modified_detects_manual_override(self, mock_st):
        from src.ui.help_page import _help_runtime_is_modified

        mock_st.session_state = {
            "help_assistant_personality_mode": "Technical",
            "help_assistant_temperature": 0.4,
            "help_assistant_top_p": 0.85,
            "help_assistant_frequency_penalty": 0.0,
            "help_assistant_presence_penalty": 0.0,
            "help_assistant_max_tokens": 1100,
        }

        assert _help_runtime_is_modified() is True

    def test_validate_help_submit_rejects_blank_questions(self):
        from src.ui.help_page import _validate_help_submit

        question, message, level = _validate_help_submit("   ")
        assert question == ""
        assert message == "Enter a question before submitting."
        assert level == "warning"

    def test_validate_help_submit_returns_transient_error_for_missing_value(self):
        from src.ui.help_page import _validate_help_submit

        question, message, level = _validate_help_submit(None)
        assert question == ""
        assert message == "Something went wrong. Please try again."
        assert level == "error"

    def test_app_workflow_answers_get_fallback_context_panel(self):
        from src.ui.help_page import _get_help_context_panel

        panel = _get_help_context_panel(
            {
                "status": "answered",
                "sources": [],
                "trace": ["retrieve_local_context: skipped — app workflow context preferred"],
            }
        )

        assert panel is not None
        assert panel[0] == "App workflow context"

    def test_history_answers_get_conversation_context_panel(self):
        from src.ui.help_page import _get_help_context_panel

        panel = _get_help_context_panel(
            {
                "status": "answered",
                "sources": [],
                "trace": ["follow_up_answer: answered from chat history — last question"],
            }
        )

        assert panel is not None
        assert panel[0] == "Conversation context"

    @patch("src.ui.help_page.st")
    def test_chat_history_appends_turns(self, mock_st):
        from src.ui.help_page import _append_help_chat_turn

        mock_st.session_state = {"help_assistant_chat_history": []}
        _append_help_chat_turn({"question": "Q1", "status": "answered", "answer_markdown": "A1"})
        _append_help_chat_turn({"question": "Q2", "status": "answered", "answer_markdown": "A2"})

        history = mock_st.session_state["help_assistant_chat_history"]
        assert [turn["question"] for turn in history] == ["Q1", "Q2"]

    @patch("src.ui.help_page.st")
    def test_clear_chat_resets_history_and_input(self, mock_st):
        from src.ui.help_page import _clear_help_chat

        mock_st.session_state = {
            "help_assistant_chat_history": [{"question": "Q1"}],
            "help_assistant_question": "What does the dashboard show?",
            "help_assistant_draft_question": "What does the dashboard show?",
        }
        _clear_help_chat()

        assert mock_st.session_state["help_assistant_chat_history"] == []
        assert mock_st.session_state["help_assistant_question"] == ""
        assert mock_st.session_state["help_assistant_reset_draft"] is True
        assert mock_st.session_state["help_assistant_draft_question"] == "What does the dashboard show?"

    def test_source_split_groups_grounded_and_live_rows(self):
        from src.ui.help_page import _split_help_sources

        grounded, live = _split_help_sources(
            [
                {"Kind": "Curated KB", "Title": "A"},
                {"Kind": "Official Snapshot", "Title": "B"},
                {"Kind": "Live Official Docs", "Title": "C"},
            ]
        )

        assert len(grounded) == 2
        assert len(live) == 1
