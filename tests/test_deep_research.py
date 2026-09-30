import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from actions import deep_research as research
from agent import planner
from agent.task_queue import TaskQueue
from core.qa_audit import declared_tools


class DeepResearchTests(unittest.TestCase):
    def setUp(self):
        research._pending_parameters = None
        research._pending_created_at = 0.0
        research._latest_result = None

    def test_request_normalizes_depth_focus_and_source_limit(self):
        request = research.ResearchRequest.from_parameters({
            "topic": "Local AI privacy",
            "depth": "thorough",
            "focus_areas": "privacy, Mac performance",
            "max_sources": 200,
        })

        self.assertEqual(request.question, "Local AI privacy")
        self.assertEqual(request.depth, "deep")
        self.assertEqual(request.focus_areas, ["privacy", "Mac performance"])
        self.assertEqual(request.max_sources, 50)

    def test_grounding_metadata_extracts_only_web_urls(self):
        response = SimpleNamespace(candidates=[SimpleNamespace(
            grounding_metadata=SimpleNamespace(grounding_chunks=[
                SimpleNamespace(web=SimpleNamespace(title="Primary study", uri="https://example.org/study")),
                SimpleNamespace(web=SimpleNamespace(title="Unsafe", uri="file:///tmp/result")),
                SimpleNamespace(web=None),
            ])
        )])

        self.assertEqual(research._extract_grounding_sources(response), [{
            "title": "Primary study",
            "url": "https://example.org/study",
        }])

    def test_build_keeps_cited_report_in_memory_without_writing(self):
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "decision.md"
            query_results = [
                ("Evidence from the primary study.", [{
                    "title": "Primary study", "url": "https://example.org/study",
                }]),
                ("Evidence from the benchmark.", [
                    {"title": "Primary duplicate", "url": "https://example.org/study/"},
                    {"title": "Benchmark", "url": "https://example.net/benchmark"},
                ]),
            ]
            with (
                patch("google.genai.Client", return_value=object()),
                patch.object(research, "_api_key", return_value="test-key"),
                patch.object(research, "_plan_queries", return_value=["primary evidence", "benchmarks"]),
                patch.object(research, "_research_query", side_effect=query_results),
                patch.object(research, "_synthesize", return_value="## Answer\nSupported conclusion [1] with context [2]."),
            ):
                result = research.build_deep_research(
                    {"question": "Which option is best?", "output_path": str(output)},
                    progress_callback=lambda **update: progress.append(update),
                )

            self.assertEqual(len(result.sources), 2)
            self.assertEqual(result.artifacts, [])
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".evidence.json").exists())
            self.assertIn("Supported conclusion [1]", result.report_markdown)
            self.assertIn("[Primary study](https://example.org/study)", result.report_markdown)
            self.assertEqual(result.evidence["queries"], ["primary evidence", "benchmarks"])
            self.assertEqual(len(result.evidence["evidence"]), 2)
            self.assertEqual(progress[-1]["percent"], 100)
            self.assertEqual(progress[-1]["phase"], "Deep research complete")

    def test_cancelled_request_never_creates_api_client(self):
        cancel_flag = threading.Event()
        cancel_flag.set()
        with patch("google.genai.Client") as client:
            with self.assertRaises(research.ResearchCancelled):
                research.build_deep_research({"question": "Cancelled"}, cancel_flag=cancel_flag)
        client.assert_not_called()

    def test_owner_policy_blocks_queue_uses_specialized_research_job(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.queue_deep_research({'question': 'Research batteries'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_queue_completion_keeps_report_in_memory_and_requests_detailed_summary(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.queue_deep_research({'question': 'Battery recycling'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_research_task_is_not_written_to_persistent_task_history(self):
        queue = TaskQueue(max_concurrent=1)
        task_id = queue.submit_job(
            "Deep research: private topic",
            lambda cancel_flag, progress: "complete",
            kind="research",
        )
        task = queue._tasks[task_id]
        task.status = task.status.RUNNING
        queue._active_count = 1
        with patch("agent.task_queue.record_task") as record:
            queue._run_task(task)

        record.assert_not_called()

    def test_owner_policy_blocks_background_mode_opens_labeled_status_surface_only(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.queue_deep_research({'question': 'Research batteries', 'execution_mode': 'background'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_visible_mode_opens_browser_instead_of_progress_surface(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.queue_deep_research({'question': 'Research batteries', 'execution_mode': 'visible'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_visible_research_searches_and_opens_a_source_tab(self):
        results = [{
            "title": "Battery study",
            "snippet": "Measured recycling outcomes.",
            "url": "https://example.org/battery-study",
        }]
        browser_results = [
            "Opened: https://www.google.com/search?q=batteries",
            "Opened: https://example.org/battery-study",
            "Full source text from the study.",
        ]
        with (
            patch.object(research, "_search_web", return_value=results),
            patch("actions.browser_control.browser_control", side_effect=browser_results) as browser,
        ):
            findings, sources = research._research_query(
                object(),
                "batteries",
                "Which battery is best?",
                visible=True,
                opened_urls=set(),
            )

        actions = [call.args[0]["action"] for call in browser.call_args_list]
        self.assertEqual(actions, ["search", "new_tab", "get_text"])
        self.assertIn("Full source text from the study", findings)
        self.assertEqual(sources[0]["url"], "https://example.org/battery-study")

    def test_search_uses_second_http_engine_when_first_is_unavailable(self):
        bing_results = [{
            "title": "Independent benchmark",
            "snippet": "Comparison results.",
            "url": "https://benchmark.example/results",
        }]
        with (
            patch.object(research, "_duckduckgo_html_search", side_effect=RuntimeError("blocked")),
            patch.object(research, "_bing_html_search", return_value=bing_results) as bing,
        ):
            results = research._search_web("AI model benchmarks", max_results=3)

        bing.assert_called_once()
        self.assertEqual(results, bing_results)

    def test_quota_failure_uses_source_first_report_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "quota-fallback.md"
            with (
                patch("google.genai.Client", return_value=object()),
                patch.object(research, "_api_key", return_value="test-key"),
                patch.object(research, "_plan_queries", return_value=["battery evidence"]),
                patch.object(research, "_research_query", return_value=(
                    "Battery study evidence.",
                    [{"title": "Study", "url": "https://example.org/study"}],
                )),
                patch.object(research, "_synthesize", side_effect=RuntimeError("429 RESOURCE_EXHAUSTED")),
            ):
                result = research.build_deep_research({
                    "question": "Which battery is best?",
                    "output_path": str(output),
                })

            self.assertFalse(output.exists())
            self.assertIn("source-first fallback", result.warnings[0])
            self.assertIn("JARVIS reviewed 1 distinct web sources", result.report_markdown)

    def test_owner_policy_blocks_save_and_read_actions_require_a_completed_in_memory_report(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'action': 'save'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_report_is_written_only_after_explicit_save_action(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'action': 'save', 'output_path': 'report.md'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_read_action_returns_the_complete_unsaved_report(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'action': 'read'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_initial_request_waits_for_display_choice(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'question': 'Battery recycling'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_visible_confirmation_starts_remembered_request(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'action': 'confirm', 'execution_mode': 'visible'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_visible_mode_cannot_skip_the_initial_choice(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'question': 'Battery recycling', 'execution_mode': 'visible'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_owner_policy_blocks_background_confirmation_does_not_open_live_view(self):
        # Display/run preferences are untrusted arguments, never owner consent.
        player, speak = Mock(), Mock()
        with patch("agent.task_queue.get_queue") as queue, patch.object(research, "build_deep_research") as build:
            result = research.request_deep_research({'action': 'confirm', 'execution_mode': 'background'}, player=player, speak=speak)
        self.assertTrue(result.startswith("denied:"), result)
        queue.assert_not_called()
        build.assert_not_called()
        speak.assert_not_called()
        self.assertEqual(player.mock_calls, [])

    def test_live_tool_contract_declares_deep_research_once(self):
        project_root = Path(__file__).resolve().parent.parent
        tools = declared_tools(project_root / "main.py")
        self.assertEqual(tools.count("deep_research"), 1)

    def test_planner_fallback_keeps_deep_research_on_dedicated_action(self):
        plan = planner._fallback_plan("Do deep research on solid-state batteries")
        self.assertEqual(plan["steps"][0]["tool"], "deep_research")
        self.assertEqual(plan["steps"][0]["parameters"]["depth"], "deep")

    def test_denied_research_runner_cannot_announce_completion(self):
        spoken = []
        queue = TaskQueue(max_concurrent=1)

        class Result:
            artifacts = ["report.md", "report.evidence.json"]
            warnings = []

            def __str__(self):
                return "Deep research complete. Reviewed 12 sources. Report: report.md"

        task_id = queue.submit_job(
            "Deep research: batteries",
            lambda cancel_flag, progress: Result(),
            speak=spoken.append,
            kind="research",
        )
        task = queue._tasks[task_id]
        task.status = task.status.RUNNING
        queue._active_count = 1
        queue._run_task(task)

        self.assertEqual(spoken, [])
        self.assertEqual(task.action_receipts[0]["authorization_decision"], "DENY")

    def test_cancelling_unadmitted_runner_does_not_invoke_its_callback(self):
        cancelled = []
        queue = TaskQueue(max_concurrent=1)
        task_id = queue.submit_job(
            "Deep research: batteries",
            lambda cancel_flag, progress: None,
            on_cancel=lambda: cancelled.append(True),
            kind="research",
        )

        self.assertTrue(queue.cancel(task_id))
        self.assertEqual(cancelled, [])
        self.assertEqual(queue.get_status(task_id)["phase"], "Cancelled")


if __name__ == "__main__":
    unittest.main()
