#!/usr/bin/env python3
"""Non-interactive, side-effect-safe probes for the JARVIS live checklist."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actions import browser_control, file_controller
from core.qa_mode import guard_tool_call


def _case(name: str, status: str, notes: str, evidence: dict | None = None) -> dict:
    return {
        "name": name,
        "status": status,
        "notes": notes,
        "evidence": evidence or {},
    }


def _run_tests(*tests: str) -> tuple[bool, str]:
    command = [sys.executable, "-m", "unittest", *tests, "-q"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "JARVIS_QA_MODE": "1"},
    )
    details = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode == 0, details[-1200:]


def _ui_probe(output: Path) -> tuple[bool, dict, str]:
    result = subprocess.run(
        [sys.executable, "scripts/qa_ui_probe.py", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "JARVIS_QA_MODE": "1"},
    )
    probe_path = output / "ui-probe.json"
    payload = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
    return result.returncode == 0 and bool(payload), payload, (result.stdout + result.stderr)[-1200:]


def _audio_probe() -> tuple[bool, dict, str]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        inputs = [device.get("name", "") for device in devices if int(device.get("max_input_channels", 0)) > 0]
        outputs = [device.get("name", "") for device in devices if int(device.get("max_output_channels", 0)) > 0]
        evidence = {
            "input_devices": inputs,
            "output_devices": outputs,
            "default_device": list(sd.default.device),
        }
        return bool(inputs and outputs), evidence, "Audio hardware enumerated without opening a recording stream."
    except Exception as exc:
        return False, {}, f"Audio enumeration failed: {exc}"


def _display_probe() -> tuple[bool | None, dict, str]:
    if platform.system() != "Darwin":
        return None, {"platform": platform.system()}, "DEFERRED: no physical display certification on this platform."
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    resolutions = [line.strip() for line in result.stdout.splitlines() if "Resolution:" in line]
    evidence = {"detected_displays": len(resolutions), "resolutions": resolutions}
    if result.returncode != 0 or not resolutions:
        return None, evidence, "Physical display inventory was unavailable in the headless QA environment."
    return True, evidence, "Physical display inventory read successfully."


def _screen_permission_probe() -> tuple[bool | None, dict, str]:
    if platform.system() != "Darwin":
        return None, {}, "Screen Recording permission probe is only implemented for macOS."
    try:
        core_graphics = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        probe = core_graphics.CGPreflightScreenCaptureAccess
        probe.restype = ctypes.c_bool
        allowed = bool(probe())
        return allowed, {"screen_recording_permission": allowed}, (
            "Screen Recording permission is granted."
            if allowed else
            "Screen Recording permission is not currently granted; no permission prompt was triggered."
        )
    except Exception as exc:
        return None, {}, f"Could not query Screen Recording permission: {exc}"


class _LocalHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html><title>JARVIS QA</title><button id='qa'>Ready</button></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


async def _playwright_local(url: str) -> dict:
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        title = await page.title()
        button = await page.locator("#qa").inner_text()
        await browser.close()
        return {"title": title, "button": button}


def _browser_probe() -> tuple[bool | None, dict, str]:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _LocalHandler)
    except OSError as exc:
        return None, {"local_server_error": str(exc)}, "The environment blocked creation of a localhost QA server."
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            local_ok = response.status == 200
        evidence = {"url": browser_control._normalize_url(url), "local_http": local_ok}
        try:
            evidence.update(asyncio.run(_playwright_local(url)))
            return evidence.get("title") == "JARVIS QA", evidence, "Headless Chromium navigated only to the local QA page."
        except Exception as exc:
            evidence["playwright_error"] = str(exc)[:300]
            unavailable = isinstance(exc, ModuleNotFoundError) or any(marker in str(exc) for marker in (
                "Executable doesn't exist", "Connection closed while reading from the driver",
                "Host system is missing dependencies", "error while loading shared libraries",
            ))
            return (None if unavailable else False), evidence, (
                "Local page worked, but the Playwright runtime is unavailable."
                if unavailable else "Local browser integration failed; see retained evidence."
            )
    finally:
        server.shutdown()
        server.server_close()


def _file_probe(workspace: Path) -> tuple[bool, dict, str]:
    """Exercise only admitted operations, through exact owner consent and receipts."""
    from core.action_gateway import OwnerRuntime
    from core.action_contracts import ActionStatus as S
    from core.authority_contracts import PolicyContext

    root = (workspace / "file-workflow").resolve()
    runtime = OwnerRuntime(PolicyContext("qa-owner", "qa-session", root))
    try:
        def approved(arguments):
            proposal = runtime.gateway.run_tool("file_controller", arguments, runtime=runtime)
            if proposal.result.status is not S.REQUIRE_CONFIRMATION:
                return proposal
            ticket = runtime.owner.approve(proposal.request_id, proposal.normalized_argument_digest)
            return runtime.execute_approved(proposal.request_id, ticket.ticket_id)

        target = root / "qa.txt"
        created = approved({"action": "write", "path": "qa.txt", "content": "JARVIS QA"})
        read = approved({"action": "read", "path": "qa.txt"})
        outside_path = workspace.resolve() / "escape.txt"
        outside = runtime.gateway.run_tool("file_controller", {
            "action": "write", "path": "../escape.txt", "content": "blocked",
        }, runtime=runtime)
        rename = runtime.gateway.run_tool("file_controller", {
            "action": "rename", "path": "qa.txt", "new_name": "moved.txt",
        }, runtime=runtime)
        passed = (created.result.status is S.SUCCEEDED and read.result.status is S.SUCCEEDED
                  and target.read_text() == "JARVIS QA"
                  and outside.result.status is S.DENIED and rename.result.status is S.DENIED
                  and not outside_path.exists() and not (root / "moved.txt").exists())
        return passed, {
            "created": created.to_dict(), "read": read.to_dict(),
            "outside_write": outside.to_dict(), "unsupported_rename": rename.to_dict(),
        }, "Verified consented workspace create/read through the Owner Kernel. Outside writes and unsupported rename denied."
    finally:
        runtime.gateway.close()


def _safety_probe(workspace: Path) -> tuple[bool, dict, str]:
    decisions = {
        "message_send": guard_tool_call("send_message", {}).allowed,
        "shutdown": guard_tool_call("computer_settings", {"action": "shutdown"}).allowed,
        "game_install": guard_tool_call("game_updater", {"action": "install"}).allowed,
        "outside_file": guard_tool_call("file_controller", {"action": "write", "path": str(workspace.parent / "outside.txt")}).allowed,
    }
    passed = not any(decisions.values())
    return passed, decisions, "All irreversible and out-of-workspace probes were rejected before execution."


def _stability_probe(seconds: float) -> tuple[bool, dict, str]:
    import psutil

    process = psutil.Process(os.getpid())
    start = process.memory_info().rss
    start_threads = process.num_threads()
    samples = []
    deadline = time.monotonic() + max(1.0, seconds)
    process.cpu_percent(None)
    while time.monotonic() < deadline:
        samples.append({
            "rss_mb": process.memory_info().rss / (1024 * 1024),
            "threads": process.num_threads(),
            "cpu_percent": process.cpu_percent(None),
        })
        time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
    growth_mb = (process.memory_info().rss - start) / (1024 * 1024)
    thread_growth = process.num_threads() - start_threads
    passed = growth_mb < 50 and thread_growth < 10
    return passed, {
        "duration_seconds": seconds,
        "samples": samples,
        "rss_growth_mb": growth_mb,
        "thread_growth": thread_growth,
    }, "Short harness stability probe completed; the 30-minute real JARVIS soak remains DEFERRED and is required for certification."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--stability-seconds", type=float, default=5.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Owner policy intentionally forbids source-tree mutations. Evidence lives
    # with the report; executable probes use a fresh workspace outside the repo.
    with tempfile.TemporaryDirectory(prefix="nexus-qa-workspace-") as directory:
        return _run_checklist(args, Path(directory))


def _run_checklist(args, workspace: Path) -> int:
    os.environ["JARVIS_QA_MODE"] = "1"
    os.environ["JARVIS_QA_WORKSPACE"] = str(workspace)

    cases = []
    startup_ok, startup_details = _run_tests(
        "tests.test_ui_regressions.UIRegressionTests.test_startup_blocks_text_and_file_callbacks_even_when_setup_is_hidden",
        "tests.test_ui_regressions.UIRegressionTests.test_operational_ready_requires_key_gate_and_dismissed_setup",
    )
    cases.append(_case("Clean startup and API-only gate", "passed" if startup_ok else "failed", "Startup gating and setup isolation were exercised offscreen.", {"test_output": startup_details}))

    key_ok, key_details = _run_tests(
        "tests.test_ui_regressions.UIRegressionTests.test_failed_validation_is_recoverable_and_never_emits_done",
        "tests.test_ui_regressions.UIRegressionTests.test_successful_setup_keeps_key_in_memory_and_preserves_config",
    )
    cases.append(_case("Invalid then valid Gemini key", "passed" if key_ok else "failed", "Validation failure and recovery were tested with a mocked Gemini boundary; no paid API call was made.", {"test_output": key_details}))

    voice_ok, voice_details = _run_tests(
        "tests.test_ui_regressions.UIRegressionTests.test_saved_voice_survives_restart_without_persisting_keys",
        "tests.test_ui_regressions.UIRegressionTests.test_voice_selection_notifies_engine_once",
    )
    cases.append(_case("Selected voice and introduction", "partial" if voice_ok else "failed", "Voice selection/persistence tested with a fake engine boundary. The proposed cinematic introduction is not implemented; real playback is DEFERRED.", {"test_output": voice_details}))

    subtitle_ok, subtitle_details = _run_tests(
        "tests.test_ui_regressions.UIRegressionTests.test_subtitle_chunks_remain_visible_until_turn_completion",
        "tests.test_ui_regressions.UIRegressionTests.test_muted_facade_never_displays_speech_subtitles",
    )
    cases.append(_case("Subtitle and audio synchronization", "partial" if subtitle_ok else "failed", "Subtitle turn-completion and mute behavior tested offscreen. Actual audio synchronization is DEFERRED.", {"test_output": subtitle_details}))

    cases.append(_case("Microphone turn-taking and interruption", "deferred", "DEFERRED: cloud runner has no authorized target microphone/speaker; no stream opened."))

    reconnect_ok, reconnect_details = _run_tests(
        "tests.test_core_resilience.CoreResilienceTests.test_live_model_falls_back_when_listing_fails",
        "tests.test_qa_system.QAModeSafetyTests.test_live_reconnect_backoff_is_bounded",
    )
    cases.append(_case("Gemini disconnect and reconnect", "partial" if reconnect_ok else "failed", "Offline/fallback behavior passed with fault injection; a real network interruption was not performed.", {"test_output": reconnect_details}))

    ui_ok, ui_evidence, ui_notes = _ui_probe(args.output / "ui")
    display_evidence, display_notes = {"hardware_certified": False}, "Physical displays: DEFERRED."
    display_status = "partial" if ui_ok else "failed"
    cases.append(_case("Window and secondary display", display_status, f"{display_notes} Minimum and standard sizes rendered. Physical window movement remains manual.", {"ui": ui_evidence, "display": display_evidence, "probe_log": ui_notes}))

    theme_ok, theme_details = _run_tests(
        "tests.test_ui_regressions.UIRegressionTests.test_theme_change_updates_existing_widget_styles",
        "tests.test_ui_regressions.UIRegressionTests.test_settings_existing_graphics_choices_emit_owner_selection",
        "tests.test_ui_regressions.UIRegressionTests.test_graphics_choice_updates_renderer_cadence_and_persistence",
    )
    cases.append(_case("Themes, graphics, settings, compact mode", "partial" if theme_ok else "failed", "Themes and graphics passed automatically; physical dock detachment and compact-mode feel remain manual.", {"test_output": theme_details}))

    missing_names = sum(len(item.get("missing_accessible_names", [])) for item in ui_evidence.values()) if ui_evidence else 0
    cases.append(_case("Keyboard and focus accessibility", "partial" if ui_ok else "failed", f"Rendered focusable surfaces were inspected. {missing_names} custom controls lack an accessible name; full VoiceOver navigation remains manual.", {"missing_accessible_names": missing_names}))

    browser_ok, browser_evidence, browser_notes = _browser_probe()
    browser_status = "passed" if browser_ok is True else "blocked" if browser_ok is None else "failed"
    if browser_ok is None and os.environ.get("NEXUS_REQUIRE_BROWSER_QA") == "1":
        browser_status = "failed"
    cases.append(_case("Controlled browser page", browser_status, browser_notes, browser_evidence))

    file_ok, file_evidence, file_notes = _file_probe(workspace)
    cases.append(_case("Temporary file workflow", "passed" if file_ok else "failed", file_notes, file_evidence))

    safety_ok, safety_evidence, safety_notes = _safety_probe(workspace)
    cases.append(_case("Messaging draft safety", "partial" if safety_ok else "failed", safety_notes + " Draft typing and cancellation remain manual because they affect a visible composer.", safety_evidence))
    cases.append(_case("Read-only message awareness", "deferred", "Reading a real Messages or Instagram conversation requires user-selected content and macOS permissions; automation will not inspect private conversations.", {}))
    cases.append(_case("Reminder lifecycle", "deferred", "Creating a real launchd reminder changes macOS state, so this remains supervised even in QA mode.", {}))

    cases.append(_case("Screen and permission handling", "deferred", "DEFERRED: real screen/camera permission and capture require the authorized physical device. No capture attempted."))

    contract_ok, contract_details = _run_tests(
        "tests.test_qa_system.ToolContractTests.test_declared_tool_inventory_is_complete_and_unique",
        "tests.test_core_resilience.PureActionContractTests.test_weather_egress_requires_consent_without_opening_a_browser",
        "tests.test_core_resilience.PureActionContractTests.test_flight_url_contains_route_and_passengers",
    )
    cases.append(_case("Read-only information tools", "partial" if contract_ok else "failed", "Tool contracts and failure behavior passed without opening external sites. Live external-service results remain DEFERRED.", {"test_output": contract_details}))
    cases.append(_case("Reversible desktop actions", "deferred", "The safety guard passed, but automated keyboard or mouse input is intentionally not injected into the operator desktop.", safety_evidence))

    stability_ok, stability_evidence, stability_notes = _stability_probe(args.stability_seconds)
    cases.append(_case("Thirty-minute soak", "partial" if stability_ok else "failed", stability_notes, stability_evidence))

    output = args.output / "checklist-results.json"
    output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(output)
    return 1 if any(case["status"] == "failed" for case in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
