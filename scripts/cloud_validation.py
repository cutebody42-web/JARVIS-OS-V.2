#!/usr/bin/env python3
"""Cloud checks with machine-readable results; no hardware result is synthesized.

Run from the checkout under test. `full` is the unchanged discovery surface;
`core` is the required cloud application/security suite. A non-green run exits 1.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import unittest

CORE_MODULES = (
    "test_nexus_contracts", "test_nexus_text_route", "test_owner_kernel",
    "test_action_gateway_security", "test_owner_api", "test_phase1_decoupling",
    "test_hosted_api", "test_core_resilience", "test_qa_system",
    "test_api_key_validation", "test_graphics_capability",
    "test_deep_research", "test_presentation_maker",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    with tempfile.TemporaryDirectory(prefix="nexus-cloud-tests-") as directory:
        os.environ.setdefault("DATABASE_URL", "sqlite:///" + str(Path(directory) / "test.db"))
        os.environ.setdefault("JARVIS_ENV", "test")
        os.environ.setdefault("JWT_SECRET", "cloud-test-only-jwt-secret-with-sufficient-length")
        os.environ.setdefault("JARVIS_ENCRYPTION_KEY", "cloud-test-only-encryption-secret-with-sufficient-length")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        loader = unittest.TestLoader()
        suite = (loader.discover("tests") if args.suite == "full" else
                 loader.loadTestsFromNames(["tests." + name for name in CORE_MODULES]))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        from core.qa_report import redact
        errors = [{"test": test.id(), "traceback": redact(trace)} for test, trace in result.errors]
        failures = [{"test": test.id(), "traceback": redact(trace)} for test, trace in result.failures]
        environment = [item["test"] for item in errors if any(signature in item["traceback"] for signature in (
            "PortAudio library not found", "KeyError: 'DISPLAY'", "libEGL.so.1: cannot open",
            "libGL.so.1: cannot open", "could not connect to display",
        ))]
        payload = {
            "suite": args.suite, "measured_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "tests_run": result.testsRun, "passed": result.testsRun - len(errors) - len(failures) - len(result.skipped),
            "failures": failures, "errors": errors, "skipped": [(t.id(), reason) for t, reason in result.skipped],
            "environment_dependent_errors": environment, "successful": result.wasSuccessful(),
            "hardware_certified": False,
            "limits": "Offscreen/mocked tests verify code only. No microphone, camera, desktop, Windows, iGPU or laptop measurements.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
