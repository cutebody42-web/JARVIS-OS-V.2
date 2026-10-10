"""Contracts for passing the desktop UI secret to the local Brain."""

import ast
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import brain_sidecar


ROOT = Path(__file__).resolve().parents[1]
BRAIN_RUST = ROOT / "web" / "src-tauri" / "src" / "brain.rs"
BRAIN_SPEC = ROOT / "packaging" / "windows" / "JARVIS-Brain.spec"


class SidecarSecretBootstrapTests(unittest.TestCase):
    def test_reader_accepts_an_exact_lowercase_hex_frame(self):
        token = "a1" * 32
        self.assertEqual(brain_sidecar._read_ui_token(BytesIO((token + "\n").encode())), token)

        invalid = (
            b"",
            b"a" * 63 + b"\n",
            b"a" * 65 + b"\n",
            b"A" * 64 + b"\n",
            b"g" * 64 + b"\n",
            b"a" * 64,
            b"a" * 64 + b"\r\n",
            b"a" * 64 + b"\nX",
            b"a" * 64 + b"\n" + b"b" * 64 + b"\n",
        )
        for frame in invalid:
            with self.subTest(frame=frame[:8]), self.assertRaisesRegex(ValueError, "invalid UI token"):
                brain_sidecar._read_ui_token(BytesIO(frame))

    def test_normal_parser_has_no_secret_bearing_argument(self):
        options = {
            option
            for action in brain_sidecar._parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--ui-token-stdin", options)
        self.assertNotIn("--ui-token", options)

    def test_packaged_brain_preserves_inherited_authentication_streams(self):
        spec = ast.parse(BRAIN_SPEC.read_text(encoding="utf-8"))
        executables = [
            node for node in ast.walk(spec)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "EXE"
        ]
        self.assertEqual(len(executables), 1)
        options = {keyword.arg: keyword.value for keyword in executables[0].keywords}
        self.assertIs(ast.literal_eval(options["console"]), True,
                      "Windowed PyInstaller builds discard the private stdin pipe")

    def test_missing_bootstrap_pipe_fails_before_creating_a_host(self):
        with patch.object(brain_sidecar.sys, "stdin", None), \
                patch.object(brain_sidecar, "LocalBrainHost") as host, \
                patch.object(brain_sidecar, "resolve_companion_gateway") as gateway:
            with self.assertRaisesRegex(ValueError, "private UI token bootstrap pipe"):
                brain_sidecar.main(["--port", "8765", "--ui-token-stdin", "--parent-pid", "42"])
        host.assert_not_called()
        gateway.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows hidden-process pipe integration")
    def test_hidden_windows_console_process_retains_token_pipe(self):
        # Match the pinned Tauri shell launcher: console-subsystem executable,
        # redirected streams, CREATE_NO_WINDOW. The pipe survives with no window.
        script = (
            "import ctypes, json, sys; from brain_sidecar import _read_ui_token; "
            "token = _read_ui_token(sys.stdin.buffer); "
            "print(json.dumps({'token': token, "
            "'console_window': bool(ctypes.windll.kernel32.GetConsoleWindow())}))"
        )
        token = "cd" * 32
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=ROOT,
            input=(token + "\n").encode("ascii"), capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=15, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(json.loads(result.stdout), {"token": token, "console_window": False})

    def test_tauri_launch_contract_keeps_token_out_of_argv_and_environment(self):
        source = BRAIN_RUST.read_text(encoding="utf-8")
        self.assertNotIn('"--ui-token",', source)
        self.assertNotIn(".env(\"JARVIS_UI_TOKEN\"", source)
        self.assertIn('"--ui-token-stdin"', source)
        self.assertIn("child.write", source)
        self.assertIn("drop(child);", source)
        self.assertNotIn("guard.child = Some(child)", source)

    def test_reader_does_not_accept_the_frame_until_the_parent_closes_stdin(self):
        token = "ab" * 32
        script = (
            "import sys; from brain_sidecar import _read_ui_token; "
            "print(_read_ui_token(sys.stdin.buffer), flush=True)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        try:
            process.stdin.write((token + "\n").encode("ascii"))
            process.stdin.flush()
            time.sleep(0.1)
            self.assertIsNone(process.poll(), "sidecar accepted its token before stdin EOF")
            process.stdin.close()
            self.assertEqual(process.wait(timeout=5), 0, process.stderr.read().decode("utf-8", "replace"))
            self.assertEqual(process.stdout.read().decode("ascii").strip(), token)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
