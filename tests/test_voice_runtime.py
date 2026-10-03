"""Contracts for the local Windows JARVIS voice runtime."""

import json
import unittest

from core.voice_runtime import VoiceRuntimeError, VoiceState, WindowsVoiceRuntime


class FakeProcess:
    def __init__(self, stdout="", stderr="", returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return None if not self.terminated and self.returncode is None else self.returncode

    def communicate(self, timeout=None):
        return self._stdout, self._stderr

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminated = True
        self.returncode = -9


class Factory:
    def __init__(self, process):
        self.process = process
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return self.process


class VoiceRuntimeTests(unittest.TestCase):
    def test_non_windows_fails_closed(self):
        runtime = WindowsVoiceRuntime(platform_name="Linux")
        self.assertFalse(runtime.available)
        with self.assertRaises(VoiceRuntimeError):
            runtime.listen_once()

    def test_listen_parses_local_system_speech_result(self):
        payload = {
            "text": "hello jarvis",
            "confidence": 0.91,
            "engine": "windows-system-speech",
        }
        process = FakeProcess(json.dumps(payload) + "\n", "", 0)
        factory = Factory(process)
        runtime = WindowsVoiceRuntime(popen=factory, platform_name="Windows")

        result = runtime.listen_once(language="en-US", timeout_seconds=4)

        self.assertEqual(result.text, "hello jarvis")
        self.assertAlmostEqual(result.confidence, 0.91)
        self.assertEqual(result.engine, "windows-system-speech")
        self.assertEqual(runtime.state, VoiceState.IDLE)
        self.assertEqual(factory.calls[0][0][0], "powershell.exe")

    def test_speak_uses_windows_system_speech(self):
        process = FakeProcess("", "", 0)
        factory = Factory(process)
        runtime = WindowsVoiceRuntime(popen=factory, platform_name="Windows")

        runtime.speak("At your service.")

        self.assertEqual(runtime.state, VoiceState.IDLE)
        args = factory.calls[0][0]
        self.assertIn("-Text", args)
        self.assertIn("At your service.", args)

    def test_stop_terminates_active_voice_process(self):
        process = FakeProcess("", "", None)
        factory = Factory(process)
        runtime = WindowsVoiceRuntime(popen=factory, platform_name="Windows")
        runtime._process = process
        runtime._state = VoiceState.LISTENING

        self.assertTrue(runtime.stop())
        self.assertTrue(process.terminated)
        self.assertEqual(runtime.state, VoiceState.IDLE)


if __name__ == "__main__":
    unittest.main()
