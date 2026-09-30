"""Static contract for the installed JARVIS voice UX."""

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "web" / "src" / "components" / "jarvis-product-shell.tsx"
RUNTIME = ROOT / "web" / "src" / "lib" / "jarvis-runtime.ts"


class ProductShellVoiceContractTests(unittest.TestCase):
    def test_shell_wires_full_local_voice_lifecycle(self):
        source = SHELL.read_text(encoding="utf-8")
        for marker in (
            'setVoiceState("listening")',
            'setVoiceState("processing")',
            'setVoiceState("speaking")',
            'setVoiceState("error"',
            'await client.voiceSpeak(reply.text)',
            'await client.voiceStop()',
            '"Interrupt JARVIS speech"',
            'aria-live="polite"',
        ):
            self.assertIn(marker, source)

    def test_runtime_routes_voice_to_authenticated_local_sidecar(self):
        source = RUNTIME.read_text(encoding="utf-8")
        for endpoint in ('"/v1/voice/listen"', '"/v1/voice/speak"', '"/v1/voice/stop"'):
            self.assertIn(endpoint, source)


if __name__ == "__main__":
    unittest.main()
