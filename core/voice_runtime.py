"""Local desktop voice runtime for the installed JARVIS shell.

Windows uses the OS-owned System.Speech stack for one-shot dictation and
speech synthesis. No microphone audio or transcript is sent to a third-party
service by this module. The runtime is intentionally push-to-talk; an always-on
wake-word loop remains opt-in future work because it would keep the microphone
open continuously.

The process handle is tracked so the UI can cancel listening/speaking without
killing the JARVIS Brain process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import platform
import subprocess
import tempfile
import threading
from typing import Callable


class VoiceRuntimeError(RuntimeError):
    pass


class VoiceState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True)
class VoiceResult:
    text: str
    confidence: float | None
    engine: str


_WINDOWS_LISTEN_SCRIPT = r"""
param([string]$Language, [double]$TimeoutSeconds)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$culture = [System.Globalization.CultureInfo]::GetCultureInfo($Language)
$engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture)
try {
  $engine.SetInputToDefaultAudioDevice()
  $grammar = New-Object System.Speech.Recognition.DictationGrammar
  $engine.LoadGrammar($grammar)
  $result = $engine.Recognize([TimeSpan]::FromSeconds($TimeoutSeconds))
  if ($null -eq $result) {
    @{ text = ""; confidence = $null; engine = "windows-system-speech" } |
      ConvertTo-Json -Compress
  } else {
    @{
      text = [string]$result.Text
      confidence = [double]$result.Confidence
      engine = "windows-system-speech"
    } | ConvertTo-Json -Compress
  }
} finally {
  $engine.Dispose()
}
"""

_WINDOWS_SPEAK_SCRIPT = r"""
param([string]$Text)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voice.Rate = 0
  $voice.Volume = 100
  $voice.Speak($Text)
} finally {
  $voice.Dispose()
}
"""


class WindowsVoiceRuntime:
    def __init__(
        self,
        *,
        popen: Callable = subprocess.Popen,
        platform_name: str | None = None,
    ):
        self._popen = popen
        self._platform = platform_name or platform.system()
        self._guard = threading.RLock()
        self._process = None
        self._script_path: str | None = None
        self._state = VoiceState.IDLE
        self._last_error: str | None = None

    @property
    def available(self) -> bool:
        return self._platform == "Windows"

    @property
    def state(self) -> VoiceState:
        with self._guard:
            return self._state

    @property
    def last_error(self) -> str | None:
        with self._guard:
            return self._last_error

    def status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "engine": "windows-system-speech" if self.available else None,
            "state": self.state.value,
            "last_error": self.last_error,
            "privacy": "local_os_speech",
            "wake_word": False,
        }

    def _start(self, script: str, args: list[str], state: VoiceState):
        if not self.available:
            raise VoiceRuntimeError("Native JARVIS voice is currently available on Windows only.")
        with self._guard:
            if self._process is not None and self._process.poll() is None:
                raise VoiceRuntimeError("JARVIS voice is already active.")
            self._last_error = None
            self._state = state
            try:
                script_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".ps1",
                    prefix="jarvis-voice-",
                    delete=False,
                )
                try:
                    script_file.write(script)
                finally:
                    script_file.close()
                self._script_path = script_file.name
                process = self._popen(
                    [
                        "powershell.exe",
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        script_file.name,
                        *args,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                if self._script_path:
                    try:
                        import os
                        os.unlink(self._script_path)
                    except OSError:
                        pass
                    self._script_path = None
                self._state = VoiceState.ERROR
                self._last_error = type(exc).__name__
                raise VoiceRuntimeError("Windows speech runtime could not start.") from None
            self._process = process
            return process

    def _finish(self, process, *, timeout: float) -> tuple[str, str]:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            with self._guard:
                self._state = VoiceState.ERROR
                self._last_error = "TimeoutExpired"
                self._process = None
            raise VoiceRuntimeError("JARVIS voice operation timed out.") from None
        with self._guard:
            self._process = None
            script_path = self._script_path
            self._script_path = None
        if script_path:
            try:
                import os
                os.unlink(script_path)
            except OSError:
                pass
        if process.returncode != 0:
            with self._guard:
                self._state = VoiceState.ERROR
                self._last_error = "SystemSpeechError"
            raise VoiceRuntimeError("Windows speech runtime returned an error.")
        return stdout, stderr

    def listen_once(
        self,
        *,
        language: str = "en-US",
        timeout_seconds: float = 8.0,
    ) -> VoiceResult:
        if not isinstance(language, str) or not language.strip() or len(language) > 24:
            raise ValueError("language must be a short culture name")
        if not 1.0 <= float(timeout_seconds) <= 30.0:
            raise ValueError("timeout_seconds must be between 1 and 30")
        process = self._start(
            _WINDOWS_LISTEN_SCRIPT,
            ["-Language", language.strip(), "-TimeoutSeconds", str(float(timeout_seconds))],
            VoiceState.LISTENING,
        )
        stdout, _ = self._finish(process, timeout=float(timeout_seconds) + 12.0)
        line = next((item.strip() for item in reversed(stdout.splitlines()) if item.strip()), "")
        try:
            payload = json.loads(line) if line else {}
        except json.JSONDecodeError:
            payload = {}
        text = payload.get("text")
        confidence = payload.get("confidence")
        if not isinstance(text, str):
            text = ""
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence_value = None
        else:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        with self._guard:
            self._state = VoiceState.IDLE
        return VoiceResult(text.strip(), confidence_value, "windows-system-speech")

    def speak(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        clean = text.strip()
        if len(clean) > 12000:
            raise ValueError("text is too long for one speech request")
        process = self._start(
            _WINDOWS_SPEAK_SCRIPT,
            ["-Text", clean],
            VoiceState.SPEAKING,
        )
        self._finish(process, timeout=max(30.0, min(180.0, len(clean) / 12.0 + 20.0)))
        with self._guard:
            self._state = VoiceState.IDLE

    def stop(self) -> bool:
        with self._guard:
            process = self._process
            self._process = None
            self._state = VoiceState.IDLE
        if process is None or process.poll() is not None:
            return False
        try:
            process.terminate()
            process.wait(timeout=3)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        return True
