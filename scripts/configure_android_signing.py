#!/usr/bin/env python3
"""Configure the generated Tauri Android app for release signing.

This script never reads credentials. It only wires Gradle to the private
``keystore.properties`` file that CI creates after ``tauri android init``.
The generated project is disposable, so the patch is deterministic and kept
out of the tracked source tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "// JARVIS_ANDROID_RELEASE_SIGNING"
_IMPORTS = ("import java.util.Properties", "import java.io.FileInputStream")


class AndroidSigningConfigError(RuntimeError):
    """Raised when the generated Gradle layout is not safe to patch."""


def _matching_brace(text: str, opening: int) -> int:
    if opening < 0 or opening >= len(text) or text[opening] != "{":
        raise AndroidSigningConfigError("Expected a Gradle block opening brace.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AndroidSigningConfigError("Generated Gradle file has unbalanced braces.")


def _block(text: str, needle: str, *, start: int = 0, end: int | None = None) -> tuple[int, int, int]:
    limit = len(text) if end is None else end
    position = text.find(needle, start, limit)
    if position < 0:
        raise AndroidSigningConfigError(f"Generated Gradle file is missing {needle.strip()} block.")
    opening = text.find("{", position, limit)
    if opening < 0:
        raise AndroidSigningConfigError(f"Generated Gradle {needle.strip()} block is malformed.")
    closing = _matching_brace(text, opening)
    if closing > limit:
        raise AndroidSigningConfigError(f"Generated Gradle {needle.strip()} block escapes its parent block.")
    return position, opening, closing


def configure_gradle(text: str) -> str:
    """Return an idempotently patched Tauri app ``build.gradle.kts``."""
    if not isinstance(text, str) or not text.strip():
        raise AndroidSigningConfigError("Generated Gradle file is empty.")
    if MARKER in text:
        required = (
            'create("release")',
            'rootProject.file("keystore.properties")',
            'signingConfig = signingConfigs.getByName("release")',
        )
        if not all(item in text for item in required):
            raise AndroidSigningConfigError("Existing JARVIS signing block is incomplete.")
        return text

    android_start, android_open, android_close = _block(text, "android {")
    build_start, _, _ = _block(text, "buildTypes {", start=android_open + 1, end=android_close)

    signing_block = f'''{MARKER}\n    signingConfigs {{\n        create("release") {{\n            val keystorePropertiesFile = rootProject.file("keystore.properties")\n            val keystoreProperties = Properties()\n            if (!keystorePropertiesFile.isFile) {{\n                throw GradleException("JARVIS release signing requires keystore.properties")\n            }}\n            keystoreProperties.load(FileInputStream(keystorePropertiesFile))\n\n            keyAlias = keystoreProperties.getProperty("keyAlias")\n            keyPassword = keystoreProperties.getProperty("keyPassword")\n            storeFile = file(keystoreProperties.getProperty("storeFile"))\n            storePassword = keystoreProperties.getProperty("storePassword")\n        }}\n    }}\n\n    '''
    text = text[:build_start] + signing_block + text[build_start:]

    # Recompute positions after insertion and scope the release lookup strictly
    # to buildTypes so a signingConfigs release block cannot be mistaken for it.
    _, android_open, android_close = _block(text, "android {")
    _, build_open, build_close = _block(text, "buildTypes {", start=android_open + 1, end=android_close)
    release_start, release_open, release_close = _block(
        text,
        'getByName("release")',
        start=build_open + 1,
        end=build_close,
    )
    release_body = text[release_open + 1 : release_close]
    if "signingConfig" in release_body:
        raise AndroidSigningConfigError("Generated release build already defines a signingConfig.")
    line_start = text.rfind("\n", 0, release_start) + 1
    indent = text[line_start:release_start]
    insertion = f'\n{indent}    signingConfig = signingConfigs.getByName("release")'
    text = text[: release_open + 1] + insertion + text[release_open + 1 :]

    missing_imports = [item for item in _IMPORTS if item not in text]
    if missing_imports:
        text = "\n".join(missing_imports) + "\n" + text
    return text


def configure_file(path: str | Path) -> None:
    gradle = Path(path)
    if not gradle.is_file():
        raise AndroidSigningConfigError("Generated Android app build.gradle.kts was not found.")
    original = gradle.read_text(encoding="utf-8")
    configured = configure_gradle(original)
    gradle.write_text(configured, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gradle_file", help="Path to generated app/build.gradle.kts")
    args = parser.parse_args()
    try:
        configure_file(args.gradle_file)
    except AndroidSigningConfigError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
