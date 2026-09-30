"""Optional desktop import. Absence is an environment failure, never a pass."""
import os
import sys


def load_pyautogui():
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise ImportError("Desktop input requires a real display; unavailable in headless cloud.")
    import pyautogui
    return pyautogui
