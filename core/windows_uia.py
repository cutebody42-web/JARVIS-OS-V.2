"""Structured Windows UI Automation perception for NEXUS.

JARVIS should prefer semantic UI structure over blind coordinates. This module
uses the BSD-licensed pywinauto UIA backend to *observe* windows and controls.
It intentionally exposes no click, type, invoke, or mutation primitive; action
execution remains behind the separate Owner Kernel policy boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any, Callable


class WindowsUIAError(RuntimeError):
    pass


@dataclass(frozen=True)
class UIARect:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class UIAElementSnapshot:
    name: str
    control_type: str
    automation_id: str
    class_name: str
    process_id: int | None
    enabled: bool | None
    visible: bool | None
    rectangle: UIARect | None


@dataclass(frozen=True)
class UIAWindowSnapshot:
    title: str
    handle: int | None
    process_id: int | None
    rectangle: UIARect | None
    controls: tuple[UIAElementSnapshot, ...]
    truncated: bool


def _clean_text(value: Any, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "")[:limit]


def _safe_call(obj: Any, name: str, default=None):
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:
        return default


def _rect(wrapper: Any) -> UIARect | None:
    value = _safe_call(wrapper, "rectangle")
    if value is None:
        return None
    try:
        return UIARect(int(value.left), int(value.top), int(value.right), int(value.bottom))
    except Exception:
        return None


def _pid(wrapper: Any) -> int | None:
    value = _safe_call(wrapper, "process_id")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        element_info = getattr(wrapper, "element_info", None)
        value = getattr(element_info, "process_id", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class WindowsUIAObserver:
    def __init__(
        self,
        *,
        desktop_factory: Callable[..., Any] | None = None,
        platform_name: str | None = None,
    ) -> None:
        self._desktop_factory = desktop_factory
        self._platform = platform_name or platform.system()

    @property
    def available(self) -> bool:
        if self._platform != "Windows":
            return False
        if self._desktop_factory is not None:
            return True
        try:
            import pywinauto  # noqa: F401
        except ImportError:
            return False
        return True

    def _desktop(self):
        if self._platform != "Windows":
            raise WindowsUIAError("Windows UIA observation is available on Windows only.")
        factory = self._desktop_factory
        if factory is None:
            try:
                from pywinauto import Desktop
            except ImportError as exc:
                raise WindowsUIAError("pywinauto is not installed.") from exc
            factory = Desktop
        try:
            return factory(backend="uia")
        except Exception as exc:
            raise WindowsUIAError(f"UIA backend failed to initialize ({type(exc).__name__}).") from None

    @staticmethod
    def _element(wrapper: Any) -> UIAElementSnapshot:
        info = getattr(wrapper, "element_info", None)
        control_type = _clean_text(getattr(info, "control_type", ""), 120)
        automation_id = _clean_text(getattr(info, "automation_id", ""), 300)
        class_name = _clean_text(getattr(info, "class_name", ""), 300)
        name = _clean_text(getattr(info, "name", ""), 500)
        if not name:
            name = _clean_text(_safe_call(wrapper, "window_text", ""), 500)
        enabled = _safe_call(wrapper, "is_enabled")
        visible = _safe_call(wrapper, "is_visible")
        return UIAElementSnapshot(
            name=name,
            control_type=control_type,
            automation_id=automation_id,
            class_name=class_name,
            process_id=_pid(wrapper),
            enabled=enabled if isinstance(enabled, bool) else None,
            visible=visible if isinstance(visible, bool) else None,
            rectangle=_rect(wrapper),
        )

    def snapshot_windows(
        self,
        *,
        title_contains: str | None = None,
        max_windows: int = 40,
        max_controls_per_window: int = 300,
    ) -> tuple[UIAWindowSnapshot, ...]:
        if isinstance(max_windows, bool) or not isinstance(max_windows, int) or not 1 <= max_windows <= 100:
            raise ValueError("max_windows must be between 1 and 100.")
        if (
            isinstance(max_controls_per_window, bool)
            or not isinstance(max_controls_per_window, int)
            or not 1 <= max_controls_per_window <= 1000
        ):
            raise ValueError("max_controls_per_window must be between 1 and 1000.")
        needle = None
        if title_contains is not None:
            if not isinstance(title_contains, str) or len(title_contains) > 300:
                raise ValueError("Invalid title filter.")
            needle = title_contains.casefold()

        desktop = self._desktop()
        try:
            windows = list(desktop.windows())[:max_windows]
        except Exception as exc:
            raise WindowsUIAError(f"UIA window enumeration failed ({type(exc).__name__}).") from None

        snapshots: list[UIAWindowSnapshot] = []
        for window in windows:
            title = _clean_text(_safe_call(window, "window_text", ""), 500)
            if needle is not None and needle not in title.casefold():
                continue
            try:
                descendants = list(window.descendants())
            except Exception:
                descendants = []
            controls = tuple(self._element(item) for item in descendants[:max_controls_per_window])
            handle = getattr(window, "handle", None)
            snapshots.append(UIAWindowSnapshot(
                title=title,
                handle=handle if isinstance(handle, int) and not isinstance(handle, bool) else None,
                process_id=_pid(window),
                rectangle=_rect(window),
                controls=controls,
                truncated=len(descendants) > max_controls_per_window,
            ))
        return tuple(snapshots)
