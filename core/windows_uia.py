"""Structured Windows UI Automation perception for NEXUS.

JARVIS should prefer semantic UI structure over blind coordinates. This module
uses the BSD-licensed pywinauto UIA backend to *observe* windows and controls.
It intentionally exposes no click, type, invoke, or mutation primitive; action
execution remains behind the separate Owner Kernel policy boundary.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
import platform
import re
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
    sensitive: bool = False


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
    return "".join(
        character if ord(character) >= 32 and ord(character) != 127 else " "
        for character in value[:limit]
    )


def _safe_call(obj: Any, name: str, default=None):
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:
        return default


def _password_state(wrapper: Any, info: Any) -> bool:
    """Read the UIA password property without requesting the control value.

    pywinauto's UIAWrapper does not expose an ``is_password()`` method.  The
    authoritative flag lives on the wrapped IUIAutomationElement as
    ``CurrentIsPassword``.  Keep a wrapper-method fallback for compatible test
    and alternate providers, but never mistake an unavailable property for a
    non-sensitive control.
    """
    try:
        element = getattr(info, "element", None)
    except Exception:
        element = None
    value = _safe_call(element, "CurrentIsPassword")
    if value is None:
        value = _safe_call(wrapper, "is_password")
    # An inaccessible COM property must not expose a potentially secret name.
    return not isinstance(value, (bool, int)) or value != 0


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
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


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
        password = _password_state(wrapper, info)
        control_type = _clean_text(getattr(info, "control_type", ""), 120)
        automation_id = _clean_text(getattr(info, "automation_id", ""), 300)
        class_name = _clean_text(getattr(info, "class_name", ""), 300)
        if password:
            name = "[redacted sensitive control]"
        else:
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
            sensitive=password,
        )

    @staticmethod
    def _bounded_descendants(window: Any, limit: int) -> tuple[tuple[Any, ...], bool]:
        """Breadth-first UIA walk that never retains more than limit + 1 nodes."""
        queue: deque[Any] = deque()
        output: list[Any] = []
        truncated = False

        def extend_children(node: Any, budget: int) -> None:
            nonlocal truncated
            children = getattr(node, "children", None)
            if not callable(children):
                raise WindowsUIAError("UIA control does not expose bounded child enumeration.")
            try:
                iterator = iter(children())
                batch = list(islice(iterator, max(0, budget) + 1))
            except WindowsUIAError:
                raise
            except Exception as exc:
                raise WindowsUIAError(
                    f"UIA child enumeration failed ({type(exc).__name__})."
                ) from None
            if len(batch) > budget:
                truncated = True
                batch = batch[:budget]
            queue.extend(batch)

        extend_children(window, limit + 1)
        while queue and len(output) < limit:
            item = queue.popleft()
            output.append(item)
            remaining = limit + 1 - len(output) - len(queue)
            if remaining > 0:
                extend_children(item, remaining)
        if queue:
            truncated = True
        return tuple(output), truncated

    def snapshot_windows(
        self,
        *,
        title_contains: str | None = None,
        process_id: int | None = None,
        allow_all_windows: bool = False,
        include_hidden: bool = False,
        max_windows: int = 40,
        max_controls_per_window: int = 300,
    ) -> tuple[UIAWindowSnapshot, ...]:
        if self._platform != "Windows":
            raise WindowsUIAError("Windows UIA observation is available on Windows only.")
        if isinstance(max_windows, bool) or not isinstance(max_windows, int) or not 1 <= max_windows <= 100:
            raise ValueError("max_windows must be between 1 and 100.")
        if (
            isinstance(max_controls_per_window, bool)
            or not isinstance(max_controls_per_window, int)
            or not 1 <= max_controls_per_window <= 1000
        ):
            raise ValueError("max_controls_per_window must be between 1 and 1000.")
        if not isinstance(allow_all_windows, bool) or not isinstance(include_hidden, bool):
            raise ValueError("UIA scope flags must be boolean.")
        if process_id is not None and (
            isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0
        ):
            raise ValueError("process_id must be a positive integer.")
        needle = None
        if title_contains is not None:
            if (
                not isinstance(title_contains, str)
                or not title_contains.strip()
                or len(title_contains) > 300
            ):
                raise ValueError("Invalid title filter.")
            needle = title_contains.strip().casefold()
        if process_id is None and needle is None and not allow_all_windows:
            raise ValueError(
                "UIA observation requires an exact process_id, a non-empty title filter, "
                "or explicit allow_all_windows=True."
            )

        desktop = self._desktop()
        provider_filters: dict[str, Any] = {"visible_only": not include_hidden}
        if process_id is not None:
            provider_filters["process"] = process_id
        if needle is not None:
            provider_filters["title_re"] = "(?i).*" + re.escape(title_contains.strip()) + ".*"
        try:
            windows = iter(desktop.windows(**provider_filters))
        except Exception as exc:
            raise WindowsUIAError(f"UIA window enumeration failed ({type(exc).__name__}).") from None

        snapshots: list[UIAWindowSnapshot] = []
        scanned = 0
        scan_limit = min(400, max_windows * 4)
        while True:
            try:
                window = next(windows)
            except StopIteration:
                break
            except Exception as exc:
                raise WindowsUIAError(
                    f"UIA window enumeration failed ({type(exc).__name__})."
                ) from None
            scanned += 1
            if scanned > scan_limit:
                raise WindowsUIAError(
                    "UIA provider scan exceeds the explicit bound; narrow the scope."
                )
            title = _clean_text(_safe_call(window, "window_text", ""), 500)
            if needle is not None and needle not in title.casefold():
                continue
            window_pid = _pid(window)
            if process_id is not None and window_pid != process_id:
                continue
            if len(snapshots) >= max_windows:
                raise WindowsUIAError(
                    "UIA window result exceeds the explicit bound; narrow the scope."
                )
            descendants, truncated = self._bounded_descendants(
                window, max_controls_per_window
            )
            controls = tuple(self._element(item) for item in descendants)
            handle = getattr(window, "handle", None)
            snapshots.append(UIAWindowSnapshot(
                title=title,
                handle=handle if isinstance(handle, int) and not isinstance(handle, bool) else None,
                process_id=window_pid,
                rectangle=_rect(window),
                controls=controls,
                truncated=truncated,
            ))
        return tuple(snapshots)
