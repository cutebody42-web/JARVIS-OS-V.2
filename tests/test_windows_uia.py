from types import SimpleNamespace

import pytest

from core.windows_uia import WindowsUIAError, WindowsUIAObserver


class _Rect:
    left = 1
    top = 2
    right = 301
    bottom = 202


class _Control:
    def __init__(self, name="Save", control_type="Button"):
        self.element_info = SimpleNamespace(
            name=name,
            control_type=control_type,
            automation_id="save-button",
            class_name="Button",
            process_id=77,
        )

    def rectangle(self):
        return _Rect()

    def is_enabled(self):
        return True

    def is_visible(self):
        return True


class _Window(_Control):
    handle = 1234

    def __init__(self, title, controls):
        super().__init__(title, "Window")
        self._title = title
        self._controls = controls

    def window_text(self):
        return self._title

    def descendants(self):
        return list(self._controls)


class _Desktop:
    def __init__(self, windows):
        self._windows = windows

    def windows(self):
        return list(self._windows)


def test_observer_returns_bounded_semantic_snapshots():
    desktop = _Desktop([_Window("Editor", [_Control(), _Control("Cancel")])])
    observer = WindowsUIAObserver(
        platform_name="Windows",
        desktop_factory=lambda **kwargs: desktop,
    )

    result = observer.snapshot_windows(max_controls_per_window=1)

    assert observer.available is True
    assert len(result) == 1
    assert result[0].title == "Editor"
    assert result[0].handle == 1234
    assert result[0].truncated is True
    assert result[0].controls[0].name == "Save"
    assert result[0].controls[0].control_type == "Button"
    assert result[0].controls[0].rectangle.right == 301


def test_observer_title_filter_is_case_insensitive():
    desktop = _Desktop([_Window("Calculator", []), _Window("Notes", [])])
    observer = WindowsUIAObserver(
        platform_name="Windows",
        desktop_factory=lambda **kwargs: desktop,
    )
    result = observer.snapshot_windows(title_contains="calc")
    assert [item.title for item in result] == ["Calculator"]


def test_non_windows_observer_fails_closed():
    observer = WindowsUIAObserver(platform_name="Linux", desktop_factory=lambda **kwargs: _Desktop([]))
    assert observer.available is False
    with pytest.raises(WindowsUIAError):
        observer.snapshot_windows()


def test_observer_rejects_unbounded_limits():
    observer = WindowsUIAObserver(platform_name="Windows", desktop_factory=lambda **kwargs: _Desktop([]))
    with pytest.raises(ValueError):
        observer.snapshot_windows(max_windows=101)
    with pytest.raises(ValueError):
        observer.snapshot_windows(max_controls_per_window=1001)
