import unittest
from types import SimpleNamespace

from core.windows_uia import WindowsUIAError, WindowsUIAObserver


class _Rect:
    left = 1
    top = 2
    right = 301
    bottom = 202


class _Control:
    def __init__(self, name="Save", control_type="Button"):
        self.element_info = SimpleNamespace(
            name=name, control_type=control_type, automation_id="save-button",
            class_name="Button", process_id=77,
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


class WindowsUIATests(unittest.TestCase):
    def test_observer_returns_bounded_semantic_snapshots(self):
        desktop = _Desktop([_Window("Editor", [_Control(), _Control("Cancel")])])
        observer = WindowsUIAObserver(platform_name="Windows", desktop_factory=lambda **kwargs: desktop)
        result = observer.snapshot_windows(max_controls_per_window=1)
        self.assertTrue(observer.available)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "Editor")
        self.assertEqual(result[0].handle, 1234)
        self.assertTrue(result[0].truncated)
        self.assertEqual(result[0].controls[0].name, "Save")
        self.assertEqual(result[0].controls[0].control_type, "Button")
        self.assertEqual(result[0].controls[0].rectangle.right, 301)

    def test_observer_title_filter_is_case_insensitive(self):
        desktop = _Desktop([_Window("Calculator", []), _Window("Notes", [])])
        observer = WindowsUIAObserver(platform_name="Windows", desktop_factory=lambda **kwargs: desktop)
        result = observer.snapshot_windows(title_contains="calc")
        self.assertEqual([item.title for item in result], ["Calculator"])

    def test_non_windows_observer_fails_closed(self):
        observer = WindowsUIAObserver(platform_name="Linux", desktop_factory=lambda **kwargs: _Desktop([]))
        self.assertFalse(observer.available)
        with self.assertRaises(WindowsUIAError):
            observer.snapshot_windows()

    def test_observer_rejects_unbounded_limits(self):
        observer = WindowsUIAObserver(platform_name="Windows", desktop_factory=lambda **kwargs: _Desktop([]))
        with self.assertRaises(ValueError):
            observer.snapshot_windows(max_windows=101)
        with self.assertRaises(ValueError):
            observer.snapshot_windows(max_controls_per_window=1001)


if __name__ == "__main__":
    unittest.main()
