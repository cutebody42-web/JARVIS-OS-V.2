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
            element=SimpleNamespace(CurrentIsPassword=0),
        )

    def rectangle(self):
        return _Rect()

    def is_enabled(self):
        return True

    def is_visible(self):
        return True

    def is_password(self):
        return False

    def children(self):
        return []


class _Window(_Control):
    handle = 1234

    def __init__(self, title, controls):
        super().__init__(title, "Window")
        self._title = title
        self._controls = controls

    def window_text(self):
        return self._title

    def children(self):
        return iter(self._controls)


class _Desktop:
    def __init__(self, windows):
        self._windows = windows

        self.filters = None

    def windows(self, **filters):
        self.filters = filters
        return iter(self._windows)


class WindowsUIATests(unittest.TestCase):
    def test_observer_returns_bounded_semantic_snapshots(self):
        desktop = _Desktop([_Window("Editor", [_Control(), _Control("Cancel")])])
        observer = WindowsUIAObserver(platform_name="Windows", desktop_factory=lambda **kwargs: desktop)
        result = observer.snapshot_windows(
            allow_all_windows=True, max_controls_per_window=1
        )
        self.assertTrue(observer.available)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].title, "Editor")
        self.assertEqual(result[0].handle, 1234)
        self.assertTrue(result[0].truncated)
        self.assertEqual(result[0].controls[0].name, "Save")
        self.assertEqual(result[0].controls[0].control_type, "Button")
        self.assertEqual(result[0].controls[0].rectangle.right, 301)
        self.assertEqual(desktop.filters["visible_only"], True)

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
            observer.snapshot_windows(allow_all_windows=True, max_windows=101)
        with self.assertRaises(ValueError):
            observer.snapshot_windows(
                allow_all_windows=True, max_controls_per_window=1001
            )

    def test_global_observation_requires_explicit_scope(self):
        observer = WindowsUIAObserver(
            platform_name="Windows", desktop_factory=lambda **kwargs: _Desktop([])
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            observer.snapshot_windows()

    def test_exact_process_scope_is_forwarded_and_rechecked(self):
        matching = _Window("Target", [])
        other = _Window("Other", [])
        other.element_info.process_id = 88
        desktop = _Desktop([other, matching])
        observer = WindowsUIAObserver(
            platform_name="Windows", desktop_factory=lambda **kwargs: desktop
        )
        result = observer.snapshot_windows(process_id=77)
        self.assertEqual([item.title for item in result], ["Target"])
        self.assertEqual(desktop.filters["process"], 77)

    def test_password_control_text_is_redacted(self):
        password = _Control("owner-secret", "Edit")
        # Match pywinauto's real shape: the password flag is a property on the
        # wrapped IUIAutomationElement, not a UIAWrapper method.
        password.element_info.element.CurrentIsPassword = 1
        password.is_password = lambda: False
        observer = WindowsUIAObserver(
            platform_name="Windows",
            desktop_factory=lambda **kwargs: _Desktop([_Window("Login", [password])]),
        )
        result = observer.snapshot_windows(title_contains="Login")
        self.assertTrue(result[0].controls[0].sensitive)
        self.assertEqual(result[0].controls[0].name, "[redacted sensitive control]")

    def test_child_enumeration_failure_is_not_reported_as_empty(self):
        window = _Window("Broken", [])
        window.children = lambda: (_ for _ in ()).throw(RuntimeError("COM detail"))
        observer = WindowsUIAObserver(
            platform_name="Windows",
            desktop_factory=lambda **kwargs: _Desktop([window]),
        )
        with self.assertRaisesRegex(WindowsUIAError, "RuntimeError") as caught:
            observer.snapshot_windows(title_contains="Broken")
        self.assertNotIn("COM detail", str(caught.exception))

    def test_unknown_password_state_redacts_without_reading_name(self):
        class UnavailableInfo:
            control_type = "Edit"
            automation_id = "password"
            class_name = "Edit"
            process_id = 77

            @property
            def element(self):
                raise RuntimeError("COM unavailable")

            @property
            def name(self):
                raise AssertionError("sensitive name must not be read")

        control = _Control()
        control.element_info = UnavailableInfo()
        control.is_password = None
        snapshot = WindowsUIAObserver._element(control)
        self.assertTrue(snapshot.sensitive)
        self.assertEqual(snapshot.name, "[redacted sensitive control]")

    def test_window_iterator_stops_at_bound_instead_of_materializing_everything(self):
        class BoundedDesktop(_Desktop):
            def windows(self, **filters):
                self.filters = filters
                yield _Window("One", [])
                yield _Window("Two", [])
                raise AssertionError("observer consumed beyond max_windows + 1")

        observer = WindowsUIAObserver(
            platform_name="Windows",
            desktop_factory=lambda **kwargs: BoundedDesktop([]),
        )
        with self.assertRaisesRegex(WindowsUIAError, "exceeds"):
            observer.snapshot_windows(allow_all_windows=True, max_windows=1)

    def test_window_iterator_failure_is_redacted(self):
        class BrokenDesktop(_Desktop):
            def windows(self, **filters):
                yield _Window("One", [])
                raise RuntimeError("private COM detail")

        observer = WindowsUIAObserver(
            platform_name="Windows",
            desktop_factory=lambda **kwargs: BrokenDesktop([]),
        )
        with self.assertRaisesRegex(WindowsUIAError, "RuntimeError") as caught:
            observer.snapshot_windows(allow_all_windows=True)
        self.assertNotIn("private COM detail", str(caught.exception))

    def test_provider_scan_is_bounded_even_when_it_ignores_filters(self):
        class IgnoringDesktop(_Desktop):
            def windows(self, **filters):
                while True:
                    yield _Window("other-window", [])

        observer = WindowsUIAObserver(
            platform_name="Windows",
            desktop_factory=lambda **kwargs: IgnoringDesktop([]),
        )
        with self.assertRaisesRegex(WindowsUIAError, "provider scan"):
            observer.snapshot_windows(title_contains="target", max_windows=1)


if __name__ == "__main__":
    unittest.main()
