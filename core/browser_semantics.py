"""Read-only semantic DOM observation for NEXUS browser reasoning.

The architecture follows the accessibility/DOM-first lesson from modern browser
agent projects such as browser-use, but keeps JARVIS' authority boundary: this
module can observe a page; it cannot click, type, navigate, submit, download, or
run model-provided JavaScript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BrowserSemanticError(RuntimeError):
    pass


@dataclass(frozen=True)
class DOMRect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DOMElementSnapshot:
    index: int
    tag: str
    role: str
    name: str
    text: str
    input_type: str
    href: str
    disabled: bool
    visible: bool
    rectangle: DOMRect | None


@dataclass(frozen=True)
class DOMSnapshot:
    url: str
    title: str
    elements: tuple[DOMElementSnapshot, ...]
    truncated: bool


# Fixed, reviewable observation script. No caller-provided script fragments are
# accepted. It reads semantic/interactive nodes and geometry only.
_OBSERVE_SCRIPT = r"""
({ maxNodes, includeHidden }) => {
  const selector = [
    'a[href]', 'button', 'input', 'textarea', 'select', 'option',
    '[role]', '[contenteditable="true"]',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'main', 'nav', 'header', 'footer', 'form', 'label', 'summary'
  ].join(',');
  const all = Array.from(document.querySelectorAll(selector));
  const out = [];
  for (const el of all) {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const visible = !!(
      rect.width > 0 && rect.height > 0 &&
      style.visibility !== 'hidden' && style.display !== 'none' &&
      Number(style.opacity || '1') > 0
    );
    if (!includeHidden && !visible) continue;
    const aria = el.getAttribute('aria-label') || '';
    const labelledBy = el.getAttribute('aria-labelledby') || '';
    let labelled = '';
    if (labelledBy) {
      labelled = labelledBy.split(/\s+/).map(id => document.getElementById(id)?.textContent || '').join(' ');
    }
    const alt = el.getAttribute('alt') || '';
    const placeholder = el.getAttribute('placeholder') || '';
    const name = aria || labelled || alt || placeholder || '';
    out.push({
      tag: (el.tagName || '').toLowerCase(),
      role: el.getAttribute('role') || '',
      name,
      text: el.innerText || el.textContent || '',
      inputType: el.getAttribute('type') || '',
      href: el instanceof HTMLAnchorElement ? (el.href || '') : '',
      disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
      visible,
      rect: {x: rect.x, y: rect.y, width: rect.width, height: rect.height}
    });
    if (out.length >= maxNodes + 1) break;
  }
  return {url: location.href, title: document.title || '', nodes: out};
}
"""


def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", "").split())[:limit]


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return 0.0
    return max(-1_000_000.0, min(1_000_000.0, number))


class BrowserSemanticObserver:
    """Create bounded semantic snapshots from a Playwright-compatible Page."""

    async def snapshot(
        self,
        page,
        *,
        max_nodes: int = 300,
        include_hidden: bool = False,
    ) -> DOMSnapshot:
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= 800:
            raise ValueError("max_nodes must be between 1 and 800.")
        if not isinstance(include_hidden, bool):
            raise ValueError("include_hidden must be boolean.")
        if page is None or not callable(getattr(page, "evaluate", None)):
            raise ValueError("A Playwright-compatible page is required.")
        try:
            payload = await page.evaluate(
                _OBSERVE_SCRIPT,
                {"maxNodes": max_nodes, "includeHidden": include_hidden},
            )
        except Exception as exc:
            raise BrowserSemanticError(f"DOM observation failed ({type(exc).__name__}).") from None
        if not isinstance(payload, dict):
            raise BrowserSemanticError("DOM observation returned an invalid payload.")
        raw_nodes = payload.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raw_nodes = []
        truncated = len(raw_nodes) > max_nodes
        nodes: list[DOMElementSnapshot] = []
        for raw in raw_nodes[:max_nodes]:
            if not isinstance(raw, dict):
                continue
            rect_raw = raw.get("rect")
            rectangle = None
            if isinstance(rect_raw, dict):
                rectangle = DOMRect(
                    _number(rect_raw.get("x")),
                    _number(rect_raw.get("y")),
                    max(0.0, _number(rect_raw.get("width"))),
                    max(0.0, _number(rect_raw.get("height"))),
                )
            nodes.append(DOMElementSnapshot(
                index=len(nodes),
                tag=_text(raw.get("tag"), 40).lower(),
                role=_text(raw.get("role"), 100),
                name=_text(raw.get("name"), 500),
                text=_text(raw.get("text"), 1000),
                input_type=_text(raw.get("inputType"), 80),
                href=_text(raw.get("href"), 2048),
                disabled=raw.get("disabled") is True,
                visible=raw.get("visible") is True,
                rectangle=rectangle,
            ))
        return DOMSnapshot(
            url=_text(payload.get("url"), 4096),
            title=_text(payload.get("title"), 500),
            elements=tuple(nodes),
            truncated=truncated,
        )

    def observation_script(self) -> str:
        """Expose the fixed script for audits/tests; callers cannot replace it."""
        return _OBSERVE_SCRIPT
