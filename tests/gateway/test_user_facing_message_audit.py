"""Guard gateway send paths against new hard-coded user-facing prose.

The gateway has several transport-specific send methods, so searching for a
variable named ``message`` misses literals assigned to ``msg``, ``notice``, or
``content``.  This AST audit starts at the actual send sinks and follows one
simple local assignment backwards.  Static prose must pass through ``t()`` or
another renderer (for example the SOUL status-label renderer) before sending.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDITED_PATHS = (
    ROOT / "gateway" / "run.py",
    ROOT / "gateway" / "platforms" / "base.py",
    ROOT / "plugins" / "platforms" / "discord" / "adapter.py",
)
SEND_METHODS = {
    "send",
    "_send_with_retry",
    "send_message",
    "send_response",
    "edit_message",
}
CONTENT_KEYWORDS = {"content", "text", "message"}
DISCORD_UI_FIELDS = {
    "discord.Embed": {"title", "description"},
    "discord.SelectOption": {"label", "description"},
    "discord.ui.Button": {"label"},
    "discord.ui.Select": {"placeholder"},
    "discord.ui.button": {"label"},
}
PROSE_RE = re.compile(r"[A-Za-z]{3}|[\u3040-\u30ff\u4e00-\u9fff]")


def _call_name(node: ast.expr) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_text(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                nested = _literal_text(value.value)
                if nested is not None:
                    parts.append(nested)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_text(node.left)
        right = _literal_text(node.right)
        text = "".join(part for part in (left, right) if part is not None)
        return text or None
    if isinstance(node, ast.BoolOp):
        text = "".join(
            part
            for value in node.values
            if (part := _literal_text(value)) is not None
        )
        return text or None
    if isinstance(node, ast.IfExp):
        text = "".join(
            part
            for value in (node.body, node.orelse)
            if (part := _literal_text(value)) is not None
        )
        return text or None
    return None


def _positional_content_index(call_name: str, method: str) -> int:
    """Return the content position for Hermes adapters or SDK send calls."""
    if method == "_send_with_retry":
        return 1
    if method == "edit_message":
        if call_name == "self.edit_message" or call_name.endswith(
            "adapter.edit_message"
        ):
            return 2
        return 0
    if method != "send":
        return 0
    if call_name == "self.send" or call_name.endswith("adapter.send"):
        return 1
    return 0


class _SendLiteralVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self._scopes: list[dict[str, ast.expr]] = [{}]
        self.failures: list[tuple[int, str, str]] = []

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._scopes.append({})
        self.generic_visit(node)
        self._scopes.pop()

    visit_FunctionDef = _visit_scoped
    visit_AsyncFunctionDef = _visit_scoped

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._scopes[-1][target.id] = node.value
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._scopes[-1][node.target.id] = node.value
        if node.value is not None:
            self.visit(node.value)

    def _resolve_name(
        self,
        node: ast.expr | None,
        seen: set[str] | None = None,
    ) -> ast.expr | None:
        if not isinstance(node, ast.Name):
            return node
        seen = seen or set()
        if node.id in seen:
            return node
        seen.add(node.id)
        for scope in reversed(self._scopes):
            if node.id in scope:
                return self._resolve_name(scope[node.id], seen)
        return node

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        method = name.rsplit(".", 1)[-1]
        if method in SEND_METHODS:
            content = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in CONTENT_KEYWORDS
                ),
                None,
            )
            if content is None:
                positional_index = _positional_content_index(name, method)
                if len(node.args) > positional_index:
                    content = node.args[positional_index]

            text = _literal_text(self._resolve_name(content))
            if text and PROSE_RE.search(text):
                self.failures.append((node.lineno, name, text[:120]))

        # Plain-content prompt builders are user-facing sinks too. Inspecting
        # their arguments catches fallback prose hidden behind expressions such
        # as ``title or "Confirm"`` before the result reaches ``send()``.
        if method == "_self_contained_prompt_content":
            builder_values = list(node.args)
            builder_values.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "tail"
            )
            for value in builder_values:
                text = _literal_text(self._resolve_name(value))
                if text and PROSE_RE.search(text):
                    self.failures.append((node.lineno, name, text[:120]))

        ui_fields = DISCORD_UI_FIELDS.get(name, set())
        if name.endswith(".add_field"):
            ui_fields = {"name", "value"}
        elif name.endswith(".set_footer"):
            ui_fields = {"text"}

        for keyword in node.keywords:
            if keyword.arg not in ui_fields:
                continue
            text = _literal_text(self._resolve_name(keyword.value))
            if text and PROSE_RE.search(text):
                self.failures.append(
                    (node.lineno, f"{name}.{keyword.arg}", text[:120])
                )

        self.generic_visit(node)


def test_gateway_send_sinks_do_not_receive_hard_coded_prose() -> None:
    failures: list[str] = []
    for path in AUDITED_PATHS:
        visitor = _SendLiteralVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        failures.extend(
            f"{path.relative_to(ROOT)}:{line} ({call_name}): {preview!r}"
            for line, call_name, preview in visitor.failures
        )

    assert not failures, (
        "User-facing prose is flowing directly into a send sink. "
        "Route it through agent.i18n.t() or a SOUL-aware renderer:\n"
        + "\n".join(failures)
    )


def test_audit_detects_boolop_fallback_hidden_in_prompt_builder() -> None:
    tree = ast.parse(
        """
async def send_prompt(self, channel, title, message):
    content = self._self_contained_prompt_content(
        f"**{title or 'Confirm'}**", message
    )
    await channel.send(content=content)
"""
    )
    visitor = _SendLiteralVisitor()
    visitor.visit(tree)

    assert any("Confirm" in preview for _, _, preview in visitor.failures)
