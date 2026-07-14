"""
Surgical AST-guided source editor.

The LLM emits *structured* edit operations (see :mod:`agent.schemas`) instead of
rewriting whole files.  This is the main **token-cost lever**: output tokens are
the expensive part of an LLM call, and re-emitting a 200-line file to change one
function costs ~200 lines every iteration.  A ``replace_function_body`` edit
costs only the new body.

Design
------
We locate edit targets with the ``ast`` module (which gives exact
``lineno`` / ``end_lineno`` / ``col_offset`` for every node on Python 3.8+),
then splice the change into the *original source lines*.  Untouched code keeps
its exact formatting — no round-trip through a code generator.

Every operation is validated: after applying an edit set, the whole file must
still ``ast.parse`` cleanly, or the edit is rejected and nothing is written.

Supported operations (matching ``agent.schemas``):
  * ``replace_function_body``  — swap a function/method body, keep the signature
  * ``replace_definition``     — swap an entire def / async def / class
  * ``insert_definition``      — insert a def/class before/after/into a target
  * ``delete_definition``      — remove a def / class
  * ``add_import``             — append import statements to the import block
  * ``replace_imports``        — replace the whole top-level import block
  * ``replace_global``         — replace a module-level assignment
  * ``replace_file``           — escape hatch: overwrite the whole file
"""

from __future__ import annotations

import ast
import logging
import textwrap
from typing import Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)


class EditError(Exception):
    """Raised when an edit cannot be located or applied."""


# ---------------------------------------------------------------------------
# AST lookup helpers
# ---------------------------------------------------------------------------

_DefNode = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _parse(code: str) -> ast.Module:
    try:
        return ast.parse(code)
    except SyntaxError as exc:  # pragma: no cover - surfaced to caller
        raise EditError(f"Source does not parse: {exc}") from exc


def _find_def(
    tree: ast.Module, qualified: str
) -> Tuple[ast.AST, Optional[ast.AST]]:
    """Locate a definition by dotted name, e.g. ``Foo.bar`` or ``baz``.

    Returns ``(node, parent)`` where *parent* is the enclosing class node for a
    method, else ``None``.
    """
    parts = qualified.split(".")
    scope: List[ast.stmt] = list(tree.body)
    parent: Optional[ast.AST] = None
    node: Optional[ast.AST] = None

    for i, name in enumerate(parts):
        node = None
        for candidate in scope:
            if isinstance(candidate, _DefNode) and candidate.name == name:
                node = candidate
                break
        if node is None:
            raise EditError(f"Definition not found: {qualified!r} (missing {name!r})")
        if i < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                raise EditError(f"{name!r} in {qualified!r} is not a class")
            parent = node
            scope = list(node.body)

    assert node is not None
    return node, parent


def _node_line_span(node: ast.AST) -> Tuple[int, int]:
    """1-indexed inclusive ``(start, end)`` line span, decorators included."""
    start = node.lineno  # type: ignore[attr-defined]
    decorators = getattr(node, "decorator_list", []) or []
    if decorators:
        start = min(start, min(d.lineno for d in decorators))
    end = node.end_lineno  # type: ignore[attr-defined]
    return start, end


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _signature_end_line(lines: List[str], node: ast.AST) -> int:
    """1-indexed line holding the ``:`` that ends a def/class header.

    Handles multi-line signatures by tracking bracket depth.  Falls back to the
    line before the first body statement if no clean colon is found.
    """
    depth = 0
    start = node.lineno  # type: ignore[attr-defined]
    for i in range(start - 1, len(lines)):
        stripped = lines[i].split("#", 1)[0]
        for ch in stripped:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
        if depth <= 0 and stripped.rstrip().endswith(":"):
            return i + 1
    return node.body[0].lineno - 1  # type: ignore[attr-defined]


def _reindent(content: str, indent: str) -> List[str]:
    """Dedent *content* to column 0 then re-indent every non-blank line."""
    dedented = textwrap.dedent(content.strip("\n"))
    out: List[str] = []
    for ln in dedented.splitlines():
        out.append(indent + ln if ln.strip() else "")
    return out


def _parses_as_block(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def _normalize_body(content: str) -> str:
    """Normalize LLM-supplied function-body statements to column 0.

    LLMs frequently emit a *ragged first line*: the first statement flush-left
    but the remaining lines indented (as if the body's base indent were 4).
    ``textwrap.dedent`` can't fix this (the common prefix is 0 because of the
    first line), leaving code that fails to parse. We detect that specific,
    common case — first non-blank line less-indented than the rest and not a
    block opener — and lift the first line to match, then dedent.
    """
    text = content.strip("\n")
    dedented = textwrap.dedent(text)
    if _parses_as_block(dedented):
        return dedented

    lines = text.split("\n")
    nb = [(i, l) for i, l in enumerate(lines) if l.strip()]
    if len(nb) >= 2:
        first_indent = len(nb[0][1]) - len(nb[0][1].lstrip())
        rest_min = min(len(l) - len(l.lstrip()) for _, l in nb[1:])
        if first_indent < rest_min and not nb[0][1].rstrip().endswith(":"):
            lines[nb[0][0]] = " " * rest_min + nb[0][1].lstrip()
            repaired = textwrap.dedent("\n".join(lines))
            if _parses_as_block(repaired):
                return repaired
    return dedented  # best effort; the whole-file parse check will reject if bad


# ---------------------------------------------------------------------------
# Individual operations — each returns a NEW list of source lines
# ---------------------------------------------------------------------------


def _op_replace_function_body(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    node, _ = _find_def(tree, op.target)
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise EditError(f"{op.target!r} is not a function/method")
    if not node.body:
        raise EditError(f"{op.target!r} has an empty body")

    # Replace everything after the signature's ``:`` (incl. any leading body
    # comments/blank lines) up to the end of the function.
    sig_end = _signature_end_line(lines, node)
    body_end = node.end_lineno  # type: ignore[attr-defined]
    body_indent = " " * node.body[0].col_offset
    # Normalize ragged LLM indentation to column 0, then indent to body level.
    block = _normalize_body(op.content)
    new_body = [body_indent + ln if ln.strip() else "" for ln in block.splitlines()]
    if not any(l.strip() for l in new_body):
        raise EditError("replace_function_body content is empty")
    return lines[:sig_end] + new_body + lines[body_end:]


def _op_replace_definition(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    node, _ = _find_def(tree, op.target)
    start, end = _node_line_span(node)
    indent = _indent_of(lines[start - 1])
    new_def = _reindent(op.content, indent)
    return lines[: start - 1] + new_def + lines[end:]


def _op_delete_definition(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    node, _ = _find_def(tree, op.target)
    start, end = _node_line_span(node)
    return lines[: start - 1] + lines[end:]


def _op_insert_definition(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    node, parent = _find_def(tree, op.target)
    position = op.position

    if position in ("before", "after"):
        start, end = _node_line_span(node)
        indent = _indent_of(lines[start - 1])
        block = _reindent(op.content, indent)
        block = block + [""]  # blank line separator
        if position == "before":
            return lines[: start - 1] + block + lines[start - 1 :]
        return lines[:end] + [""] + _reindent(op.content, indent) + lines[end:]

    # start / end → insert INTO the target class body
    if not isinstance(node, ast.ClassDef):
        raise EditError(f"'start'/'end' require a class target, got {op.target!r}")
    if not node.body:
        raise EditError(f"class {op.target!r} has an empty body")
    child_indent = " " * node.body[0].col_offset
    block = _reindent(op.content, child_indent)

    if position == "start":
        anchor = node.body[0].lineno
        return lines[: anchor - 1] + block + [""] + lines[anchor - 1 :]
    # end
    anchor = node.body[-1].end_lineno  # type: ignore[attr-defined]
    return lines[:anchor] + [""] + block + lines[anchor:]


def _top_level_imports(tree: ast.Module) -> List[ast.stmt]:
    return [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def _op_add_import(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    imports = _top_level_imports(tree)
    new_lines = _reindent(op.content, "")
    if imports:
        anchor = max(n.end_lineno for n in imports)  # type: ignore[attr-defined]
        return lines[:anchor] + new_lines + lines[anchor:]
    # No imports: insert after a module docstring if present, else at top.
    insert_at = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insert_at = tree.body[0].end_lineno  # type: ignore[attr-defined]
    return lines[:insert_at] + new_lines + [""] + lines[insert_at:]


def _op_replace_imports(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    imports = _top_level_imports(tree)
    new_lines = _reindent(op.content, "")
    if not imports:
        # Nothing to replace → behave like add_import.
        return _op_add_import(lines, op)
    start = min(n.lineno for n in imports)
    end = max(n.end_lineno for n in imports)  # type: ignore[attr-defined]
    return lines[: start - 1] + new_lines + lines[end:]


def _op_replace_global(lines: List[str], op) -> List[str]:
    tree = _parse("\n".join(lines))
    target = op.target
    for stmt in tree.body:
        names: List[str] = []
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.append(stmt.target.id)
        if target in names:
            start, end = stmt.lineno, stmt.end_lineno  # type: ignore[attr-defined]
            indent = _indent_of(lines[start - 1])
            new_lines = _reindent(op.content, indent)
            return lines[: start - 1] + new_lines + lines[end:]
    raise EditError(f"Global assignment not found: {target!r}")


def _op_replace_file(lines: List[str], op) -> List[str]:
    return op.content.splitlines()


_DISPATCH = {
    "replace_function_body": _op_replace_function_body,
    "replace_definition": _op_replace_definition,
    "insert_definition": _op_insert_definition,
    "delete_definition": _op_delete_definition,
    "add_import": _op_add_import,
    "replace_imports": _op_replace_imports,
    "replace_global": _op_replace_global,
    "replace_file": _op_replace_file,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_edits(original: str, ops: Iterable) -> str:
    """Apply a sequence of edit ops to *original* source, return new source.

    Operations are applied **in order**, each on the result of the previous one
    (re-parsing every time so later ops see up-to-date line numbers).  If the
    final result does not parse, an :class:`EditError` is raised and the caller
    should discard the change.
    """
    ops = list(ops)
    lines = original.splitlines()
    is_python = True  # non-python files (yaml) only allow replace_file

    for op in ops:
        operation = op.operation
        handler = _DISPATCH.get(operation)
        if handler is None:
            raise EditError(f"Unknown edit operation: {operation!r}")
        lines = handler(lines, op)

    result = "\n".join(lines)
    if original.endswith("\n") and not result.endswith("\n"):
        result += "\n"

    # Final validation: the whole file must still parse as Python.
    # (Callers editing non-Python files should use replace_file exclusively.)
    if _looks_like_python(ops):
        try:
            ast.parse(result)
        except SyntaxError as exc:
            raise EditError(f"Result does not parse after edits: {exc}") from exc
    return result


def _looks_like_python(ops: List) -> bool:
    """Only validate-as-python when we used AST ops (not a raw replace_file)."""
    return not (len(ops) == 1 and ops[0].operation == "replace_file")
