from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from a.schemas import AstEditOp

class EditError(Exception):
    """Base exception for edit operations."""

class ApplyError(EditError):
    """Error applying parsed edit to source code."""

AstNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def apply(self, original_code: str, ops: list[AstEditOp]):
    lines = original_code.splitlines()

    ops_with_pos = _resolve_positions(original_code, ops)



def _resolve_positions(code: str, ops: list[AstEditOp]):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Cannot parse original code: {e}")

    result: list[tuple[int, int, AstEditOp]] = []
    for op in ops:
        if op.operation in ("add_import", "replace_imports"):
            result.append((0, 0, op))
            continue

        if op.operation == "replace_global":
            pos = _find_global_assignment(tree, op.target or "")
            if pos is None:
                raise ApplyError(f"Global assignment not found: {op.target!r}")
            result.append((pos, 0, op))
            continue


# def _find_global_assignment(code: str, target: str):
#     """Find the line number of a module-level assignment."""
#     for i, line in enumerate(code.splitlines(), 1):
#         stripped = line.strip()
#         prefixes = (f"{target} =", f"{target}:")
#         if stripped.startswith(prefixes) and line[0] not in (" ", "\t"):
#             return i
#     return None
