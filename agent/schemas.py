"""
Structured-output schemas for the agent.

The LLM emits **surgical edits** rather than whole-file rewrites — this is the
primary token-cost lever, since output tokens dominate LLM cost and re-emitting
a whole file to change one function is wasteful.  See :mod:`agent.editor` for
the applier.

Schema shape
------------
We use a single *flat* ``EditOp`` model (an ``operation`` enum plus optional
``target`` / ``position`` / ``content`` fields) instead of a discriminated
union.  Flat schemas are far more portable across ``json_schema`` structured
outputs (no ``oneOf`` / ``anyOf``, which several providers reject in strict
mode).  Which fields a given operation needs is validated in Python, not the
schema — see ``agent.orchestrator.validate_plan``.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# The operations understood by agent.editor._DISPATCH.
Operation = Literal[
    "replace_function_body",
    "replace_definition",
    "insert_definition",
    "delete_definition",
    "add_import",
    "replace_imports",
    "replace_global",
    "replace_file",
]


class EditOp(BaseModel):
    """A single surgical edit to one file."""

    operation: Operation = Field(
        description=(
            "The kind of edit. "
            "'replace_function_body': swap a function/method body, keeping its "
            "signature (content = statements only, no 'def' line). "
            "'replace_definition': swap an entire function/class (content = full "
            "definition incl. signature). "
            "'insert_definition': add a new function/class (needs 'position'). "
            "'delete_definition': remove a function/class (no content). "
            "'add_import': append import line(s) (content = import statements). "
            "'replace_imports': replace the whole top-level import block. "
            "'replace_global': replace a module-level assignment (target = var "
            "name, content = full assignment). "
            "'replace_file': overwrite the whole file — LAST RESORT, expensive; "
            "use only to bootstrap a file or for a non-Python file (e.g. YAML)."
        )
    )
    target: Optional[str] = Field(
        default=None,
        description=(
            "Qualified name the edit acts on, e.g. 'build_model', "
            "'MyNet.forward', or a global variable name. Leave null for "
            "add_import / replace_imports / replace_file."
        ),
    )
    position: Optional[Literal["before", "after", "start", "end"]] = Field(
        default=None,
        description=(
            "Only for insert_definition. 'before'/'after' place a sibling "
            "relative to target; 'start'/'end' insert into the target class."
        ),
    )
    content: Optional[str] = Field(
        default=None,
        description=(
            "The new code/text for the edit. Omit for delete_definition. "
            "Do NOT include outer indentation — it is applied automatically."
        ),
    )


class FileEdits(BaseModel):
    """An ordered set of edits to a single file."""

    filename: str = Field(
        description="File to edit, relative to the model dir (e.g. 'model.py')."
    )
    edit_list: List[EditOp] = Field(
        description=(
            "Edits applied in order. Emit the minimum number of edits needed."
        )
    )


class ExperimentPlan(BaseModel):
    """One atomic experiment: reasoning + edits to a single target group."""

    reasoning: str = Field(
        description="Step-by-step reasoning for the proposed change."
    )
    short_description: str = Field(
        description="1-2 sentences describing the experiment (used as commit msg)."
    )
    edits: List[FileEdits] = Field(
        description=(
            "File edits. All files MUST belong to a single atomic target group "
            "(see the system prompt)."
        )
    )
