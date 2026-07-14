from enum import Enum
from typing import Annotated, Literal
from pydantic import BaseModel, Field


class EditOperation(str, Enum):
    REPLACE_FUNCTION_BODY = "replace_function_body"
    REPLACE_DEFINITION = "replace_definition"
    INSERT_DEFINITION = "insert_definition"
    DELETE_DEFINITION = "delete_definition"
    ADD_IMPORT = "add_import"
    REPLACE_IMPORTS = "replace_imports"
    REPLACE_GLOBAL = "replace_global"


class ReplaceFunctionBody(BaseModel):
    """
    Replace only the body of an existing function or method.

    The function signature, decorators, docstring (unless included in the new
    body), and surrounding definition remain unchanged.
    """
    operation: Literal["replace_function_body"]
    target: str = Field(
        description=(
            "Qualified function or method name. "
            "Examples: 'foo', 'MyClass.calculate'."
        )
    )
    content: str = Field(
        description=(
            "New function body containing only statements. "
            "Do not include the 'def' line or leading indentation."
        )
    )


class ReplaceDefinition(BaseModel):
    """Replace an entire top-level definition.

    Applicable to functions, async functions, and classes.
    """

    operation: Literal["replace_definition"]

    target: str = Field(
        description=(
            "Qualified name of the definition to replace. "
            "Examples: 'foo', 'MyClass.bar', 'MyClass'."
        )
    )

    content: str = Field(
        description=(
            "Complete replacement definition, including the function "
            "signature or class declaration. Do not include outer indentation."
        )
    )


class InsertDefinition(BaseModel):
    """Insert a new definition relative to an existing definition or class."""

    operation: Literal["insert_definition"]

    target: str = Field(
        description=(
            "Qualified name of the reference definition or class. "
            "Examples: 'foo', 'MyClass', 'MyClass.bar'."
        )
    )

    position: Literal["before", "after", "start", "end"] = Field(
        description=(
            "Insertion position relative to the target. "
            "'before'/'after' insert sibling definitions. "
            "'start'/'end' insert into the target class or module."
        )
    )

    content: str = Field(
        description=(
            "Complete definition to insert. Include the function or class "
            "declaration. Do not include outer indentation."
        )
    )


class DeleteDefinition(BaseModel):
    """Delete an existing function, async function, or class definition."""

    operation: Literal["delete_definition"]

    target: str = Field(
        description=(
            "Qualified name of the definition to delete. "
            "Examples: 'foo', 'MyClass.bar', 'MyClass'."
        )
    )


class AddImport(BaseModel):
    """
    Add one or more import statements to the module.
    """
    operation: Literal["add_import"]
    content: str = Field(
        description=(
            "One or more complete import statements. "
            "Examples: 'import os' or 'from pathlib import Path'."
        )
    )


class ReplaceImports(BaseModel):
    """
    Replace the entire module import section.
    """
    operation: Literal["replace_imports"]
    content: str = Field(
        description=(
            "Complete replacement import block containing every import "
            "that should remain in the module."
        )
    )


class ReplaceGlobal(BaseModel):
    """
    Replace a module-level variable assignment.

    This operation applies only to top-level assignments, not class
    attributes or instance attributes.
    """

    operation: Literal["replace_global"]
    target: str = Field(
        description="Name of the module-level variable."
    )
    content: str = Field(
        description=(
            "Complete replacement assignment, e.g. "
            "'MAX_SIZE = 100'."
        )
    )


AstEditOp = Annotated[
    ReplaceFunctionBody
    | ReplaceDefinition
    | InsertDefinition
    | DeleteDefinition
    | AddImport
    | ReplaceImports
    | ReplaceGlobal,
    Field(discriminator="operation"),
]


class FileEdits(BaseModel):
    filename: str
    edit_list: list[AstEditOp] = Field(
        description=(
            "Ordered edit operations. Emit only the minimum number of edits "
            "required to satisfy the user's request."
        )
    )


class ExperimentPlan(BaseModel):
    """Output of the LLM: one proposed experiment with mutations."""

    reasoning: str = Field(description="Step-by-step reasoning for the proposed changes.")
    short_description: str = Field(description="1-2 sentences describing the experiment.")
    edits: list[FileEdits]

