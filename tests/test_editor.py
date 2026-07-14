"""Standalone tests for agent/editor.py — run before wiring into the agent."""
import ast, sys, types
sys.path.insert(0, "/home/olga/myworld/Projects/research_agent_2/mysearch_v0")
from agent.editor import apply_edits, EditError

# Lightweight op stand-ins (mirror the pydantic schema fields the editor reads).
def op(operation, **kw):
    o = types.SimpleNamespace(operation=operation)
    for k, v in kw.items():
        setattr(o, k, v)
    return o

SRC = '''\
"""Module docstring."""
import os
import sys

X = 1
Y = [1, 2, 3]


def helper(a, b):
    """Add."""
    return a + b


class Model:
    def __init__(self, n):
        self.n = n

    def forward(self, x):
        # old body
        return x * self.n
'''

def check_parses(s):
    ast.parse(s)
    return s

def t_replace_function_body():
    out = apply_edits(SRC, [op("replace_function_body", target="Model.forward",
                               content="return x * self.n + 1")])
    check_parses(out)
    assert "return x * self.n + 1" in out
    assert "old body" not in out
    # signature + siblings preserved
    assert "def forward(self, x):" in out and "def __init__(self, n):" in out
    print("ok replace_function_body")

def t_replace_definition():
    out = apply_edits(SRC, [op("replace_definition", target="helper",
                               content="def helper(a, b, c=0):\n    return a + b + c")])
    check_parses(out)
    assert "c=0" in out and "def helper" in out
    print("ok replace_definition")

def t_replace_method_definition():
    out = apply_edits(SRC, [op("replace_definition", target="Model.__init__",
                               content="def __init__(self, n, bias=0):\n    self.n = n\n    self.bias = bias")])
    check_parses(out)
    assert "bias=0" in out
    assert "    def __init__" in out  # re-indented into class
    print("ok replace_definition (method)")

def t_delete_definition():
    out = apply_edits(SRC, [op("delete_definition", target="helper")])
    check_parses(out)
    assert "def helper" not in out
    assert "class Model" in out
    print("ok delete_definition")

def t_insert_after():
    out = apply_edits(SRC, [op("insert_definition", target="helper", position="after",
                               content="def helper2(x):\n    return x - 1")])
    check_parses(out)
    assert "def helper2" in out
    # order: helper before helper2 before class
    assert out.index("def helper2") < out.index("class Model")
    print("ok insert_definition after")

def t_insert_before():
    out = apply_edits(SRC, [op("insert_definition", target="helper", position="before",
                               content="def helper0(x):\n    return x")])
    check_parses(out)
    assert out.index("def helper0") < out.index("def helper(")
    print("ok insert_definition before")

def t_insert_class_end():
    out = apply_edits(SRC, [op("insert_definition", target="Model", position="end",
                               content="def extra(self):\n    return self.n")])
    check_parses(out)
    assert "    def extra(self):" in out  # indented as method
    tree = ast.parse(out)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef)][0]
    assert any(getattr(m, "name", "") == "extra" for m in cls.body)
    print("ok insert_definition class end")

def t_insert_class_start():
    out = apply_edits(SRC, [op("insert_definition", target="Model", position="start",
                               content="CONST = 5")])
    check_parses(out)
    tree = ast.parse(out)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef)][0]
    assert isinstance(cls.body[0], ast.Assign)
    print("ok insert_definition class start")

def t_add_import():
    out = apply_edits(SRC, [op("add_import", content="import math\nfrom typing import List")])
    check_parses(out)
    assert "import math" in out
    # inserted within/after the import block, before X = 1
    assert out.index("import math") < out.index("X = 1")
    print("ok add_import")

def t_replace_imports():
    out = apply_edits(SRC, [op("replace_imports", content="import numpy as np")])
    check_parses(out)
    assert "import numpy as np" in out
    assert "import os" not in out and "import sys" not in out
    assert '"""Module docstring."""' in out  # docstring preserved
    print("ok replace_imports")

def t_replace_global():
    out = apply_edits(SRC, [op("replace_global", target="Y", content="Y = [4, 5, 6, 7]")])
    check_parses(out)
    assert "[4, 5, 6, 7]" in out
    assert "X = 1" in out
    print("ok replace_global")

def t_chained_ops():
    # multiple edits in one plan, applied sequentially
    out = apply_edits(SRC, [
        op("add_import", content="import math"),
        op("replace_function_body", target="helper", content="return math.hypot(a, b)"),
        op("replace_global", target="X", content="X = 42"),
    ])
    check_parses(out)
    assert "import math" in out and "math.hypot" in out and "X = 42" in out
    print("ok chained ops")

def t_reject_broken():
    try:
        apply_edits(SRC, [op("replace_function_body", target="helper",
                             content="return a +")])  # syntax error
        print("FAIL: broken edit was accepted")
    except EditError:
        print("ok rejects broken result")

def t_missing_target():
    try:
        apply_edits(SRC, [op("replace_function_body", target="nope", content="pass")])
        print("FAIL: missing target accepted")
    except EditError:
        print("ok rejects missing target")

def t_ragged_first_line_body():
    # LLM pattern: first stmt flush-left, rest indented by 4 (real bug from run).
    ragged = "X = np.asarray(x)\n    y = x + 1\n    if X > 0:\n        y = 0\n    return y"
    out = apply_edits(SRC, [op("replace_function_body", target="helper", content=ragged)])
    check_parses(out)
    assert "return y" in out and "X = np.asarray(x)" in out
    print("ok ragged-first-line body normalized")

def t_uniformly_indented_body():
    # LLM adds outer indent uniformly — must still work.
    body = "    a = 1\n    b = 2\n    return a + b"
    out = apply_edits(SRC, [op("replace_function_body", target="helper", content=body)])
    check_parses(out); assert "return a + b" in out
    print("ok uniformly-indented body")

def t_replace_file():
    out = apply_edits(SRC, [op("replace_file", content="X = 1\n")])
    assert out.strip() == "X = 1"
    print("ok replace_file")

for fn in sorted([f for f in dir() if f.startswith("t_")]):
    globals()[fn]()
print("\nALL EDITOR TESTS PASSED")
