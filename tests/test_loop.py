"""End-to-end smoke test of the agent loop with a MOCK LLM (no API cost).

IMPORTANT: this test runs entirely inside a throwaway TEMP copy of model_dir.
It never touches the real model_dir, its logs.md/runs, or the persistent
history/ archive. (An earlier version pointed at the live model_dir and wiped a
real run — never again: MODEL_DIR + AUTORESEARCH_HISTORY are redirected to a
tempdir BEFORE importing agent.)

Covers: context build -> structured plan -> atomic validation -> surgical edit
apply -> training subprocess -> LocalTracker fetch -> classify -> git protocol
-> logs.md -> persistent archive. Also exercises the two-strike crash path.
"""
import os, sys, shutil, subprocess, tempfile, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC_MODEL = REPO / "model_dir"

# --- build an isolated temp copy of model_dir (tracked files only) ----------
TMP = pathlib.Path(tempfile.mkdtemp(prefix="autoresearch_test_"))
TEST_MD = TMP / "model_dir"
TEST_MD.mkdir(parents=True)
for name in ["data.py", "run.py", "model.py", "loss.py", "optimizer.py",
             "transforms.py", "config.yaml", "wiki.md", ".gitignore", "pyproject.toml"]:
    src = SRC_MODEL / name
    if src.exists():
        shutil.copy2(src, TEST_MD / name)
# Force a fast config so the test doesn't inherit whatever epochs the agent evolved.
(TEST_MD / "config.yaml").write_text("epochs: 5\nbatch_size: 256\nseed: 0\n")
subprocess.run(["git", "init", "-q"], cwd=TEST_MD)
subprocess.run(["git", "config", "user.email", "t@t.local"], cwd=TEST_MD)
subprocess.run(["git", "config", "user.name", "test"], cwd=TEST_MD)
subprocess.run(["git", "add", "-A"], cwd=TEST_MD)
subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=TEST_MD)

# Redirect ALL agent paths into the tempdir BEFORE importing agent.
os.environ["MODEL_DIR"] = str(TEST_MD)
os.environ["AUTORESEARCH_HISTORY"] = str(TMP / "history")
os.environ["EXEC_COMMAND"] = "python3 run.py"
os.environ["TRACKER"] = "local"
os.environ["TARGET_METRIC"] = "val/auprc"
os.environ["TARGET_DIRECTION"] = "max"
os.environ["PROMPT_CACHE"] = "0"
sys.path.insert(0, str(REPO))

from agent import config, run_loop
from agent.schemas import ExperimentPlan, FileEdits, EditOp

assert config.MODEL_DIR == TEST_MD, "SAFETY: config must point at the temp dir"
assert str(config.ARCHIVE_DIR).startswith(str(TMP)), "SAFETY: archive must be in tempdir"


class MockLLM:
    def __init__(self, plans):
        self.plans = plans; self.i = 0; self.last_usage = {}

    def chat_structured(self, messages, response_format, model=None, reasoning_effort="medium", **kw):
        joined = "\n".join(str(m["content"]) for m in messages)
        assert "Current code" in joined and "build_optimizer" in joined, "context missing code"
        plan = self.plans[min(self.i, len(self.plans) - 1)]; self.i += 1
        self.last_usage = {"prompt_tokens": 1500, "completion_tokens": 300,
                           "cached_tokens": 1200, "cost": 0.0042}
        return plan


PLAN_ADAM = ExperimentPlan(
    reasoning="SGD under-trains; Adam converges faster.",
    short_description="Switch optimizer from SGD to Adam",
    edits=[FileEdits(filename="optimizer.py", edit_list=[
        EditOp(operation="replace_function_body", target="build_optimizer",
               content="optimizer = torch.optim.Adam(params, lr=0.01)\nscheduler = None\nreturn optimizer, scheduler")])],
)
PLAN_BREAK = ExperimentPlan(
    reasoning="(intentionally broken) delete build_model to test crash+revert.",
    short_description="Broken edit (should be reverted)",
    edits=[FileEdits(filename="model.py", edit_list=[
        EditOp(operation="delete_definition", target="build_model")])],
)
PLAN_POSWEIGHT = ExperimentPlan(
    reasoning="Address imbalance with BCE pos_weight.",
    short_description="Add pos_weight to BCE loss",
    edits=[FileEdits(filename="loss.py", edit_list=[
        EditOp(operation="replace_function_body", target="build_loss", content=(
            "pos = float(y_train.sum()); neg = float(len(y_train) - pos)\n"
            "pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)\n"
            "criterion = nn.BCEWithLogitsLoss(pos_weight=pw)\n"
            "def loss_fn(logits, targets):\n    return criterion(logits, targets)\n"
            "return loss_fn"))])],
)

try:
    print("### Iteration set 1: happy path (Adam) ###")
    run_loop(model_dir=TEST_MD, max_iterations=1, llm=MockLLM([PLAN_ADAM]))

    print("### Iteration set 2: crash (strike 1) then recover (pos_weight) ###")
    run_loop(model_dir=TEST_MD, max_iterations=2, llm=MockLLM([PLAN_BREAK, PLAN_POSWEIGHT]))

    print("\n=== logs.md ===")
    print(config.LOGS_FILE.read_text())
    print("=== persistent archive (history/experiments.jsonl) ===")
    arch = config.ARCHIVE_DIR / "experiments.jsonl"
    print(arch.read_text())

    assert "build_model" in (TEST_MD / "model.py").read_text(), "broken edit not reverted!"
    assert arch.exists() and arch.read_text().count("\n") >= 3, "archive should have >=3 records"
    print("ALL LOOP CHECKS PASSED")
finally:
    shutil.rmtree(TMP, ignore_errors=True)  # always clean up the tempdir
