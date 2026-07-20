"""
Syntax-only checks for the two one-off notebook-patching scripts
(fix_notebook_imports.py, make_notebook_graceful.py).

Deliberately NOT an execution-based smoke test: both scripts run their file-mutating
logic unconditionally at module import time (no `if __name__ == "__main__":` guard)
against a hardcoded absolute NB_PATH pointing at the real notebooks/03_fine_tuning.ipynb.
Actually importing or exec'ing them would silently rewrite that real file as a side
effect of running the test suite — unsafe, and not something a test should do. This is
a testability/safety limitation surfaced here rather than worked around, since fixing it
would mean editing the scripts themselves (out of scope for a test-generation pass — see
the "never edit source/production code" rule).

This only confirms each script is syntactically valid Python, which is what "skip full
testing" as agreed in the plan review reduces to for genuinely disposable, already-run
one-off utilities.
"""
import ast
import os

import pytest

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))


@pytest.mark.parametrize("script_name", [
    "fix_notebook_imports.py",
    "make_notebook_graceful.py",
])
def test_script_is_syntactically_valid(script_name):
    path = os.path.join(SCRIPTS_DIR, script_name)
    with open(path) as f:
        source = f.read()
    ast.parse(source, filename=script_name)  # raises SyntaxError if invalid
