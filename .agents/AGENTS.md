# StyleLens Project Rules & Conventions

This file enforces Vibe Engineering rules and workspace conventions for all agents operating in this workspace.

## 1. Spec-Driven BDD Checkpoints
* Before generating or modifying any code for features, you MUST formulate a technical BDD specification in `specs/` (narrative + Gherkin scenarios). 
* Do not write or execute code changes until the user has approved the specification or implementation plan.

## 2. Shift Intelligence Left
* Do not implement complex, multi-step logical operations (like caption parsing, file mappings, or mathematical masks) purely through prompt guidance or inline in Jupyter Notebooks.
* Instead, place the logic in deterministic, Pytest-tested Python helper modules (like `fashion_ds_processing.py`) and import them.

## 3. Context Budgeting & Hygiene
* Do not print verbose training loops, epoch metrics, or large logs to the standard stdout or chat interface.
* Always pipe training or evaluation outputs to a log file (e.g., `models/*.log`) and present only a concise table of key metrics (like Recall@K) in the chat.

## 4. Vibe Diff Checkpoints
* Before executing any long-running or resource-intensive tasks (e.g., training runs, large evaluation passes, database migrations), present a concise "Vibe Diff" checkpoint summarizing the exact hyperparameters, freeze strategy, and target files.

## 5. Durable Memory Primitives
* Every model fine-tuning run or retrieval evaluation MUST be recorded in a durable log under `.agents/memory-bank/` (e.g., `fineTuningLog.md` or `evaluationLog.md`). Include the exact commit hash, hyperparameters used, and Recall@K scores.

## 6. Environment & Path Safety
* **Virtual Environment**: Always use the local virtual environment interpreter and test runner located at `.venv/bin/python` and `.venv/bin/pytest`. Do not use system-level python commands.
* **Path Resolution**: Never hardcode dataset base paths. Always import paths from `config.py` and resolve them dynamically. Use `os.path.join` for cross-platform compatibility.

## 7. Document Code Logic
* Any code added or modified (scripts, datasets, helper functions, wrappers) MUST contain thorough inline comments, method-level docstrings, and file-level docstrings explaining the technical logic, inputs, outputs, and design trade-offs.

