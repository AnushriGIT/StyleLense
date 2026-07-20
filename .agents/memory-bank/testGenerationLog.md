# Test Generation Log

Versioned record of every `test-suite-generator` invocation: what was targeted, what was covered, what was mocked, and any genuine bugs the generated tests uncovered. Each entry is append-only — never edit a past version, add a new one. Same convention as `reviewLog.md`.

---

## v1 — 2026-07-18

**Target:** "Truly everything" (user's explicit scope choice at Step 0) — `api/main.py`, `retrieval_core.py`, `fashion_ds_processing.py`, `config.py`, `scripts/evaluate_retrieval.py`, `scripts/fix_notebook_imports.py`, `scripts/make_notebook_graceful.py`.

**Invoked via:** `~/.claude/skills/test-suite-generator/SKILL.md` (global copy).

### Files created/modified

| File | Status | Covers |
|---|---|---|
| `tests/conftest.py` | new | Session-scoped fixture patching `pandas.read_csv`, `torch.hub.load`, `transformers.CLIPModel/CLIPProcessor.from_pretrained` before `retrieval_core` is ever imported — needed because that module loads real models/dataset CSV unconditionally at import time (see Findings below). |
| `tests/test_config.py` | new | Directory auto-creation, `DATASET_BASE_PATH` env-var override (regression for reviewLog v1 #4), `ALLOWED_EXTENSIONS`/`MAX_CONTENT_LENGTH` values (regression for #6/#8). |
| `tests/test_retrieval_core.py` | new | `l2_normalize`, `nearest_search` ranking + empty-gallery edge case, `_validate_gallery` (regression for reviewLog v2 fix of #10), `_clean_tag`, `_image_to_base64`. Deliberately does not test `embed_text`/`embed_image_clip`/`embed_image_dinov2` directly — would need much deeper tensor-pipeline mocking for low return. |
| `tests/test_fashion_ds_processing.py` | **merged into existing** | Added `single_item_dataframe` fixture + 4 new tests: `ValueError` raised on <2 unique `item_id`s across all 4 affected classes (regression for reviewLog v1 #14 / v4 fix). Existing happy-path tests untouched. |
| `tests/test_api.py` | new | `/health` shape (incl. regression check that `resnet_gallery` key is gone, v3); `/search` input validation (400/413/422) and 503s; the 503-not-500 regression (v1 #1); happy paths for both image and text search with mocked model calls. |
| `tests/test_evaluate_retrieval.py` | new | `recall_at_k` math; `check_regression`/`last_result_for`/history round-trip (v5 additions). |
| `tests/test_notebook_scripts.py` | new | Syntax-only `ast.parse` check for the two one-off scripts — **not** an execution smoke test (see Findings). |

### Mocking strategy
CLIP/DINOv2 model loading and the DeepFashion eval-partition CSV are mocked at import time via `conftest.py` so the whole suite runs without network, real model weights, or the real dataset present. FastAPI's `TestClient` is used for `api/main.py`; individual endpoint tests monkeypatch `retrieval_core` module attributes (`CLIP_GALLERY`, `DINOV2_MODEL`, etc.) per test case rather than re-mocking the whole module.

### Genuine bugs the suite uncovered (not fixed — surfaced per Step 5/6 rule)

1. **`retrieval_core._clean_tag` doesn't implement its own documented example.** The docstring explicitly gives `"A-line silhouette" → "A-line"` as an example, but `"silhouette"` is missing from the `stopwords` set, so the function returns `"A-line silhouette"` unchanged. `test_clean_tag_strips_stopwords[A-line silhouette-A-line]` fails and was left failing intentionally — it encodes the documented/intended behavior, not weakened to match the bug.
2. **`scripts/evaluate_retrieval.py`'s `tabulate` import is unguarded.** The script has a `try/except ImportError` around the actual `tabulate(...)` *call* for graceful fallback, but the top-level `from tabulate import tabulate` import isn't wrapped — so if the package is missing, the whole module fails to import before that fallback is ever reached, making it dead code. `tabulate` also isn't declared in `requirements.txt` at all. Not fixed (source-code + manifest changes, out of scope for this loop) — `tabulate` was installed into `.venv` only as an environment fix to unblock test collection, no source files touched.

### Testability/safety limitation surfaced (not a bug, a design constraint)
`scripts/fix_notebook_imports.py` and `scripts/make_notebook_graceful.py` both run their file-mutating logic unconditionally at module import time (no `if __name__ == "__main__":` guard) against a hardcoded absolute path pointing at the real `notebooks/03_fine_tuning.ipynb`. Actually importing or executing either script as part of a test would silently rewrite that real file as a side effect of running the test suite. Rather than work around this (would mean editing the scripts, out of scope) or risk it, `test_notebook_scripts.py` only validates syntax via `ast.parse` — a real execution-based smoke test was judged unsafe to write.

### Run result
55 passed, 1 failed (the genuine `_clean_tag` bug above, left failing intentionally) — 2 of the allowed 3 bounded passes used; stopped early once the only remaining failure was confirmed as a real bug, not something to keep iterating on.

### Status
**Not committed.** All new/modified test files are in the working tree, pending explicit go-ahead per Step 6's human checkpoint.

---

## v2 — 2026-07-18

**Target:** "All modules" — Enhancing test coverage for `api/main.py`, `retrieval_core.py`, `fashion_ds_processing.py`, and `config.py` following approval of the test suite enhancement plan.

**Invoked via:** `~/.claude/skills/test-suite-generator/SKILL.md` (global copy).

### Files created/modified

| File | Status | Covers |
|---|---|---|
| `retrieval_core.py` | modified | Added `"silhouette"` to the list of `stopwords` in `_clean_tag` to resolve the failing test. |
| `tests/test_retrieval_core.py` | modified | Added unit tests using mocks for model embedding and search helpers (`embed_text`, `embed_image_clip`, `embed_image_dinov2`, `text_search`, `_text_refine`, and `_tag_attributes`). |
| `tests/test_api.py` | modified | Added tests for GET `/` frontend routes (including missing template 404 path), middleware validation (`X-Request-ID` generation and propagation), and non-ASCII user queries. |
| `tests/test_fashion_ds_processing.py` | modified | Added `test_cached_triplet_fashion_dataset_caching_and_load_fallback` to validate pre-caching behaviors and uncached image loader fallbacks. |

### Mocking strategy
Retained the existing mock setup from `tests/conftest.py` and added explicit monkeypatches in the new unit tests for model-calling functions in `tests/test_retrieval_core.py`, utilizing `unittest.mock.MagicMock` to simulate PyTorch, Hugging Face transformers, and other ML components.

### Genuine bugs fixed
1. **`retrieval_core._clean_tag` "silhouette" stopword bug**: Added `"silhouette"` to the stopwords set so that `"A-line silhouette"` successfully reduces to `"A-line"` as documented and tested.

### Run result
All 68 tests collected and passed successfully.

### Status
**Not committed.** All new and modified files are in the working tree, pending user go-ahead.
