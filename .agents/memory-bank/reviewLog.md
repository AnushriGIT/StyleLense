# Code Review Log

Versioned record of every `review-fix-loop` invocation: findings surfaced, actions taken, and what's still open. Each entry is append-only — never edit a past version, add a new one. This keeps `activeContext.md` focused on current task state rather than review history.

---

## v1 — 2026-07-14

**Target:** `api/main.py`, `retrieval_core.py`, `fashion_ds_processing.py` (whole-file review — first-ever pass, no git history existed yet to diff against).
**Invoked via:** `.claude/skills/review-fix-loop/SKILL.md`. Pass 1 reused an earlier exhaustive manual review from this same session instead of re-running a fresh 8-agent `/code-review` (cost-conscious reuse, per the skill's own guidance).

### Findings & Actions

| # | File | Finding | Class | Action |
|---|---|---|---|---|
| 1 | `api/main.py` | `/search`'s broad `except Exception` caught its own intentionally-raised `HTTPException`s, rewriting 503→500 | CONFIRMED | **Fixed** — added `except HTTPException: raise` before the generic handler |
| 2 | `api/main.py` | `TextSearchRequest.top_k` accepted explicit `null` → `nearest_search` sliced `[:None]`, returning the entire gallery | CONFIRMED | **Fixed** — `top_k: int = Field(default=10, ge=1, le=100)` |
| 3 | `config.py` | `DATASET_BASE_PATH` hardcoded to a personal absolute path — breaks portability/deployment | CONFIRMED | **Fixed** — `os.getenv("DATASET_BASE_PATH", <old value>)` |
| 4 | `api/main.py` | Hardcoded `top_k=50` cap on DINOv2 visual search regardless of requested `top_k` | CONFIRMED | **Fixed** — `top_k=max(50, top_k)` |
| 5 | `api/main.py` | No bounds validation on `top_k` anywhere (negative/zero → confusing negative-slice behavior) | CONFIRMED | **Fixed** — `ge=1, le=100` on both `/search` Form param and `TextSearchRequest` |
| 6 | `api/main.py` | `MAX_CONTENT_LENGTH` / `ALLOWED_EXTENSIONS` declared in `config.py` but never enforced | CONFIRMED | **Fixed** — extension + size checks added to `/search` upload path |
| 7 | `api/main.py` | Corrupt/non-image uploads surfaced as 500 instead of 400 | CONFIRMED | **Fixed** — `except UnidentifiedImageError` → 400 |
| 8 | `fashion_ds_processing.py` | Infinite-loop risk in negative sampling (4 dataset classes) when a subset has <2 unique `item_id`s | CONFIRMED | **Fixed** — fail-fast `ValueError` at construction in all 4 classes |
| 9 | `api/main.py` | (Self-caught in Pass 2) stale docstring said `Optional[int]` after fix #2 changed the type | CONFIRMED | **Fixed** |
| 10 | `retrieval_core.py` | No runtime assertion that gallery embedding row-counts match `GALLERY_PATHS` length (the documented dimension-mismatch "Critical Rule" has no code enforcement) | PLAUSIBLE | **Flagged, not fixed** — where/how to assert is a design choice |
| 11 | `retrieval_core.py` | Inconsistent startup fault-tolerance: ResNet + eval-partition CSV loading isn't wrapped in try/except like CLIP/DINOv2 are | PLAUSIBLE | **Flagged, not fixed** — unclear if crash-on-missing-file is intentional fail-fast behavior |
| 12 | `fashion_ds_processing.py`, `retrieval_core.py` | Silent zero-tensor/blank-image fallback with no logging (4 `_load` methods, plus `_text_refine`/`_image_to_base64`) | PLAUSIBLE | **Flagged, not fixed** — silent degradation may be intentional for training robustness |
| 13 | `retrieval_core.py` | `embed_image` (ResNet) is dead code, never called | Cleanup | **Flagged, not fixed** — deletion is a judgment call, not a bug |

### Security Gate (Step 5)

Ran manually — the skill's default `git diff origin/HEAD` failed (no remote configured on this fresh repo). Used `git diff HEAD` instead.

| Finding | Class | Action |
|---|---|---|
| New `MAX_CONTENT_LENGTH` check runs *after* `await image.read()` already buffers the full upload into memory — doesn't prevent the memory spike, only rejects after the fact | PLAUSIBLE | **Flagged, not fixed** — full fix needs streaming reads or an ASGI/reverse-proxy-level limit |
| Personal path/username remains checked into `config.py` as the fallback default for `DATASET_BASE_PATH` | Low severity | **Flagged, not fixed** — worth genericizing before the repo goes public |
| Pillow decompression-bomb guard (`Image.MAX_IMAGE_PIXELS`) | Checked | **No action needed** — default is intact, nothing disables it |

### Environment change made to enable this run
`git init` + initial commit. `.gitignore` expanded first: added `models/clip_epoch*/` (5 dirs of `model.safetensors`, ~2.9GB, previously uncovered), `.pytest_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`, `*.egg-info/`, `.env*`, `antigravity-skills/` (nested git repo — has its own `.git/`), and `.claude/settings.local.json` (personal/local config).

### Status
**Not committed.** Fixes #1–9 are applied to the working tree, awaiting explicit user go-ahead per the loop's Step 6 human checkpoint (never auto-commit).

---

## v2 — 2026-07-16

**Target:** `retrieval_core.py` — follow-up fix for v1 finding #10 (user requested it directly, out of band from a full loop invocation).

| # | File | Finding | Class | Action |
|---|---|---|---|---|
| 10 (from v1) | `retrieval_core.py` | No runtime assertion that gallery embedding row-counts match `GALLERY_PATHS` length | PLAUSIBLE (v1) | **Fixed** — added `_validate_gallery(gallery, name)` helper right after `GALLERY_PATHS` is computed; called on `RESNET_GALLERY`, `CLIP_GALLERY`, and `DINOV2_GALLERY` immediately after each is loaded. On a row-count mismatch it logs a clear message and returns `None` instead of the mismatched array — same fail-soft treatment `main.py` already expects (its existing `if utils.XXX_GALLERY is None: raise 503` checks now also catch this case), rather than crashing the app or silently corrupting search results. |

**Status:** Uncommitted, same as v1's fixes — still awaiting go-ahead to commit together.

---

## v3 — 2026-07-16

**Target:** `retrieval_core.py` + `api/main.py` — follow-up fix for v1 finding #13 (user requested removal directly).

| # | File(s) | Finding | Class | Action |
|---|---|---|---|---|
| 13 (from v1) | `retrieval_core.py` | `embed_image` (ResNet) was dead code, never called; `RESNET_MODEL`/`RESNET_GALLERY` loaded at startup but never queried | Cleanup (v1) | **Fixed** — removed the entire ResNet-50 model/weights/gallery-loading block, the `embed_image()` function, and the now-unused `torch.nn`/`torchvision.models` imports. |

**Cross-file dependency caught and handled (not scope creep — required for correctness):**
- `RESNET_TRANSFORM` was **not** ResNet-only — `embed_image_dinov2()` also depended on it for its Resize/CenterCrop/Normalize pipeline. Kept the transform, renamed it `IMAGE_TRANSFORM` to reflect actual usage, updated its one remaining call site.
- `api/main.py`'s `/health` endpoint read `utils.RESNET_GALLERY` — would have thrown `AttributeError` once the attribute was removed from `retrieval_core.py`. Removed the `resnet_gallery` block from the `/health` response.
- Verified via repo-wide grep: no remaining references to `RESNET_MODEL`/`RESNET_GALLERY`/`RESNET_TRANSFORM`/`embed_image` anywhere (`.py` files, tests). `scripts/evaluate_retrieval.py` and `scripts/make_notebook_graceful.py` still reference `resnet50` — left untouched, since those are offline benchmarking/training tools (per `projectDocs.md`'s Recall@K table, which still tracks ResNet-50 as a baseline comparison), not the live serving path.
- Both files verified with `python3 -m py_compile` — compile cleanly.

**Not touched (out of scope for this request):** `config.py`'s `MODEL_NAME = "resnet50"` and `EMBEDDING_DIM = 2048` are also unused/dead now, but the user's request was scoped to `retrieval_core.py` — flagging for awareness, not removing unprompted.

**Status:** Uncommitted — joins v1/v2 fixes, all still pending a single commit go-ahead.

---

## v4 — 2026-07-18

**Target:** `notebooks/03_fine_tuning.ipynb`, `notebooks/04_siamese.ipynb` — bug report from user (`ModuleNotFoundError: No module named 'datasets'` when running a notebook cell).

| # | File(s) | Finding | Class | Action |
|---|---|---|---|---|
| 14 | `notebooks/03_fine_tuning.ipynb` (cell `ft-imports`), `notebooks/04_siamese.ipynb` (cell `siam-imports`) | Leftover from the earlier `datasets.py` → `fashion_ds_processing.py` rename (see main task-focus log): `activeContext.md` said imports were updated in `api/main.py`, test files, and linter/test scripts — but these two notebooks were missed, so `from datasets import ...` pointed at a module that no longer exists | CONFIRMED (user-reported, reproduced from traceback) | **Fixed** — both cells updated to `from fashion_ds_processing import ...`; also fixed a stale comment in `03_fine_tuning.ipynb` referencing `datasets.py` by name. |

**Verified:** repo-wide grep confirms no remaining `from datasets import` / `import datasets` in any `.ipynb`. Both notebook JSON files still parse valid after the edit; diffs are minimal (import lines only, no reformatting).

**Noted, not fixed:** `scripts/fix_notebook_imports.py` (a one-off dev utility for patching `03_fine_tuning.ipynb`'s `sys.path`) still checks for the old `from datasets import FashionDataset` string — now stale, but it's a already-used, one-time utility script, not part of the live app; left untouched since it wasn't part of this request.

**Status:** Uncommitted — joins v1/v2/v3 fixes, all still pending a single commit go-ahead.

---

## v5 — 2026-07-18

**Target:** `scripts/evaluate_retrieval.py` — enhancement, not a bug fix. Discussed whether an "eval test suite" is needed for this project (distinct from pytest unit tests); concluded the LLM-as-judge pattern from the Day 4 Ambient Expense Agent codelab doesn't map onto a non-agentic retrieval system, but the existing Recall@K script was a reporting tool, not a real regression gate (no persistence, no baseline comparison, no pass/fail signal).

| # | File | Change | Action |
|---|---|---|---|
| 15 | `scripts/evaluate_retrieval.py` | Script printed Recall@K to stdout and exited 0 unconditionally — no way to detect a regression across runs without a human manually remembering and comparing numbers | **Enhanced** — added `load_history()`/`save_history()` (persists every run to `eval_results/history.json`, auto-created), `last_result_for()` (looks up each model's most recent prior entry), and `check_regression()` (flags any Recall@K drop beyond `--tolerance` percentage points, default 2.0). `main()` now saves results after every run and calls `sys.exit(1)` if any regression is found vs. the last recorded run for that model, `sys.exit(0)` otherwise (including first-ever runs, since there's no baseline yet). |

**New CLI flag:** `--tolerance <float>` (default `2.0`) — max acceptable Recall@K drop in percentage points before it's flagged as a regression.

**New artifact:** `eval_results/history.json` — one entry per model per run (timestamp + Recall@K values). Not gitignored; intended to be committed as the audit trail. Doesn't exist until the script is first run.

**Verified:** `python3 -m py_compile` — compiles cleanly. All existing behavior (Recall@K math, table output, CLI args) unchanged — diff is purely additive (persistence + comparison + gating), confirmed via full `git diff`.

**Status:** Uncommitted — joins v1–v4 fixes, all still pending a single commit go-ahead.
