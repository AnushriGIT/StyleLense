# CLIP Image & Text Hybrid Fine-Tuning — Reconciliation

**Status: implementation already exists.** This started as a design review of an externally-authored plan, but by the time this doc was written the hybrid model had already been built, trained, and evaluated outside this tracking flow (`fashion_ds_processing.py`'s `CLIPHybridDataset`, `specs/clip_hybrid_fine_tuning.md`, `models/clip_hybrid_best.pt`/`clip_hybrid_run.log`, and a result row in `fineTuningLog.md`, all dated 2026-07-19). This doc is therefore a **reconciliation** — what the reviewed design called for vs. what actually shipped — not a forward-looking proposal. Treat it as the durable record of that review's outcome.

## Constraints the review was scoped against
1. The existing fine-tuned CLIP (image-only, `models/clip_best/`) must be preserved as-is, never overwritten.
2. A new, separate fine-tuned CLIP (image+text hybrid) model is a distinct artifact.
3. `fineTuningLog.md` should capture all three CLIP variants (base, image-only fine-tuned, image+text fine-tuned) alongside the other existing models.
4. The hybrid model is comparison-only for now — no `retrieval_core.py`/`api/main.py` wiring. That's a later, separate human decision once all models are evaluated.

## Reviewed design vs. what's actually in the codebase

| # | Design point | Status | Notes |
|---|---|---|---|
| 1 | Category-constrained hard-negative sampling by `clothes_type` (not folder taxonomy) | ✅ Implemented | `CLIPHybridDataset.__getitem__` in `fashion_ds_processing.py:471-490`, grouped via `type_to_items` at construction. |
| 2 | Deduplicated caption generation: garment whitelist + `clothes_type` fallback, lowest-column-index tie-break | ✅ Implemented | `resolve_garment_name` + `build_item_captions` in `fashion_ds_processing.py:248-430`. Matches `specs/clip_hybrid_fine_tuning.md`'s scenarios. |
| 3 | Caption-collision masking in InfoNCE (`mask[i,j]=True` where captions collide) | ✅ Implemented | `masked_contrastive_loss` in notebook cell `ft-clip-hybrid-train`. |
| 4 | Staged training: epoch 1 contrastive-only warmup, epochs 2-5 joint loss, λ_triplet 0.0→0.5→1.0, λ_contrastive fixed 0.25 | ✅ Implemented, matches exactly | Same cell, epoch-indexed `if/else` schedule. |
| 5 | Asymmetric freezing: freeze full text transformer body, unfreeze last 2 vision ViT blocks + both projection heads | ✅ Implemented | Trainable params logged as 14.8M (9.80%) in `fineTuningLog.md`, matching the design intent (narrow unfreezing, `text_projection` stays trainable for shared-space recalibration). |
| 6 | `clothes_type` int-coercion (source file field is untyped) | ✅ Already applied | `load_item_metadata`: `clothes_type = int(tokens[1])` at `fashion_ds_processing.py:289`. |
| 7 | Distinct save path — never overwrite the image-only fine-tuned CLIP | ✅ Respected | New model saved to `models/clip_hybrid_best.pt`; `models/clip_best/` untouched. |
| 8 | No `retrieval_core.py`/`api/main.py` wiring yet (comparison-only, per explicit scope) | ✅ Respected | Confirmed via grep — no `hybrid` references in either file. |
| 9 | Reuse `evaluate_retrieval.py` / `eval_results/history.json` / `projectDocs.md` for 3-way CLIP tracking, instead of a new bespoke log | ⚠️ Partially done | A hybrid entry landed in `fineTuningLog.md` with real Recall@K, but the table skips the image-only fine-tuned CLIP row entirely (jumps baseline → hybrid). Fixed as part of this reconciliation — see below. Not run through `evaluate_retrieval.py`/`history.json`, so that history file still only has the image-only CLIP entry, not the hybrid one. |
| 10 | Persisted (not transient) caption-collision-rate diagnostic | ❌ Not implemented | No collision rate is computed or logged anywhere in the training loop. |
| 11 | Action-on-failure rule for a text-alignment secondary check | ❌ Not implemented as designed | The "Sanity check" logged per epoch (`t-shirt: norm=1.000, ...`) only checks that text embeddings are unit-norm after L2 normalization — that's true by construction and doesn't verify text-image alignment. There's no real alignment check and no defined action if one were to fail. |
| 12 | 4 new unit tests as a natural `test-suite-generator` fit | ✅ Partially covered | `tests/test_fashion_ds_processing.py::test_clip_hybrid_dataset` already exists and covers the happy path + category-constrained negative sampling. Missing: a `ValueError`-on-`<2`-items-per-`clothes_type` regression test for `CLIPHybridDataset`, matching the pattern already used for the other 4 dataset classes. |

## New findings from this reconciliation (not in the original 7 fixes)

**`hybrid_val_loader` is built but never used.** The training cell (`ft-clip-hybrid-train`) constructs a validation `DataLoader` but the training loop only ever computes `epoch_loss` from the training set and stores it into a variable named `best_val_loss` — there is no actual validation-loss computation anywhere. Since train loss happened to decrease monotonically across all 5 logged epochs (1.8328 → 0.3517 → 0.3184 → 0.2539 → 0.1805), the checkpoint saved to `models/clip_hybrid_best.pt` is functionally just "the last epoch," not genuinely "the best-on-validation" epoch the name implies. Not fixed here (out of scope for a doc-only reconciliation pass) — flagged for whoever next touches this training cell.

**Unfrozen vision-block count (2) breaks precedent set elsewhere in the same notebook, with no stated reason.** The image-only CLIP fine-tune (`ft-clip-train`) and DINOv2 fine-tune both freeze the first 9 of 12 ViT blocks and train the last 3 — DINOv2's markdown cell even says "Same ratio as CLIP" to make the precedent explicit. The hybrid cell (`ft-clip-hybrid-train`) instead unfreezes only the last 2 (`for i in [10, 11]:`), and no comment, spec scenario, or log entry anywhere explains the deviation — every other freeze-strategy choice in this notebook gets an explanatory comment; this one doesn't. Combined with the hybrid model now splitting that smaller trainable surface across two loss terms (triplet + contrastive) instead of one, this narrows the vision-side capacity available to close the domain gap on two independent axes at once, compounding rather than offsetting. See the experiment below, which isolates this one variable.

## Result, for context on the pending "should this go live" decision

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| CLIP ViT-B/32 (fine-tuned, image-only) | 16.58% | 31.59% | 38.62% | 46.53% |
| CLIP Image & Text (hybrid fine-tuned, 2-block) | 13.21% | 26.29% | 32.96% | 40.70% |
| CLIP Image & Text (hybrid fine-tuned, 3-block) | 12.57% | 25.40% | 31.78% | 39.41% |

The hybrid model currently underperforms the image-only fine-tuned CLIP on every K. Per constraint 4 above, no wiring decision is being made here — this is just the number that decision will eventually turn on.

### Controlled experiment: does matching the freeze depth (3 blocks) close the gap?

Re-ran the hybrid training with the freeze depth changed from 2 unfrozen vision blocks to 3 (matching the image-only CLIP/DINOv2 precedent — see the finding above), everything else held identical (same LR 2e-5, margin 0.2, staged λ schedule, same 5%-subsampled data, same 5 epochs). Trainable parameters went from 14.8M (9.80%) to 21.9M (14.49%), and training loss did end lower (0.1218 vs. 0.1805 final epoch) — the model fit the training objective better with the extra capacity, as expected.

**But Recall@1 got worse, not better: 13.21% → 12.57%.** Every K moved in the same direction (R@5: 26.29%→25.40%, R@10: 32.96%→31.78%, R@20: 40.70%→39.41%). This is the opposite of what the "less trainable capacity is the bottleneck" hypothesis predicted, and it's worth taking at face value rather than explaining away.

**What this actually points to:** more vision-side capacity gave the competing caption-contrastive objective more room to reshape the embedding space, and — combined with lower training loss — this looks like overfitting to the coarse, attribute-level caption clusters rather than genuinely improving instance-level separation. The captions group many visually-distinct items under identical text (that's exactly what caption-collision masking exists to handle), so a model that fits the joint loss *better* can still get *worse* at the specific instance-level retrieval task Recall@K measures — pulling same-caption images closer together works against telling those same items apart. Freeze depth was never the bottleneck; the dual-objective design itself (triplet vs. caption-alignment, at the wrong granularity for this eval) is the more likely limiter, and giving it more room to pursue both makes the tension worse, not better.

**Practical takeaway:** reverted `models/clip_hybrid_best.pt` back to the 2-block checkpoint (the better of the two) as the current "best" hybrid artifact. The 3-block variant's checkpoint/log/embeddings are kept as `models/clip_hybrid_best_3block.pt` / `clip_hybrid_run_3block.log` / `embeddings/clip_hybrid_3block_{gallery,queries}.npy` for reference, not as the active comparison artifact.

## What this reconciliation changed
- Added the missing "CLIP ViT-B/32 (fine-tuned, image-only)" row to `fineTuningLog.md` so all three CLIP variants (base, image-only fine-tuned, hybrid fine-tuned) sit in one table, per constraint 3.
- This file itself — previously the plan review had no durable home; it's now recorded here rather than in `Reference/` (which is reserved for the Pre-Commit Remediation Loop / TDD Planning Gate documentation, a different topic).

## Open items (not acted on, flagged for a future pass)
- Persisted caption-collision-rate diagnostic (#10).
- A real text-alignment secondary check with a defined failure action (#11).
- `ValueError`-on-single-item regression test for `CLIPHybridDataset` (#12), matching the other 4 dataset classes' existing coverage.
- `hybrid_val_loader` dead-code / misleading `best_val_loss` naming (new finding above).
- Hybrid model's Recall@K was recorded only in `fineTuningLog.md`, not run through `scripts/evaluate_retrieval.py`'s regression-tracking history (`eval_results/history.json`) — so it isn't covered by that gate's baseline-comparison tooling yet.
