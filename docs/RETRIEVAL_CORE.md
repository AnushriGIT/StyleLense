# StyleLens — Retrieval Core Specification Document

> Scope: **Layer 1 only** — the deep-learning clothing image retrieval system.
> This is the base specification for a **new, standalone project folder**,
> deliberately separated from the multi-agent retail assistant (MyStoreHelp)
> that currently lives alongside it in `fashion_assistant/agents/`.
>
> Written to bootstrap a fresh repo — assume nothing exists yet except this
> document and the raw DeepFashion dataset.

---

## 1. Purpose & Boundary

**This project does exactly one thing well:** given a clothing image (and
optionally a text description), return the most visually/semantically similar
items from a catalogue, ranked by similarity — with the model choices and
evaluation methodology to justify *why* a given retrieval approach works.

**Explicitly out of scope** (belongs to a separate downstream project that will
*consume* this one as a library/service, not live inside it):

- Authentication, JWT, roles/permissions
- LangGraph / multi-agent orchestration
- Personal Stylist, Find-in-Store, Auto Return, Trend Forecasting, Visual
  Inventory Auditor, or any retail-workflow logic
- CRM/order/loyalty data, MCP server integration
- Conversational LLM responses, session memory, chat history

Those all depend on this project's output (an embedding + search API) but do
not belong in its codebase. Keeping the boundary here is what makes both
projects independently maintainable — this one can be re-evaluated, re-trained,
or swapped to a new backbone without touching agent logic, and the agent
project can evolve its retail workflows without touching model code.

## 2. Business & Technical Problem

Fashion imagery search is hard to do by text alone. This project builds the
retrieval core: turn a clothing photo into a vector representation, and find
its nearest neighbours in a catalogue of clothing images, fast enough for
interactive use.

The harder version of the problem — and the one this project targets — is
**cross-domain retrieval**: the query photo (consumer-taken, arbitrary
lighting/background/pose) is not the same domain as the catalogue photo
(studio shop image). A model that only matches shop-to-shop images doesn't
prove much; matching a phone photo to the correct studio shot does.

## 3. Dataset

**DeepFashion — Consumer-to-shop Clothes Retrieval Benchmark.**

```
<DATASET_ROOT>/
├── img/                                  # consumer + shop images, organized by category/item ID
├── Eval/
│   └── list_eval_partition.txt           # columns: consumer_image, shop_image, item_id, split
└── Anno/
    ├── list_item_consumer2shop.txt       # item ID ↔ image path mapping
    ├── list_attr_cloth.txt               # attribute vocabulary (names)
    ├── list_attr_items.txt               # per-item attribute labels
    └── list_bbox_consumer2shop.txt       # bounding boxes (x1, y1, x2, y2) per image
```

- `split` values: `train` (used for fine-tuning) and the evaluation split (test) —
  each item has one or more consumer photos and one or more matching shop photos.
- Evaluation protocol: **for each consumer (query) photo in the test split,
  the system must rank the matching shop (gallery) photo as close to rank 1 as
  possible**, against a gallery built from *all* unique shop images in the test
  split.
- EDA must establish, before any modeling: split sizes, images-per-item
  distribution, category long-tail shape, image dimension variance, attribute
  coverage, and — most importantly — a qualitative look at the **domain gap**
  between consumer and shop photos of the same item (lighting, crop, pose,
  background clutter). This shapes every downstream preprocessing decision
  (e.g. whether bbox-cropping the subject out of a cluttered consumer photo is
  worth the added pipeline complexity).

## 4. Model Architecture Options

The specification requires **comparing multiple embedding backbones**, not
committing to one up front — the grading rubric rewards a documented
experimentation process, and the domain-gap problem means backbone choice
matters more here than in a same-domain retrieval task.

| Backbone | Role | Notes |
|---|---|---|
| **ResNet-50** | Baseline | Pretrained on ImageNet, classification head removed, 2048-dim output. Zero fine-tuning to start — establishes the "does transfer learning alone work" floor. |
| **EfficientNet-B4** | Baseline alternative | Different inductive bias/receptive field than ResNet — worth comparing head-to-head on the same Recall@K harness. |
| **CLIP (ViT backbone)** | Primary — multi-modal | 512-dim, image and text share one embedding space. This is what unlocks a text-query path (search by description) without a separate cross-modal model. Evaluate both zero-shot and fine-tuned. |
| **DINOv2** | Secondary comparison | Self-supervised backbone; worth including in the comparison table for completeness even if not chosen as production. |
| **Siamese network (contrastive loss)** | Re-ranker or standalone | Two modes worth testing separately: (a) as its own embedding model trained end-to-end on consumer/shop pairs, (b) as a re-ranker applied only to another model's top-N candidates — cheaper to train, potentially most of the benefit. |

### 4.1 Fine-tuning strategy

- **Triplet loss** (anchor = consumer photo, positive = matching shop photo,
  negative = a shop photo from a different item) for ResNet-50 / EfficientNet /
  CLIP.
- **Contrastive loss** (binary same/different label) for the Siamese variant.
- Document the **freeze-layer strategy** explicitly — which backbone layers are
  frozen vs. fine-tuned, and why (this is a required rubric item: "specify any
  challenges encountered during fine-tuning and the strategies employed").

### 4.2 Dataset classes needed (data pipeline)

Kept as plain Python modules (not notebook cells) so `DataLoader` workers can
pickle them for multiprocessing:

| Class | Purpose |
|---|---|
| `FashionDataset` | Single-image loader — `(transformed_image, index)`. Used for bulk embedding extraction. |
| `TripletFashionDataset` | `(anchor, positive, negative)` for triplet-loss fine-tuning, sampled from the train split. |
| `CachedTripletFashionDataset` | Same, but preloads every unique image into RAM at init — use when training is disk-I/O bound. |
| `PairFashionDataset` | `(img1, img2, label)` same/different pairs for the Siamese contrastive-loss model. |
| `CLIPTripletDataset` | Returns raw PIL images (not tensors) so the CLIP processor's own preprocessing runs in the main process — workers only do disk I/O. |

### 4.3 Preprocessing & Color Format Standards

All input images must be normalized consistently before extraction or training. The standard ImageNet normalization transform is:
$$\text{Normalize}(\text{mean}=[0.485, 0.456, 0.406], \text{std}=[0.229, 0.224, 0.225])$$

* **Why Mean/Std values differ:** These constants represent the color distribution of the 1.2 million images in ImageNet. The channels differ because natural images are dominated by warm subject tones/earth colors (red is highest at `0.485`) and foliage/landscapes (green is `0.456`), whereas sky and water blues (blue is lowest at `0.406`) are less prevalent in the classified objects.
* **Why explicit RGB conversion (`Image.open(path).convert("RGB")`) is required:** Real-world datasets (and user uploads in production) contain formats other than 3-channel RGB: Grayscale (`L` mode with 1 channel), transparent PNGs (`RGBA` with 4 channels), or print files (`CMYK` with 4 channels). If an image is loaded in any of these non-RGB formats, the PyTorch DataLoader will yield tensors with mismatched dimensions (e.g. `(1, H, W)` or `(4, H, W)`), causing immediate model crashes during inference or training. Explicit `.convert("RGB")` behaves as a safe defensive shield.
* **Dataset Scan Verification:** A random sample of 1,000 images from the DeepFashion benchmark was analyzed to verify actual format usage:
  * **`RGB` (True Color):** 998 images (99.8%)
  * **`L` (Grayscale):** 2 images (0.2%)
  * **Impact:** Although 99.8% of the images are already standard RGB (making the conversion a cheap, instantaneous no-op), the 0.2% grayscale rate indicates that approximately **470 images** in the full DeepFashion corpus (239,557 total images) would cause runtime crashes if not defensively converted.

## 5. Embedding Generation & Storage

- Batch-extract embeddings for the full gallery (and, separately, for the
  query set used in evaluation) once per model variant; cache to disk so
  re-runs don't recompute.
- **L2-normalize** every embedding at extraction time — this is what makes
  inner product equivalent to cosine similarity downstream, avoiding a
  normalization step at every query.
- **Pre-allocate storage matrix (`np.zeros`)**: For optimal memory allocation performance, pre-allocate the entire embeddings matrix as `np.zeros((len(image_paths), embedding_dim), dtype=np.float32)`. This reserves a contiguous chunk of RAM up front, eliminating dynamic array resizing overhead during iteration and ensuring strict indexing alignment with image paths.
- Naming convention (one pair of files per model variant, so multiple
  backbones can be compared side by side without overwriting each other):
  ```
  embeddings/
  ├── {model_name}_gallery.npy          # e.g. clip_gallery.npy, trained_resnet50_gallery.npy
  ├── {model_name}_queries.npy          # matching query-set embeddings, same order as the eval partition
  └── image_paths.npy                   # shared path index, order-aligned with gallery embeddings
  ```
- Configurable batch size (default 64; document that it should drop to 16–32
  under memory pressure) and an optional `MAX_IMAGES` cap for fast iteration
  during development.

## 6. Similarity Search

- Cosine similarity via inner product on L2-normalized vectors.
- Brute-force (`query_emb @ gallery_emb.T`) is sufficient at DeepFashion scale
  and keeps the implementation simple and exact — note in the spec that an
  approximate index (FAISS `IndexIVFFlat` or similar) is a documented,
  **optional** upgrade path if the gallery grows large enough that brute-force
  latency becomes a problem, not a day-one requirement.
- Support two query modes from the start, since the CLIP backbone makes both
  possible with no extra modeling: **image similarity search** and **text
  search** (encode a text description with the same CLIP text tower, search
  the same gallery).

## 7. Evaluation Methodology

**Recall@K** is the required metric, computed against the DeepFashion test
partition:

1. Build the gallery from all unique shop images in the test split.
2. For each consumer (query) image, rank all gallery images by cosine
   similarity to the query embedding.
3. `Recall@K` = the percentage of queries where the ground-truth matching
   shop image appears within the top K ranked results.
4. Report at multiple K values (e.g. 1, 5, 10, 20) and compare **all trained
   model variants side by side in one table** — this comparison, not any
   single number, is the actual deliverable, since the rubric explicitly asks
   for "articulating the selection process for the optimal model."
5. Sanity-check embedding counts against the partition before scoring (a
   silent shape mismatch produces meaningless results).

## 8. Serving Layer (Minimal)

This project's API surface should be **only** what's needed to prove the
retrieval core works end-to-end and to give a downstream project (or a grader)
something to call — no auth, no roles, no conversational layer.

| Endpoint | Purpose |
|---|---|
| `POST /search` | Accepts an image (and optionally text); returns top-K similar items with similarity scores |
| `POST /text-search` | Accepts a text description only; returns top-K similar items |
| `GET /health` | Liveness check |

A minimal single-page frontend (upload an image, see results) is a reasonable
deliverable to demonstrate the system, but should not grow conversational or
role-aware features — that belongs in the consuming project.

## 9. Configuration

A single `config.py` (or `.env` + a typed settings module) covering:

- `DATASET_BASE_PATH` and derived subpaths (img/Eval/Anno)
- Embedding/model settings: which backbone is "production," embedding dim, image size
- `EMBEDDINGS_DIR` and file naming per Section 5
- `BATCH_SIZE`, `MAX_IMAGES` (dev-speed knobs)
- `TOP_K` (default search result count)
- Upload constraints (max size, allowed extensions) — only if the serving layer is included

No database URL, no JWT secret, no MCP config — none of that belongs to this project.

## 10. Proposed Project Structure

```
stylelens-retrieval/
├── config.py
├── requirements.txt
├── setup_check.py                    # environment + dataset path verification
├── fashion_ds_processing.py           # FashionDataset, TripletFashionDataset, CachedTripletFashionDataset,
│                                      # PairFashionDataset, CLIPTripletDataset
├── retrieval_core.py                 # shared embed_image / embed_text / nearest_search helpers
├── notebooks/
│   ├── 01_eda.ipynb                  # dataset exploration, domain-gap analysis
│   ├── 02_model_experiments.ipynb    # ResNet-50, EfficientNet-B4, CLIP zero-shot, DINOv2 — baseline Recall@K
│   ├── 03_fine_tuning.ipynb          # triplet-loss fine-tuning, freeze-layer experiments
│   └── 04_siamese.ipynb              # contrastive-loss Siamese, standalone + re-ranker modes
├── scripts/
│   ├── extract_embeddings.py         # batch embedding extraction per model variant → embeddings/
│   └── evaluate_retrieval.py         # Recall@K across all trained model variants, tabulated comparison
├── embeddings/                       # generated — {model_name}_gallery.npy, {model_name}_queries.npy, image_paths.npy
├── models/                           # generated — trained model checkpoints per backbone
├── api/
│   ├── main.py                       # FastAPI app: /search, /text-search, /health
│   └── static/, templates/           # minimal demo frontend
├── docs/
│   ├── RETRIEVAL_CORE.md                 # this specification document
│   └── WEB_ARCHITECTURE_NOTES.md         # reference notes for serving layer architecture
├── data/                             # DeepFashion dataset (gitignored)
├── README.md                         # setup + how to run each stage
```

## 11. Deliverables Mapping (Grading Rubric)

| Rubric Area | Weight | Where it lives here |
|---|---|---|
| EDA & Data Preparation | 30% | `notebooks/01_eda.ipynb`, `fashion_ds_processing.py` |
| Model Training, Tuning & Evaluation | 30% | `notebooks/02_model_experiments.ipynb`, `03_fine_tuning.ipynb`, `04_siamese.ipynb`, `scripts/evaluate_retrieval.py` |
| Creativity & Innovation | 25% | Multi-backbone comparison, multi-modal (image+text) search, Siamese re-ranking mode |
| Solution Documentation | 15% | `README.md`, this specification, docstrings |

## 12. Non-Functional Requirements

- Embedding extraction and training must run on CPU-only hardware in a
  degraded-but-functional mode (the original brief explicitly allows non-GPU
  submissions) — batch size and `MAX_IMAGES` exist specifically to make this
  practical.
- All embeddings must be L2-normalized before persisting, so every downstream
  consumer can assume inner product = cosine similarity without re-deriving it.
- Model comparison must be reproducible from cached embeddings alone — the
  evaluation script should not require re-running any model forward pass.

## 13. What Moves Where (Migration Note)

When splitting the current `fashion_assistant/` repo into this standalone
project:

| Stays in this new retrieval-core project | Moves to / stays in the agent project |
|---|---|
| `config.py` (dataset/model/embedding settings only) | `config.py` (DB URL, JWT, MCP settings) |
| `datasets.py` | — |
| `eda.ipynb`, `model_experiments.ipynb`, `fine_tuning.ipynb`, `siamese.ipynb` | — |
| `evaluate_retrieval.py` | — |
| `embeddings/`, `models/` | — |
| CLIP embed/search helpers currently in `agents/utils.py` → becomes `retrieval_core.py` here | `agents/utils.py`'s gallery-cache usage stays as a *dependency* the agent project imports/calls, not owns |
| A new minimal `/search`, `/text-search`, `/health` API | `api.py` (auth, session history, `/agent/*` endpoints), `auth.py`, all of `agents/` |
| — | `mystorehelp-mcp/` (unrelated to retrieval — CRM/orders/loyalty backend) |
