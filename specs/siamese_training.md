# Specification: Siamese Network Training & Re-Ranking

This specification defines the behavior of the Siamese network pair construction, model architecture with dimension projection, and the re-ranking verification logic.

---

## Feature: Category-Constrained Pair Dataset
To prevent trivial margin satisfaction, negative pairs must be constrained to share the same category type (`clothes_type`) but belong to different item IDs.

### Scenario: Category-constrained negative pair selection
  Given a set of items:
    | item_id | clothes_type | shop_image |
    | item_1  | 1            | shop1.jpg  |
    | item_2  | 1            | shop2.jpg  |
    | item_3  | 2            | shop3.jpg  |
  When `PairFashionDataset` samples a negative pair for anchor `item_1` (clothes_type 1)
  Then the negative item MUST be `item_2` (same clothes_type, different item_id)
  And it MUST NOT sample `item_3` (different clothes_type)

### Scenario: Fallback image dimensions matching backbone
  Given an image loading error occurs during batch construction
  When the fallback mechanism is triggered
  Then it must return a tensor of shape `(3, 380, 380)` to match EfficientNet-B4 input transforms
  And it MUST NOT return a mismatched shape (e.g. `(3, 224, 224)`).

---

## Feature: Shared Backbone with Dimensional Projection
The Siamese network uses a shared backbone and projects high-dimensional pooled features (1792-dim) into a stable compact representation space (256-dim) before L2 normalization.

### Scenario: Projection head dimensional reduction
  Given a batch of image inputs of shape `(batch_size, 3, 380, 380)`
  When passed through the Siamese model's forward path
  Then the backbone must produce a feature map of shape `(batch_size, 1792)`
  And the projection head must reduce the shape to `(batch_size, 256)`
  And the final embeddings returned must be L2 normalized along the dimension axis.

---

## Feature: Candidate Re-Ranking Transform Alignment
During evaluation, the re-ranker must apply the exact resolution transform used during training to prevent spatial grid mismatches.

### Scenario: Image transform matching training resolution
  Given a query image path
  When loaded for Siamese re-ranking evaluation
  Then it must be resized to `384` and cropped to `380` (EfficientNet resolution)
  And it MUST NOT be resized to `256` and cropped to `224`.
