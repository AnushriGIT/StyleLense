# Specification: CLIP Hybrid Fine-Tuning (clipImage&Text)

This specification defines the behavior of the data processing, caption generation, and contrastive training logic for the hybrid image-text CLIP model fine-tuning.

## Feature: Garment Name Resolution
We resolve folder names under the `img/` directory to canonical nouns using a whitelist mapping. Non-whitelisted folder names fall back to a generic noun based on `clothes_type`.

### Scenario: Resolving folder names in whitelist
  Given the whitelist mappings:
    | folder | noun |
    | Blouse | blouse |
    | Jeans | jeans |
    | T_Shirt | t-shirt |
  When `_resolve_garment_name` is called with:
    | folder | clothes_type |
    | Blouse | 1 |
    | Jeans | 2 |
  Then it should return the canonical nouns:
    | folder | expected_noun |
    | Blouse | blouse |
    | Jeans | jeans |

### Scenario: Falling back for non-whitelisted folders
  Given a non-whitelisted folder name "Chiffon"
  When `_resolve_garment_name` is called with:
    | folder | clothes_type |
    | Chiffon | 1 |
    | Chiffon | 2 |
    | Chiffon | 3 |
  Then it should return the generic noun based on `clothes_type`:
    | clothes_type | expected_noun |
    | 1            | top           |
    | 2            | bottom        |
    | 3            | outfit        |

---

## Feature: Deterministic Caption Generation
We generate captions from DeepFashion attribute labels. Positive attributes within a category type are resolved deterministically using a lowest-column-index tie-break. Optional clauses (length, sleeve) are assembled only if resolved.

### Scenario: Generating caption with all slots filled
  Given an item with:
    | slot | attribute |
    | style | distressed |
    | fabric | denim |
    | garment | jeans |
    | length | ankle-length |
    | sleeve | long sleeves |
  When `_build_item_captions` processes these attributes
  Then it should return:
    "a distressed denim jeans with ankle-length length and long sleeves"

### Scenario: Resolving tie-breaks deterministically
  Given an item has multiple positive fabric attributes:
    | attribute | column_index |
    | cotton | 50 |
    | denim | 120 |
  When the tie-break is applied
  Then it should select "cotton" (lowest column index)

### Scenario: Omitting missing slots without double spaces
  Given an item has:
    | slot | attribute |
    | garment | dress |
    | length | knee-length |
  And style, fabric, and sleeve are missing
  When `_build_item_captions` processes these attributes
  Then it should return:
    "a dress with knee-length length"

---

## Feature: Category-Constrained Hard-Negative Sampling
For a given anchor image, we sample a negative shop image from the same coarse category (`clothes_type`) but belonging to a different `item_id`.

### Scenario: Category-constrained sampling
  Given a dataset with items:
    | item_id | clothes_type | shop_image |
    | item_A | 1 | imgA.jpg |
    | item_B | 1 | imgB.jpg |
    | item_C | 2 | imgC.jpg |
  When negative sampling is performed for anchor item_A (clothes_type 1)
  Then the negative item MUST be item_B (same clothes_type, different item_id)
  And the negative item MUST NOT be item_A
  And the negative item MUST NOT be item_C

---

## Feature: Caption-Collision Masking
In InfoNCE loss, items in the same batch sharing identical captions are masked out from each other's negative sets.

### Scenario: Pairwise collision masking
  Given a batch of captions:
    | index | caption |
    | 0 | "a cotton t-shirt" |
    | 1 | "a denim jeans" |
    | 2 | "a cotton t-shirt" |
  When the collision mask is constructed
  Then `mask[0, 2]` and `mask[2, 0]` must be `True`
  And diagonal elements `mask[i, i]` must be `False`
  And all other elements must be `False`
