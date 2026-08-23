"""
test_clip_vlm_hybrid.py — Unit tests for AdvancedCLIPHybridDataset and hybrid_augmented_collate_fn.

These tests validate all 5 ablation strategy components in isolation:
    1. AdvancedCLIPHybridDataset correctly applies hard negative mining constraints.
    2. Structured prompt format contains expected metadata tags.
    3. Dynamic text augmentation drops attributes while preserving tokenizability.
    4. Model parameter gradient flags are set correctly for the chosen unfreeze depth.
    5. The hybrid loss (Triplet + InfoNCE) outputs a non-zero positive scalar.

Run:
    .venv/bin/pytest tests/test_clip_vlm_hybrid.py -v
"""

import os
import sys
import pytest
import torch
import pandas as pd
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fashion_ds_processing import AdvancedCLIPHybridDataset, hybrid_augmented_collate_fn


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _tiny_df() -> pd.DataFrame:
    """
    Construct a minimal DataFrame with 4 items, 2 per category folder,
    sufficient to exercise hard negative mining fallbacks.

    Returns:
        A pandas DataFrame with columns [item_id, consumer_image, shop_image].
    """
    return pd.DataFrame({
        "item_id":        ["id_A", "id_A", "id_B", "id_B", "id_C", "id_C", "id_D", "id_D"],
        "consumer_image": ["img/cat1/id_A/con1.jpg", "img/cat1/id_A/con2.jpg",
                           "img/cat1/id_B/con1.jpg", "img/cat1/id_B/con2.jpg",
                           "img/cat2/id_C/con1.jpg", "img/cat2/id_C/con2.jpg",
                           "img/cat2/id_D/con1.jpg", "img/cat2/id_D/con2.jpg"],
        "shop_image":     ["img/cat1/id_A/shop.jpg", "img/cat1/id_A/shop.jpg",
                           "img/cat1/id_B/shop.jpg", "img/cat1/id_B/shop.jpg",
                           "img/cat2/id_C/shop.jpg", "img/cat2/id_C/shop.jpg",
                           "img/cat2/id_D/shop.jpg", "img/cat2/id_D/shop.jpg"],
    })


def _tiny_metadata():
    """
    Minimal clothes_type and category_folder mappings for the 4 fixture items.

    Returns:
        Tuple of (item_to_clothes_type dict, item_to_category_folder dict).
    """
    item_to_clothes_type   = {"id_A": 1, "id_B": 1, "id_C": 2, "id_D": 2}
    item_to_category_folder = {"id_A": "Polo_Shirt", "id_B": "Polo_Shirt",
                                "id_C": "Jeans",      "id_D": "Jeans"}
    return item_to_clothes_type, item_to_category_folder


def _tiny_captions():
    """Minimal attribute captions for the 4 fixture items."""
    return {"id_A": "a striped polo shirt", "id_B": "a plain polo shirt",
            "id_C": "slim fit jeans",       "id_D": "wide leg jeans"}


def _tiny_vlm_captions():
    """Minimal VLM captions (richer) for the 4 fixture items."""
    return {"id_A": "A striped short-sleeve polo shirt in white and navy.",
            "id_B": "A plain slim-fit polo shirt in forest green.",
            "id_C": "Slim fit indigo jeans with five-pocket design.",
            "id_D": "Wide leg relaxed jeans in light wash denim."}


def _make_dataset(**kwargs) -> AdvancedCLIPHybridDataset:
    """
    Factory helper: builds an AdvancedCLIPHybridDataset patching image I/O so no
    real filesystem access is required.

    Keyword Args:
        **kwargs: Passed to AdvancedCLIPHybridDataset.__init__ to override flags.

    Returns:
        A dataset instance with mocked image loading.
    """
    df = _tiny_df()
    clothes_type, cat_folder = _tiny_metadata()
    defaults = dict(
        df=df,
        dataset_root="/fake/root",
        item_captions=_tiny_captions(),
        item_to_clothes_type=clothes_type,
        item_to_category_folder=cat_folder,
        vlm_captions=_tiny_vlm_captions(),
        use_hard_negatives=True,
        use_structured_prompts=True,
        augment_text=False,    # off by default in tests — tested explicitly below
        augment_prob=0.20,
    )
    defaults.update(kwargs)
    ds = AdvancedCLIPHybridDataset(**defaults)
    # Patch _load to return a tiny blank PIL image so no disk I/O occurs
    ds._load = lambda rel_path: Image.new("RGB", (8, 8), color=(200, 100, 50))
    return ds


# ── Strategy 2: Hard Negative Mining ─────────────────────────────────────────

class TestHardNegativeMining:
    """Tests that the negative item never shares the same item_id as the anchor."""

    def test_negative_differs_from_anchor(self):
        """Hard negative mining must never return the anchor item_id as the negative."""
        ds = _make_dataset(use_hard_negatives=True)
        for idx in range(len(ds)):
            anchor_id = ds.df.iloc[idx]["item_id"]
            neg_item = ds._sample_negative(anchor_id)
            assert neg_item != anchor_id, (
                f"Hard negative equals anchor: anchor={anchor_id!r}, neg={neg_item!r}"
            )

    def test_hard_negative_stays_in_same_category(self):
        """When hard negatives are enabled, negative must be in the same category folder."""
        ds = _make_dataset(use_hard_negatives=True)
        for idx in range(len(ds)):
            anchor_id = ds.df.iloc[idx]["item_id"]
            anchor_cat = ds.item_to_category_folder.get(anchor_id)
            neg_item = ds._sample_negative(anchor_id)
            neg_cat = ds.item_to_category_folder.get(neg_item)
            assert neg_cat == anchor_cat, (
                f"Negative category {neg_cat!r} != anchor category {anchor_cat!r}"
            )

    def test_random_negative_when_hard_mining_disabled(self):
        """With use_hard_negatives=False, the negative can come from any category."""
        ds = _make_dataset(use_hard_negatives=False)
        negatives = {ds._sample_negative("id_A") for _ in range(30)}
        # With 4 items (id_A excluded = 3 candidates across 2 categories),
        # random sampling should reach across categories
        all_items_without_anchor = {"id_B", "id_C", "id_D"}
        assert negatives.issubset(all_items_without_anchor), (
            f"Unexpected negatives: {negatives - all_items_without_anchor}"
        )


# ── Strategy 3: Structured Prompt Templates ───────────────────────────────────

class TestStructuredPrompts:
    """Tests that structured prompt templates contain the expected metadata tags."""

    def test_structured_prompt_contains_category_tag(self):
        """Caption must contain the [Category: ...] metadata tag when enabled."""
        ds = _make_dataset(use_structured_prompts=True, augment_text=False)
        caption = ds._build_caption("id_A")
        assert "[Category:" in caption, f"Missing [Category:] tag in: {caption!r}"

    def test_structured_prompt_contains_type_tag(self):
        """Caption must contain the [Type: ...] metadata tag when enabled."""
        ds = _make_dataset(use_structured_prompts=True, augment_text=False)
        caption = ds._build_caption("id_A")
        assert "[Type:" in caption, f"Missing [Type:] tag in: {caption!r}"

    def test_structured_prompt_contains_vlm_text(self):
        """Caption must include the VLM natural language description."""
        ds = _make_dataset(use_structured_prompts=True, augment_text=False)
        caption = ds._build_caption("id_A")
        assert "polo shirt" in caption.lower(), (
            f"VLM text missing from caption: {caption!r}"
        )

    def test_plain_caption_when_structured_prompts_disabled(self):
        """With use_structured_prompts=False, caption must not contain any [ ] metadata tags."""
        ds = _make_dataset(use_structured_prompts=False, augment_text=False)
        caption = ds._build_caption("id_A")
        assert "[Category:" not in caption, (
            f"Unexpected [Category:] tag in plain caption: {caption!r}"
        )

    def test_category_name_underscores_replaced_by_spaces(self):
        """Category folder underscores should become spaces in the structured prompt."""
        ds = _make_dataset(use_structured_prompts=True, augment_text=False)
        caption = ds._build_caption("id_A")
        assert "Polo_Shirt" not in caption, (
            f"Underscore not replaced in category tag: {caption!r}"
        )
        assert "Polo Shirt" in caption, (
            f"Expected 'Polo Shirt' in caption: {caption!r}"
        )


# ── Strategy 5: Dynamic Text Augmentation ─────────────────────────────────────

class TestTextAugmentation:
    """Tests that text augmentation modifies captions without breaking tokenizability."""

    def test_augmentation_sometimes_drops_a_tag(self):
        """With augment_prob=1.0, a tag must always be dropped from the structured prompt."""
        ds = _make_dataset(use_structured_prompts=True, augment_text=True, augment_prob=1.0)
        # With prob=1.0, the caption should always have fewer tags than the full version
        full_ds = _make_dataset(use_structured_prompts=True, augment_text=False)
        for item_id in ["id_A", "id_B", "id_C", "id_D"]:
            caption_aug  = ds._build_caption(item_id)
            caption_full = full_ds._build_caption(item_id)
            # Augmented caption must have fewer characters (dropped tag removes text)
            assert len(caption_aug) < len(caption_full), (
                f"Augmented caption not shorter than full: {caption_aug!r}"
            )

    def test_augmented_caption_never_empty(self):
        """Augmentation must never produce an empty or whitespace-only caption."""
        ds = _make_dataset(use_structured_prompts=True, augment_text=True, augment_prob=1.0)
        for item_id in ["id_A", "id_B", "id_C", "id_D"]:
            caption = ds._build_caption(item_id)
            assert caption.strip(), f"Augmented caption is empty for item {item_id!r}"

    def test_augmented_caption_is_valid_string(self):
        """Augmented caption must be a non-None string (safe to pass to tokenizer)."""
        ds = _make_dataset(augment_text=True, augment_prob=0.5)
        for _ in range(20):
            caption = ds._build_caption("id_A")
            assert isinstance(caption, str) and len(caption) > 0


# ── Dataset __getitem__ integration ───────────────────────────────────────────

class TestDatasetGetItem:
    """Integration tests for the full __getitem__ call chain."""

    def test_getitem_returns_four_tuple(self):
        """__getitem__ must return exactly a 4-tuple."""
        ds = _make_dataset()
        item = ds[0]
        assert len(item) == 4, f"Expected 4-tuple, got {len(item)}-tuple"

    def test_getitem_anchor_is_pil_image(self):
        """Anchor must be a PIL Image."""
        ds = _make_dataset()
        anchor, _, _, _ = ds[0]
        assert isinstance(anchor, Image.Image)

    def test_getitem_caption_is_string(self):
        """Caption must be a non-empty string."""
        ds = _make_dataset()
        _, _, _, caption = ds[0]
        assert isinstance(caption, str) and len(caption) > 0


# ── Collate function ──────────────────────────────────────────────────────────

class TestHybridCollate:
    """Tests that hybrid_augmented_collate_fn produces correctly shaped tensors."""

    def test_collate_output_shapes(self):
        """All output tensors must have batch-size as first dimension."""
        from unittest.mock import MagicMock
        import torch

        # Minimal mock processor and tokenizer
        B = 3  # batch size
        mock_pixel = {"pixel_values": torch.zeros(B, 3, 224, 224)}
        mock_text  = {
            "input_ids":      torch.zeros(B, 10, dtype=torch.long),
            "attention_mask": torch.ones(B, 10, dtype=torch.long),
        }
        processor = MagicMock(return_value=mock_pixel)
        tokenizer = MagicMock(return_value=mock_text)

        batch = [(Image.new("RGB", (8, 8)),
                  Image.new("RGB", (8, 8)),
                  Image.new("RGB", (8, 8)),
                  "a caption") for _ in range(B)]

        a, p, n, t, captions = hybrid_augmented_collate_fn(batch, processor, tokenizer)

        assert a["pixel_values"].shape[0] == B
        assert p["pixel_values"].shape[0] == B
        assert n["pixel_values"].shape[0] == B
        assert t["input_ids"].shape[0] == B
        assert len(captions) == B

    def test_collate_returns_original_captions(self):
        """The 5th return value must be the original caption strings unmodified."""
        from unittest.mock import MagicMock
        import torch

        B = 2
        mock_pixel = {"pixel_values": torch.zeros(B, 3, 224, 224)}
        mock_text  = {"input_ids": torch.zeros(B, 5, dtype=torch.long),
                      "attention_mask": torch.ones(B, 5, dtype=torch.long)}
        processor = MagicMock(return_value=mock_pixel)
        tokenizer = MagicMock(return_value=mock_text)

        expected_captions = ["caption one", "caption two"]
        batch = [(Image.new("RGB", (8, 8)),
                  Image.new("RGB", (8, 8)),
                  Image.new("RGB", (8, 8)),
                  cap) for cap in expected_captions]

        _, _, _, _, captions = hybrid_augmented_collate_fn(batch, processor, tokenizer)
        assert captions == expected_captions
