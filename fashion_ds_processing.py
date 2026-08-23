"""
fashion_ds_processing.py — PyTorch Dataset classes for DeepFashion training.

Defined here (not in notebooks) so DataLoader worker processes can import
them via pickling — required for num_workers > 0 on macOS.

Classes:
  FashionDataset              — single-image loader for embedding extraction
  TripletFashionDataset       — anchor / positive / negative triplets (fine_tuning.ipynb)
  CachedTripletFashionDataset — same as above but pre-loads all images into RAM (fast training)
  PairFashionDataset          — same/different pairs with binary label (siamese.ipynb)
  CLIPTripletDataset          — PIL-image triplets for CLIP fine-tuning (fine_tuning.ipynb)
  CLIPShopCaptionDataset      — shop image + VLM caption pairs for InfoNCE training
  AdvancedCLIPHybridDataset   — ablation-study dataset with hard negatives, structured prompts,
                                and dynamic text augmentation (train_clip_vlm_hybrid.py)

Functions:
  hybrid_augmented_collate_fn — custom DataLoader collate for AdvancedCLIPHybridDataset
  build_item_captions         — builds attribute-based caption dict from DeepFashion annotations
  load_item_metadata          — parses clothes_type and category_folder per item_id
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class FashionDataset(Dataset):
    """Single-image loader. Returns (transformed_image, index)."""

    def __init__(self, image_paths, dataset_root, transform):
        self.image_paths  = image_paths
        self.dataset_root = dataset_root
        self.transform    = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.dataset_root, self.image_paths[idx])
        try:
            return self.transform(Image.open(path).convert("RGB")), idx
        except Exception:
            return torch.zeros(3, 224, 224), idx


class TripletFashionDataset(Dataset):
    """
    Returns (anchor, positive, negative) image triplets for triplet margin loss.

    anchor   = consumer photo
    positive = matching shop photo (same item_id)
    negative = random shop photo from a different item_id
    """

    def __init__(self, df_train, dataset_root, transform):
        self.df            = df_train.reset_index(drop=True)
        self.dataset_root  = dataset_root
        self.transform     = transform
        self.item_to_shops = df_train.groupby('item_id')['shop_image'].unique().to_dict()
        self.all_items     = list(self.item_to_shops.keys())
        if len(self.all_items) < 2:
            raise ValueError(
                f"TripletFashionDataset needs at least 2 unique item_ids to sample negatives, got {len(self.all_items)}."
            )

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        full_path = os.path.join(self.dataset_root, path)
        try:
            return self.transform(Image.open(full_path).convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        anchor_img   = self._load(row['consumer_image'])
        positive_img = self._load(row['shop_image'])

        neg_item = row['item_id']
        while neg_item == row['item_id']:
            neg_item = np.random.choice(self.all_items)
        negative_img = self._load(np.random.choice(self.item_to_shops[neg_item]))

        return anchor_img, positive_img, negative_img


class CachedTripletFashionDataset(Dataset):
    """
    Same triplet logic as TripletFashionDataset but loads every unique image
    into RAM once at init time. Eliminates all disk I/O during training —
    much faster when the subset fits in memory (~2-4 GB for 5% of DeepFashion).

    Use this instead of TripletFashionDataset when training is bottlenecked
    by disk reads (symptom: >10s per batch with num_workers=0).
    """

    def __init__(self, df_train, dataset_root, transform):
        from tqdm import tqdm
        self.df            = df_train.reset_index(drop=True)
        self.transform     = transform
        self.item_to_shops = df_train.groupby('item_id')['shop_image'].unique().to_dict()
        self.all_items     = list(self.item_to_shops.keys())
        if len(self.all_items) < 2:
            raise ValueError(
                f"CachedTripletFashionDataset needs at least 2 unique item_ids to sample negatives, got {len(self.all_items)}."
            )

        # Collect every unique image path that could be sampled
        all_paths = set(self.df['consumer_image'].tolist() + self.df['shop_image'].tolist())
        for paths in self.item_to_shops.values():
            all_paths.update(paths)

        print(f"Pre-caching {len(all_paths)} unique images into RAM...")
        self.cache = {}
        for path in tqdm(all_paths, desc="Caching images"):
            full = os.path.join(dataset_root, path)
            try:
                self.cache[path] = Image.open(full).convert("RGB")
            except Exception:
                self.cache[path] = Image.new("RGB", (224, 224))
        print("Cache ready.")

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        try:
            return self.transform(self.cache[path])
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        anchor_img   = self._load(row['consumer_image'])
        positive_img = self._load(row['shop_image'])

        neg_item = row['item_id']
        while neg_item == row['item_id']:
            neg_item = np.random.choice(self.all_items)
        negative_img = self._load(np.random.choice(self.item_to_shops[neg_item]))

        return anchor_img, positive_img, negative_img


class PairFashionDataset(Dataset):
    """
    Returns (img1, img2, label) pairs for contrastive loss.

    label = 1.0 → same item  (consumer + matching shop photo)
    label = 0.0 → different items (random negative)

    First half of indices → positive pairs
    Second half           → negative pairs
    Total length = 2 × len(df)
    """

    def __init__(self, df_train, dataset_root, transform):
        """
        Initializes the PairFashionDataset.

        Args:
            df_train (pd.DataFrame): Dataframe containing image paths and item IDs.
            dataset_root (str): Absolute path to the dataset root folder.
            transform (callable): Image transforms to apply.
        """
        self.df            = df_train.reset_index(drop=True)
        self.dataset_root  = dataset_root
        self.transform     = transform
        self.item_to_shops = df_train.groupby('item_id')['shop_image'].unique().to_dict()
        self.all_items     = list(self.item_to_shops.keys())
        if len(self.all_items) < 2:
            raise ValueError(
                f"PairFashionDataset needs at least 2 unique item_ids to sample negatives, got {len(self.all_items)}."
            )
        self.length        = len(self.df) * 2
        
        # Load item category metadata for constrained negative mining
        try:
            self.item_to_clothes_type, _ = load_item_metadata(self.dataset_root)
        except Exception:
            self.item_to_clothes_type = {}
        
        # Group items by clothes_type to support fast same-category sampling
        self.type_to_items = {}
        for item_id in self.all_items:
            ctype = self.item_to_clothes_type.get(item_id, 1)  # Default fallback ctype=1
            self.type_to_items.setdefault(ctype, []).append(item_id)


    def __len__(self):
        """Returns the total number of pair samples (positives + negatives)."""
        return self.length

    def _load(self, path):
        """Loads and transforms a single image with dynamic fallback shape matching on exception."""
        try:
            return self.transform(Image.open(os.path.join(self.dataset_root, path)).convert("RGB"))
        except Exception:
            # Dynamically infer the exact output shape of the transform using a dummy image
            try:
                dummy_img = Image.new("RGB", (500, 500))
                dummy_tensor = self.transform(dummy_img)
                return torch.zeros_like(dummy_tensor)
            except Exception:
                # Fallback to standard EfficientNet shape if inference fails
                return torch.zeros(3, 380, 380)


    def __getitem__(self, idx):
        """
        Retrieves a pair sample. First half are positive (label=1), second half are negative (label=0).
        """
        row   = self.df.iloc[idx % len(self.df)]
        img1  = self._load(row['consumer_image'])

        if idx < len(self.df):
            # Positive pair: matching shop image of the same item ID
            img2  = self._load(row['shop_image'])
            label = torch.tensor(1.0)
        else:
            # Negative pair: category-constrained hard-negative shop image
            anchor_item = row['item_id']
            anchor_type = self.item_to_clothes_type.get(anchor_item, 1)
            
            # Retrieve items sharing the same category (clothes_type)
            candidates = self.type_to_items.get(anchor_type, [anchor_item])
            filtered = [it for it in candidates if it != anchor_item]
            
            if not filtered:
                # Fallback to any different item if no other item exists in the same category
                filtered = [it for it in self.all_items if it != anchor_item]
                
            neg_item = np.random.choice(filtered)
            img2  = self._load(np.random.choice(self.item_to_shops[neg_item]))
            label = torch.tensor(0.0)

        return img1, img2, label



class CLIPTripletDataset(Dataset):
    """
    Returns (anchor_pil, positive_pil, negative_pil) PIL images for CLIP fine-tuning.

    Returns raw PIL images (not tensors) so the clip_processor collate_fn
    can run in the main process. Workers only do disk I/O — safe to pickle.
    """

    def __init__(self, df, dataset_root):
        self.df            = df.reset_index(drop=True)
        self.dataset_root  = dataset_root
        self.item_to_shops = df.groupby('item_id')['shop_image'].unique().to_dict()
        self.all_items     = list(self.item_to_shops.keys())
        if len(self.all_items) < 2:
            raise ValueError(
                f"CLIPTripletDataset needs at least 2 unique item_ids to sample negatives, got {len(self.all_items)}."
            )

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        try:
            return Image.open(os.path.join(self.dataset_root, path)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        anchor = self._load(row['consumer_image'])
        pos    = self._load(row['shop_image'])

        neg_item = row['item_id']
        while neg_item == row['item_id']:
            neg_item = np.random.choice(self.all_items)
        return anchor, pos, self._load(np.random.choice(self.item_to_shops[neg_item]))


class CLIPShopCaptionDataset(Dataset):
    """
    Returns (shop_image_pil, caption_text) pairs for CLIP text-image fine-tuning.
    """

    def __init__(self, df, dataset_root, captions_file=None):
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        
        # Load captions from json
        if captions_file is None:
            # We assume shop_captions.json is in embeddings folder relative to the root project dir
            captions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings', 'shop_captions.json')
            
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        try:
            return Image.open(os.path.join(self.dataset_root, path)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = row['item_id']
        shop_img = self._load(row['shop_image'])
        
        caption = self.captions.get(item_id, "a piece of clothing")
        
        return shop_img, caption


# ── Whitelist mapping of garment folder names to canonical nouns ───────────────
GARMENT_FOLDER_MAP = {
    "Blouse": "blouse",
    "Coat": "coat",
    "Jeans": "jeans",
    "Pants": "pants",
    "Polo_Shirt": "polo shirt",
    "T_Shirt": "t-shirt",
    "Tank_Top": "tank top",
    "Dress": "dress",
    "Lace_Dress": "dress",
    "Skirt": "skirt",
    "Sleeveless_Dress": "dress",
    "Suspenders_Skirt": "skirt",
    "Leggings": "leggings"
}


def resolve_garment_name(folder_name: str, clothes_type: int) -> str:
    """
    Resolves folder name under img/ to a canonical noun using a whitelist.
    Falls back to a generic noun based on clothes_type if not whitelisted.
    """
    if folder_name in GARMENT_FOLDER_MAP:
        return GARMENT_FOLDER_MAP[folder_name]
    
    # Fallback based on clothes_type: 1 -> top, 2 -> bottom, 3 -> outfit
    if clothes_type == 1:
        return "top"
    elif clothes_type == 2:
        return "bottom"
    elif clothes_type == 3:
        return "outfit"
    return "clothing"


def load_item_metadata(dataset_root: str):
    """
    Parses list_bbox_consumer2shop.txt to extract item_id -> clothes_type (int)
    and item_id -> category_folder (str).
    """
    anno_dir = os.path.join(dataset_root, "Anno")
    bbox_file = os.path.join(anno_dir, "list_bbox_consumer2shop.txt")
    
    item_to_clothes_type = {}
    item_to_category_folder = {}
    
    with open(bbox_file, "r", encoding="utf-8-sig") as f:
        lines_iter = iter(f)
        try:
            next(lines_iter)  # skip count
            next(lines_iter)  # skip header
        except StopIteration:
            pass
            
        for line in lines_iter:
            tokens = line.strip().split()
            if len(tokens) >= 2:
                img_path = tokens[0]  # e.g. img/CLOTHING/Polo_Shirt/id_00006282/shop_01.jpg
                clothes_type = int(tokens[1])
                
                path_parts = img_path.split('/')
                if len(path_parts) >= 4:
                    category_folder = path_parts[2]
                    item_id = path_parts[3]
                    
                    item_to_clothes_type[item_id] = clothes_type
                    item_to_category_folder[item_id] = category_folder
                    
    return item_to_clothes_type, item_to_category_folder


def build_item_captions(dataset_root: str) -> dict:
    """
    Builds a dictionary of {item_id: caption} for all items in the dataset.
    """
    import re
    anno_dir = os.path.join(dataset_root, "Anno")
    
    # 1. Load item metadata (clothes_type and category folder)
    item_to_clothes_type, item_to_category_folder = load_item_metadata(dataset_root)
    
    # 2. Load attribute vocabulary
    cloth_attr_file = os.path.join(anno_dir, "list_attr_cloth.txt")
    with open(cloth_attr_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    num_attrs = int(lines[0].strip())
    vocab = []
    for line in lines[2:2+num_attrs]:
        tokens = re.split(r'\s{2,}', line.strip())
        if len(tokens) >= 3:
            eng_name = tokens[1]
            attr_type = int(tokens[2])
            vocab.append((eng_name, attr_type))
        else:
            parts = line.strip().split()
            eng_name = " ".join(parts[1:-1])
            attr_type = int(parts[-1])
            vocab.append((eng_name, attr_type))
            
    # Load attribute type names
    type_file = os.path.join(anno_dir, "list_attr_type.txt")
    with open(type_file, "r", encoding="utf-8-sig") as f:
        type_lines = f.readlines()
    type_names = {}
    for i, line in enumerate(type_lines[2:]):
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) >= 2:
            type_names[i+1] = parts[1]
            
    # 3. Read attribute labels per item
    items_attr_file = os.path.join(anno_dir, "list_attr_items.txt")
    item_captions = {}
    
    with open(items_attr_file, "r", encoding="utf-8-sig") as f:
        lines_iter = iter(f)
        try:
            next(lines_iter)  # skip count
            next(lines_iter)  # skip header
        except StopIteration:
            pass
            
        for line in lines_iter:
            parts = line.strip().split()
            if not parts:
                continue
            item_id = parts[0]
            labels = [int(x) for x in parts[1:]]
            
            # Group active attributes by type
            grouped = {}
            for idx, val in enumerate(labels):
                if val == 1:
                    eng_name, attr_type = vocab[idx]
                    type_name = type_names.get(attr_type, f"type-{attr_type}")
                    grouped.setdefault(type_name, []).append((eng_name, idx))
                    
            # 4. Tie-break: Select attribute with lowest index (idx) for each category
            def clean_attr(val):
                if not val:
                    return ""
                val = val.lower()
                for stop in ["length", "clothes", "fabric", "material", "clothing", "sleeves", "sleeve"]:
                    val = val.replace(stop, "").strip()
                return " ".join(val.split())
                
            style = ""
            if "style of clothes" in grouped:
                style_attr = min(grouped["style of clothes"], key=lambda x: x[1])[0]
                style = clean_attr(style_attr)
                
            fabric = ""
            if "fabric of clothes" in grouped:
                fabric_attr = min(grouped["fabric of clothes"], key=lambda x: x[1])[0]
                fabric = clean_attr(fabric_attr)
                
            length_attrs = []
            for k, v in grouped.items():
                if "length" in k.lower():
                    length_attrs.extend(v)
            length = ""
            if length_attrs:
                length_attr = min(length_attrs, key=lambda x: x[1])[0]
                length = clean_attr(length_attr)
                
            sleeve_attrs = []
            for k, v in grouped.items():
                if "sleeve" in k.lower():
                    sleeve_attrs.extend(v)
            sleeve = ""
            if sleeve_attrs:
                sleeve_attr = min(sleeve_attrs, key=lambda x: x[1])[0]
                sleeve = clean_attr(sleeve_attr)
                
            # 5. Resolve garment name
            folder = item_to_category_folder.get(item_id, "")
            clothes_type = item_to_clothes_type.get(item_id, 1)
            garment = resolve_garment_name(folder, clothes_type)
            
            # 6. Clause assembly
            words = ["a"]
            if style:
                words.append(style)
            if fabric:
                words.append(fabric)
            words.append(garment)
            
            extra = []
            if length:
                extra.append(f"{length} length")
            if sleeve:
                extra.append(f"{sleeve} sleeves")
                
            if extra:
                caption = " ".join(words) + " with " + " and ".join(extra)
            else:
                caption = " ".join(words)
                
            item_captions[item_id] = caption
            
    return item_captions


class AdvancedCLIPHybridDataset(Dataset):
    """
    Advanced CLIP dataset for ablation study of 5 fine-tuning strategies.

    This dataset supports toggling each strategy independently so we can
    measure the isolated contribution of each technique to Recall@K:

    Strategy 2 — Hard Negative Mining (``use_hard_negatives``):
        Constrains negative item sampling to the same *category folder*
        (e.g., ``Polo_Shirt``) rather than just the same clothes_type (top/
        bottom/outfit). This forces the model to learn fine-grained intra-
        category discrimination (e.g., distinguish two different polo shirts).

    Strategy 3 — Structured Prompt Templates (``use_structured_prompts``):
        Prepends DeepFashion metadata tags to VLM caption strings:
        "[Category: Polo_Shirt] [Type: top] [Attributes: ...] VLM caption..."
        Gives the text encoder explicit, deterministic anchors in addition to
        the natural language description.

    Strategy 5 — Dynamic Text Augmentation (``augment_text``):
        With probability ``augment_prob``, randomly drops or permutes
        1–2 structured metadata tags during ``__getitem__``. This simulates
        incomplete user search queries (e.g., just "a red dress") and makes
        the text encoder robust to partial input at inference time.

    Args:
        df: DataFrame with columns [item_id, consumer_image, shop_image].
        dataset_root: Absolute path to the DeepFashion dataset root.
        item_captions: Dict mapping item_id -> VLM-generated caption string.
        item_to_clothes_type: Dict mapping item_id -> int (1=top, 2=bottom, 3=outfit).
        item_to_category_folder: Dict mapping item_id -> category folder name.
        vlm_captions: Optional dict of item_id -> richer VLM description from
            ``shop_captions.json``. If provided, used as the base caption.
        use_hard_negatives: If True, constrains negative sampling to the exact
            same category folder as the anchor (Strategy 2). Falls back to
            same clothes_type if the category pool has fewer than 2 unique items.
        use_structured_prompts: If True, prepends metadata tags to caption strings
            (Strategy 3). If False, uses plain VLM / attribute captions.
        augment_text: If True, randomly drops metadata tags at ``augment_prob``
            probability per item during __getitem__ (Strategy 5).
        augment_prob: Probability of applying text augmentation on any given sample.
    """

    def __init__(
        self,
        df,
        dataset_root: str,
        item_captions: dict,
        item_to_clothes_type: dict,
        item_to_category_folder: dict,
        vlm_captions: dict = None,
        use_hard_negatives: bool = True,
        use_structured_prompts: bool = True,
        augment_text: bool = True,
        augment_prob: float = 0.20,
    ):
        import random
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.item_captions = item_captions          # attribute-based captions fallback
        self.vlm_captions = vlm_captions or {}      # richer VLM descriptions
        self.item_to_clothes_type = item_to_clothes_type
        self.item_to_category_folder = item_to_category_folder
        self.use_hard_negatives = use_hard_negatives
        self.use_structured_prompts = use_structured_prompts
        self.augment_text = augment_text
        self.augment_prob = augment_prob
        self._rng = random

        # ── Build lookup tables ────────────────────────────────────────────────
        # Map item_id -> list of available shop image paths
        self.item_to_shops = df.groupby("item_id")["shop_image"].unique().to_dict()
        self.all_items = list(self.item_to_shops.keys())

        # Group by clothes_type (broad: 1=top, 2=bottom, 3=outfit)
        self.type_to_items: dict[int, list] = {}
        for item in self.all_items:
            ctype = self.item_to_clothes_type.get(item, 1)
            self.type_to_items.setdefault(ctype, []).append(item)

        # Group by category folder (fine-grained: Polo_Shirt, Maxi_Dress, …)
        self.category_to_items: dict[str, list] = {}
        for item in self.all_items:
            cat = self.item_to_category_folder.get(item, "__unknown__")
            self.category_to_items.setdefault(cat, []).append(item)

        # Validate that at least 2 items exist per clothes_type bucket
        for ctype, items in self.type_to_items.items():
            if len(items) < 2:
                raise ValueError(
                    f"AdvancedCLIPHybridDataset: clothes_type={ctype} has only "
                    f"{len(items)} item(s) — need ≥2 to sample hard negatives."
                )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _load(self, rel_path: str):
        """Load a PIL image from a relative dataset path, returning a blank image on error."""
        full_path = os.path.join(self.dataset_root, rel_path)
        try:
            return Image.open(full_path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def _build_caption(self, item_id: str) -> str:
        """
        Construct the text description for an item.

        If ``use_structured_prompts`` is True, prepends metadata tags:
            "[Category: Polo_Shirt] [Type: top] [Attributes: <attr_caption>] <vlm_caption>"

        Otherwise returns the plain VLM or attribute caption.

        If ``augment_text`` is True, randomly drops 1–2 of the structured tags
        with probability ``augment_prob`` to simulate partial user queries.

        Args:
            item_id: The DeepFashion item identifier string.

        Returns:
            A string containing the final caption for this item.
        """
        # Prefer richer VLM description; fall back to attribute-based caption
        base_caption = self.vlm_captions.get(item_id) or self.item_captions.get(item_id, "a clothing item")

        if not self.use_structured_prompts:
            return base_caption

        # ── Build structured tags ──────────────────────────────────────────────
        category = self.item_to_category_folder.get(item_id, "")
        clothes_type_int = self.item_to_clothes_type.get(item_id, 1)
        type_name = {1: "top", 2: "bottom", 3: "outfit"}.get(clothes_type_int, "clothing")
        attr_caption = self.item_captions.get(item_id, "")

        tags = []
        if category:
            tags.append(f"[Category: {category.replace('_', ' ')}]")
        tags.append(f"[Type: {type_name}]")
        if attr_caption:
            tags.append(f"[Attributes: {attr_caption}]")

        # ── Strategy 5: Dynamic attribute dropout ─────────────────────────────
        if self.augment_text and self._rng.random() < self.augment_prob and len(tags) > 1:
            # Randomly drop 1 tag (never drop all tags)
            drop_idx = self._rng.randrange(len(tags))
            tags.pop(drop_idx)

        return " ".join(tags) + " " + base_caption

    def _sample_negative(self, anchor_id: str) -> str:
        """
        Sample a negative item_id, optionally constrained to the same category
        (Strategy 2: Hard Negative Mining).

        If ``use_hard_negatives`` is True, searches within the same category
        folder pool first. Falls back to same clothes_type if the category pool
        has only 1 unique item, and falls back to global pool as a last resort.

        Args:
            anchor_id: The item_id of the anchor (positive) item.

        Returns:
            The item_id of the sampled negative item.
        """
        neg_item = anchor_id

        if self.use_hard_negatives:
            # Fine-grained pool: same category folder (e.g., Polo_Shirt)
            cat = self.item_to_category_folder.get(anchor_id, "__unknown__")
            pool = [i for i in self.category_to_items.get(cat, []) if i != anchor_id]

            if len(pool) < 1:
                # Fallback: same clothes_type (broad: top / bottom / outfit)
                ctype = self.item_to_clothes_type.get(anchor_id, 1)
                pool = [i for i in self.type_to_items.get(ctype, self.all_items) if i != anchor_id]

            if len(pool) < 1:
                pool = [i for i in self.all_items if i != anchor_id]
        else:
            # Random global negative (Strategy 2 disabled)
            pool = [i for i in self.all_items if i != anchor_id]

        neg_item = self._rng.choice(pool)
        return neg_item

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return a 4-tuple of (anchor_pil, positive_pil, negative_pil, caption_str).

        anchor   : Consumer photo (real-world, noisy).
        positive : Matching studio shop photo for the same item.
        negative : Shop photo for a different item (hard-mined or random).
        caption  : Structured / plain text description for the anchor item.
        """
        row = self.df.iloc[idx]
        anchor_id = row["item_id"]

        anchor_img = self._load(row["consumer_image"])
        pos_img = self._load(row["shop_image"])
        caption = self._build_caption(anchor_id)

        neg_item = self._sample_negative(anchor_id)
        neg_img_path = self._rng.choice(list(self.item_to_shops[neg_item]))
        neg_img = self._load(neg_img_path)

        return anchor_img, pos_img, neg_img, caption


def hybrid_augmented_collate_fn(batch, processor, tokenizer):
    """
    Custom DataLoader collation function for AdvancedCLIPHybridDataset batches.

    Converts a list of (anchor_pil, positive_pil, negative_pil, caption_str)
    tuples produced by AdvancedCLIPHybridDataset.__getitem__ into batched
    PyTorch tensors ready for GPU consumption.

    Processing steps:
        1. Transpose the list of tuples into four separate sequences.
        2. Run image triplets (anchor, positive, negative) through CLIP's
           image processor: resize to 224x224, apply ImageNet normalization,
           return ``pixel_values`` tensors of shape (B, 3, 224, 224).
        3. Tokenize caption strings with CLIP's tokenizer: convert to token
           ids, pad to uniform length, truncate at CLIP's 77-token maximum,
           return ``input_ids`` and ``attention_mask`` tensors of shape (B, L).

    Args:
        batch: List of (anchor_pil, positive_pil, negative_pil, caption_str)
            tuples collected by the DataLoader from the dataset.
        processor: A HuggingFace ``AutoProcessor`` / ``CLIPProcessor`` instance
            responsible for image preprocessing.
        tokenizer: A HuggingFace ``AutoTokenizer`` / ``CLIPTokenizer`` instance
            responsible for text tokenization.

    Returns:
        Tuple of (a, p, n, t, captions) where:
            a       : Dict with ``pixel_values`` tensor for anchor images.
            p       : Dict with ``pixel_values`` tensor for positive images.
            n       : Dict with ``pixel_values`` tensor for negative images.
            t       : Dict with ``input_ids`` and ``attention_mask`` for captions.
            captions: Original caption strings (useful for debugging / logging).
    """
    anchors, positives, negatives, captions = zip(*batch)

    # Process image triplets into normalized pixel tensors
    a = processor(images=list(anchors),   return_tensors="pt", padding=True)
    p = processor(images=list(positives), return_tensors="pt", padding=True)
    n = processor(images=list(negatives), return_tensors="pt", padding=True)

    # Tokenize text captions: pad shorter sequences, truncate at CLIP's 77-token limit
    t = tokenizer(list(captions), return_tensors="pt", padding=True, truncation=True, max_length=77)

    return a, p, n, t, list(captions)


class CLIPHybridDataset(Dataset):
    """
    Returns (anchor_pil, positive_pil, negative_pil, positive_caption) for hybrid CLIP tuning.
    Negative sampling is category-constrained to items with the same clothes_type.
    """
    def __init__(self, df, dataset_root, item_captions, item_to_clothes_type):
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.item_captions = item_captions
        self.item_to_clothes_type = item_to_clothes_type
        
        # Group shop images by item_id
        self.item_to_shops = df.groupby('item_id')['shop_image'].unique().to_dict()
        self.all_items = list(self.item_to_shops.keys())
        
        # Group items by clothes_type for category-constrained negative sampling
        self.type_to_items = {}
        for item in self.all_items:
            ctype = self.item_to_clothes_type.get(item, 1)
            self.type_to_items.setdefault(ctype, []).append(item)
            
        # Ensure we can sample negatives
        for ctype, items in self.type_to_items.items():
            if len(items) < 2:
                raise ValueError(
                    f"CLIPHybridDataset needs at least 2 unique items for clothes_type {ctype} "
                    f"to sample negatives, got {len(items)}."
                )

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        try:
            return Image.open(os.path.join(self.dataset_root, path)).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        anchor_id = row['item_id']
        
        anchor_img = self._load(row['consumer_image'])
        pos_img = self._load(row['shop_image'])
        pos_caption = self.item_captions.get(anchor_id, "a clothing")
        
        # Category-constrained negative sampling
        ctype = self.item_to_clothes_type.get(anchor_id, 1)
        type_items = self.type_to_items.get(ctype, self.all_items)
        
        neg_item = anchor_id
        while neg_item == anchor_id:
            neg_item = np.random.choice(type_items)
            
        neg_img_path = np.random.choice(self.item_to_shops[neg_item])
        neg_img = self._load(neg_img_path)
        
        return anchor_img, pos_img, neg_img, pos_caption

