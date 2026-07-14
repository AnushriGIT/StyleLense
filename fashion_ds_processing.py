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
"""

import os
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
        self.df            = df_train.reset_index(drop=True)
        self.dataset_root  = dataset_root
        self.transform     = transform
        self.item_to_shops = df_train.groupby('item_id')['shop_image'].unique().to_dict()
        self.all_items     = list(self.item_to_shops.keys())
        self.length        = len(self.df) * 2

    def __len__(self):
        return self.length

    def _load(self, path):
        try:
            return self.transform(Image.open(os.path.join(self.dataset_root, path)).convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx % len(self.df)]
        img1  = self._load(row['consumer_image'])

        if idx < len(self.df):
            img2  = self._load(row['shop_image'])
            label = torch.tensor(1.0)
        else:
            neg_item = row['item_id']
            while neg_item == row['item_id']:
                neg_item = np.random.choice(self.all_items)
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
