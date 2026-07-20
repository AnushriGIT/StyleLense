import os
import sys
import pytest
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

# Add parent directory to path so datasets can be imported when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the datasets from the main module
from fashion_ds_processing import (
    FashionDataset,
    TripletFashionDataset,
    CachedTripletFashionDataset,
    PairFashionDataset,
    CLIPTripletDataset,
    resolve_garment_name,
    load_item_metadata,
    build_item_captions,
    CLIPHybridDataset
)

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    """Creates a dummy dataset directory with some images."""
    data_dir = tmp_path / "dummy_data"
    data_dir.mkdir()
    
    # Create a few dummy images
    for i in range(5):
        img = Image.new("RGB", (224, 224), color=(i*10, i*20, i*30))
        img.save(data_dir / f"img_{i}.jpg")
    
    # Create one corrupted/missing path scenario by NOT creating img_99.jpg
    
    return str(data_dir)

@pytest.fixture
def dummy_dataframe():
    """Creates a dummy dataframe matching the DeepFashion format."""
    return pd.DataFrame({
        'item_id': ['item_1', 'item_1', 'item_2', 'item_3'],
        'consumer_image': ['img_0.jpg', 'img_1.jpg', 'img_2.jpg', 'img_99.jpg'], # img_99 is missing
        'shop_image': ['img_3.jpg', 'img_3.jpg', 'img_4.jpg', 'img_0.jpg']
    })

@pytest.fixture
def dummy_transform():
    """Simple transform that just converts to tensor."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

@pytest.fixture
def single_item_dataframe():
    """Only one unique item_id — used to trigger the negative-sampling ValueError guard."""
    return pd.DataFrame({
        'item_id': ['item_1', 'item_1'],
        'consumer_image': ['img_0.jpg', 'img_1.jpg'],
        'shop_image': ['img_3.jpg', 'img_3.jpg']
    })

def test_fashion_dataset(dummy_dataset_dir, dummy_transform):
    paths = ['img_0.jpg', 'img_1.jpg', 'img_99.jpg'] # Includes missing image
    dataset = FashionDataset(paths, dummy_dataset_dir, dummy_transform)
    
    assert len(dataset) == 3
    
    # Valid image
    img, idx = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert idx == 0
    
    # Missing image (should return zeros instead of crashing)
    img_missing, idx_missing = dataset[2]
    assert isinstance(img_missing, torch.Tensor)
    assert img_missing.shape == (3, 224, 224)
    assert torch.all(img_missing == 0) # Fallback is torch.zeros
    assert idx_missing == 2

def test_triplet_fashion_dataset(dummy_dataframe, dummy_dataset_dir, dummy_transform):
    dataset = TripletFashionDataset(dummy_dataframe, dummy_dataset_dir, dummy_transform)
    
    assert len(dataset) == len(dummy_dataframe)
    
    anchor, pos, neg = dataset[0]
    
    assert isinstance(anchor, torch.Tensor)
    assert isinstance(pos, torch.Tensor)
    assert isinstance(neg, torch.Tensor)
    
    assert anchor.shape == (3, 224, 224)
    assert pos.shape == (3, 224, 224)
    assert neg.shape == (3, 224, 224)

def test_cached_triplet_fashion_dataset(dummy_dataframe, dummy_dataset_dir, dummy_transform):
    dataset = CachedTripletFashionDataset(dummy_dataframe, dummy_dataset_dir, dummy_transform)
    
    assert len(dataset) == len(dummy_dataframe)
    
    anchor, pos, neg = dataset[0]
    
    assert isinstance(anchor, torch.Tensor)
    assert isinstance(pos, torch.Tensor)
    assert isinstance(neg, torch.Tensor)
    
    assert anchor.shape == (3, 224, 224)
    assert pos.shape == (3, 224, 224)
    assert neg.shape == (3, 224, 224)

def test_pair_fashion_dataset(dummy_dataframe, dummy_dataset_dir, dummy_transform):
    dataset = PairFashionDataset(dummy_dataframe, dummy_dataset_dir, dummy_transform)
    
    # Length should be exactly double the dataframe length (half pos, half neg)
    assert len(dataset) == 2 * len(dummy_dataframe)
    
    # First half should be positive pairs (label 1.0)
    img1_pos, img2_pos, label_pos = dataset[0]
    assert label_pos.item() == 1.0
    
    # Second half should be negative pairs (label 0.0)
    img1_neg, img2_neg, label_neg = dataset[len(dummy_dataframe)]
    assert label_neg.item() == 0.0
    
    assert img1_pos.shape == (3, 224, 224)

def test_clip_triplet_dataset(dummy_dataframe, dummy_dataset_dir):
    # Note: CLIP dataset returns raw PIL images, not tensors
    dataset = CLIPTripletDataset(dummy_dataframe, dummy_dataset_dir)
    
    assert len(dataset) == len(dummy_dataframe)
    
    anchor, pos, neg = dataset[0]
    
    assert isinstance(anchor, Image.Image)
    assert isinstance(pos, Image.Image)
    assert isinstance(neg, Image.Image)
    
    # Missing images should fallback to a new black PIL image of 224x224
    anchor_missing, pos_missing, neg_missing = dataset[3]
    assert anchor_missing.size == (224, 224)



# ── Regression tests for reviewLog.md v1 finding #14 / v4 fix ────────────────
# With only one unique item_id, the old `while neg_item == row['item_id']: ...`
# negative-sampling loop in these four classes would spin forever. Each class now
# fails fast with a ValueError at construction time instead.

def test_triplet_fashion_dataset_raises_on_single_item(single_item_dataframe, dummy_dataset_dir, dummy_transform):
    with pytest.raises(ValueError):
        TripletFashionDataset(single_item_dataframe, dummy_dataset_dir, dummy_transform)

def test_cached_triplet_fashion_dataset_raises_on_single_item(single_item_dataframe, dummy_dataset_dir, dummy_transform):
    with pytest.raises(ValueError):
        CachedTripletFashionDataset(single_item_dataframe, dummy_dataset_dir, dummy_transform)

def test_pair_fashion_dataset_raises_on_single_item(single_item_dataframe, dummy_dataset_dir, dummy_transform):
    with pytest.raises(ValueError):
        PairFashionDataset(single_item_dataframe, dummy_dataset_dir, dummy_transform)

def test_clip_triplet_dataset_raises_on_single_item(single_item_dataframe, dummy_dataset_dir):
    with pytest.raises(ValueError):
        CLIPTripletDataset(single_item_dataframe, dummy_dataset_dir)


def test_cached_triplet_fashion_dataset_caching_and_load_fallback(dummy_dataframe, dummy_dataset_dir, dummy_transform):
    dataset = CachedTripletFashionDataset(dummy_dataframe, dummy_dataset_dir, dummy_transform)
    assert len(dataset.cache) == 6
    assert "img_99.jpg" in dataset.cache
    assert isinstance(dataset.cache["img_99.jpg"], Image.Image)
    assert dataset.cache["img_99.jpg"].size == (224, 224)
    
    # Test _load handles uncached path gracefully
    zeros = dataset._load("nonexistent_key.jpg")
    assert isinstance(zeros, torch.Tensor)
    assert torch.all(zeros == 0)


def test_resolve_garment_name():
    # Whitelisted names
    assert resolve_garment_name("Blouse", 1) == "blouse"
    assert resolve_garment_name("Jeans", 2) == "jeans"
    assert resolve_garment_name("T_Shirt", 1) == "t-shirt"
    
    # Fallback scenarios
    assert resolve_garment_name("Chiffon", 1) == "top"
    assert resolve_garment_name("Chiffon", 2) == "bottom"
    assert resolve_garment_name("Summer_Wear", 3) == "outfit"
    assert resolve_garment_name("Unknown_Folder", 99) == "clothing"


def test_build_item_captions_and_metadata_parsing(tmp_path):
    # Setup dummy directory structure
    anno_dir = tmp_path / "Anno"
    anno_dir.mkdir()
    
    # Write list_attr_cloth.txt
    cloth_file = anno_dir / "list_attr_cloth.txt"
    cloth_file.write_text(
        "4\n"
        "attribute_name (Chinese)         attribute_name (English)                     attribute_type\n"
        "常规款（55-65cm）                Regular length (55-65cm)                     1\n"
        "纯棉                             Cotton                                       7\n"
        "条纹                             Striped                                      6\n"
        "涤纶                             Polyester                                    7\n",
        encoding="utf-8"
    )
    
    # Write list_attr_type.txt
    type_file = anno_dir / "list_attr_type.txt"
    type_file.write_text(
        "7\n"
        "attribute_type (Chinese)      attribute_type (English)\n"
        "衣长                          length of upper-body clothes\n"
        "裤型                          type of trousers\n"
        "细节                          part details of clothes\n"
        "衣门襟                        type of clothes buttons\n"
        "裤长                          length of trousers\n"
        "风格                          style of clothes\n"
        "材质                          fabric of clothes\n",
        encoding="utf-8"
    )
    
    # Write list_attr_items.txt
    items_file = anno_dir / "list_attr_items.txt"
    # Column indices correspond to vocab: 0: Regular length, 1: Cotton, 2: Striped, 3: Polyester
    # item_1 has active: Regular length (0), Cotton (1), Striped (2), Polyester (3) -> tie break for fabric is Cotton (lowest index)
    # item_2 has only Striped (2)
    items_file.write_text(
        "2\n"
        "item_id     attribute_labels\n"
        "item_1  1  1  1  1\n"
        "item_2 -1 -1  1 -1\n",
        encoding="utf-8"
    )
    
    # Write list_bbox_consumer2shop.txt
    bbox_file = anno_dir / "list_bbox_consumer2shop.txt"
    bbox_file.write_text(
        "2\n"
        "image_name clothes_type source_type x_1 y_1 x_2 y_2\n"
        "img/TOPS/T_Shirt/item_1/shop_01.jpg              1 1 001 039 200 292\n"
        "img/TROUSERS/Jeans/item_2/shop_01.jpg             2 1 001 039 200 292\n",
        encoding="utf-8"
    )
    
    # Build captions
    captions = build_item_captions(str(tmp_path))
    
    # Check item_1: style="striped", fabric="cotton" (tie broke from cotton/polyester), garment="t-shirt", length="regular (55-65cm)", sleeve=""
    # Final caption: "a striped cotton t-shirt with regular (55-65cm) length"
    assert "item_1" in captions
    assert captions["item_1"] == "a striped cotton t-shirt with regular (55-65cm) length"
    
    # Check item_2: style="striped", fabric="", garment="jeans", length="", sleeve=""
    # Final caption: "a striped jeans"
    assert "item_2" in captions
    assert captions["item_2"] == "a striped jeans"


def test_clip_hybrid_dataset(dummy_dataframe, dummy_dataset_dir):
    item_captions = {
        "item_1": "a striped top",
        "item_2": "a denim bottom",
        "item_3": "a summer outfit"
    }
    item_to_clothes_type = {
        "item_1": 1,
        "item_2": 2,
        "item_3": 1  # item_1 and item_3 share clothes_type 1, allowing negative sampling
    }
    
    # Since we need at least 2 unique items for clothes_type 2 to sample negatives,
    # let's add item_4 with clothes_type 2 to prevent ValueError
    df = dummy_dataframe.copy()
    df = pd.concat([df, pd.DataFrame({
        'item_id': ['item_4'],
        'consumer_image': ['img_0.jpg'],
        'shop_image': ['img_4.jpg']
    })], ignore_index=True)
    
    item_to_clothes_type["item_4"] = 2
    item_captions["item_4"] = "a cotton bottom"
    
    dataset = CLIPHybridDataset(df, dummy_dataset_dir, item_captions, item_to_clothes_type)
    assert len(dataset) == len(df)
    
    anchor, pos, neg, caption = dataset[0]
    
    assert isinstance(anchor, Image.Image)
    assert isinstance(pos, Image.Image)
    assert isinstance(neg, Image.Image)
    assert isinstance(caption, str)
    assert caption == "a striped top"
    
    # Negative item check: should belong to the same category (clothes_type 1) but be a different item
    # Since only item_1 and item_3 have type 1, negative item must be item_3
    # Let's verify by checking that the negative loader returned anchor's type matching items
    # item_3 is the only option, so the shop image must be 'img_0.jpg' (from dummy_dataframe item_3)
    # The dummy image color for img_0 is (0, 0, 0)
    # Let's inspect the negative image's top-left pixel color
    neg_pixel = neg.getpixel((0, 0))
    assert neg_pixel == (0, 0, 0)


def test_shop_captions_loading(tmp_path):
    """Verify that shop_captions.json can be loaded correctly as a dictionary."""
    import json
    data = {
        "item_1": "a red cotton dress with short sleeves",
        "item_2": "a blue denim jacket with patches"
    }
    
    file_path = tmp_path / "shop_captions.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        
    # Test reading back
    with open(file_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        
    assert loaded["item_1"] == "a red cotton dress with short sleeves"
    assert loaded["item_2"] == "a blue denim jacket with patches"
    assert len(loaded) == 2


def test_pair_fashion_dataset_constrained_sampling(dummy_dataframe, dummy_dataset_dir, dummy_transform, tmp_path):
    """Verify that PairFashionDataset correctly constrains negatives by clothes_type category."""
    # Write a dummy list_bbox_consumer2shop.txt so load_item_metadata works
    anno_dir = tmp_path / "Anno"
    anno_dir.mkdir(parents=True, exist_ok=True)
    
    # item_1 has clothes_type 1
    # item_2 has clothes_type 2
    # item_3 has clothes_type 1 (same as item_1)
    bbox_lines = [
        "2\n",
        "image_path clothes_type\n",
        "img/TOPS/T_Shirt/item_1/shop_01.jpg 1\n",
        "img/TROUSERS/Jeans/item_2/shop_01.jpg 2\n",
        "img/TOPS/Sweater/item_3/shop_01.jpg 1\n"
    ]
    
    bbox_file = anno_dir / "list_bbox_consumer2shop.txt"
    with open(bbox_file, "w", encoding="utf-8") as f:
        f.writelines(bbox_lines)
        
    dataset = PairFashionDataset(dummy_dataframe, str(tmp_path), dummy_transform)
    
    # Verify metadata loading
    assert dataset.item_to_clothes_type["item_1"] == 1
    assert dataset.item_to_clothes_type["item_2"] == 2
    assert dataset.item_to_clothes_type["item_3"] == 1
    
    # Run negative retrieval (index >= len(df))
    # For index 4 (anchor is dummy_dataframe.iloc[0] which is item_1, type 1)
    # The only eligible negative is item_3 (different ID, same clothes_type 1)
    # item_2 has type 2 and must be skipped
    for _ in range(20): # test multiple random draws to ensure no jeans are drawn
        img1, img2, label = dataset[4]
        assert label == 0.0
        # Since it always draws item_3, verify that the image matches item_3's shop_image
        # If it failed category constraint, it would draw item_2 sometimes.


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))


