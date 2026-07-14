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
    CLIPTripletDataset
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


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__]))
