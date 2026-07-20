import os
import pytest
import pandas as pd
from PIL import Image
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fashion_ds_processing import CLIPShopCaptionDataset

@pytest.fixture
def dummy_dataset_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    return str(root)

@pytest.fixture
def dummy_df():
    data = {
        "item_id": ["id_001", "id_002"],
        "consumer_image": ["img/consumer/id_001.jpg", "img/consumer/id_002.jpg"],
        "shop_image": ["img/shop/id_001.jpg", "img/shop/id_002.jpg"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_captions_file(tmp_path):
    captions = tmp_path / "shop_captions.json"
    import json
    with open(captions, "w") as f:
        json.dump({"id_001": "a red t-shirt", "id_003": "a blue jeans"}, f)
    return str(captions)

def test_clip_shop_caption_dataset_loads_correct_caption(dummy_df, dummy_dataset_root, dummy_captions_file):
    # Patch Image.open to avoid actually reading from disk
    with patch("PIL.Image.open") as mock_open_img:
        mock_open_img.return_value = Image.new("RGB", (224, 224))
        
        dataset = CLIPShopCaptionDataset(dummy_df, dummy_dataset_root, captions_file=dummy_captions_file)
        
        # Test item_id "id_001" (has caption)
        img1, caption1 = dataset[0]
        assert caption1 == "a red t-shirt"
        
        # Test item_id "id_002" (missing caption, should fallback)
        img2, caption2 = dataset[1]
        assert caption2 == "a piece of clothing"
        
        # Test len
        assert len(dataset) == 2
