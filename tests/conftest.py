"""
conftest.py — shared pytest fixtures.

retrieval_core.py loads real CLIP/DINOv2 weights and a real DeepFashion eval-partition
CSV at *import time*, with no fallback if either is absent (see reviewLog.md finding #11 —
flagged there as a design gap, not fixed). Patch pandas/torch.hub/transformers BEFORE the
module is ever imported so the whole suite runs without network, real model weights, or the
real dataset present on this machine. This fixture is session-scoped and autouse so every
test file gets the same mocked import — api/main.py's `import retrieval_core as utils`
picks up the same already-patched module from sys.modules.
"""
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def mocked_retrieval_core():
    if "retrieval_core" in sys.modules:
        # Already imported (e.g. by a previous test run in the same session) — nothing to do.
        yield
        return

    def fake_read_csv(*args, **kwargs):
        # Shape matches what retrieval_core.py expects after it assigns
        # df_eval.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
        return pd.DataFrame([
            ["img/c1.jpg", "img/s1.jpg", "id_1", "train"],
            ["img/c2.jpg", "img/s2.jpg", "id_1", "test"],
            ["img/c3.jpg", "img/s3.jpg", "id_2", "test"],
        ])

    fake_clip_model = MagicMock(name="FakeCLIPModel")
    fake_clip_model.eval.return_value = fake_clip_model
    fake_clip_model.to.return_value = fake_clip_model

    fake_dinov2_model = MagicMock(name="FakeDINOv2Model")
    fake_dinov2_model.eval.return_value = fake_dinov2_model
    fake_dinov2_model.to.return_value = fake_dinov2_model

    with patch("pandas.read_csv", side_effect=fake_read_csv), \
         patch("torch.hub.load", return_value=fake_dinov2_model), \
         patch("transformers.CLIPModel.from_pretrained", return_value=fake_clip_model), \
         patch("transformers.CLIPProcessor.from_pretrained", return_value=MagicMock(name="FakeCLIPProcessor")):
        import retrieval_core  # noqa: F401  (imported under patches; real network/model work skipped)

    yield
