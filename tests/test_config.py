import importlib
import os

import config


def test_directories_autocreated():
    assert os.path.exists(config.EMBEDDINGS_DIR)
    assert os.path.exists(config.UPLOAD_FOLDER)


def test_default_dataset_base_path_when_no_env_var(monkeypatch):
    monkeypatch.delenv("DATASET_BASE_PATH", raising=False)
    importlib.reload(config)
    try:
        assert config.DATASET_BASE_PATH
        assert "DeepFashion" in config.DATASET_BASE_PATH
    finally:
        importlib.reload(config)


def test_dataset_base_path_env_override(monkeypatch, tmp_path):
    # Regression test for reviewLog.md v1 finding #4: DATASET_BASE_PATH was hardcoded
    # to a personal absolute path with no way to override it for another machine/container.
    override = str(tmp_path / "custom_dataset")
    monkeypatch.setenv("DATASET_BASE_PATH", override)
    importlib.reload(config)
    try:
        assert config.DATASET_BASE_PATH == override
    finally:
        monkeypatch.delenv("DATASET_BASE_PATH", raising=False)
        importlib.reload(config)


def test_allowed_extensions_and_max_content_length():
    # Regression test for reviewLog.md v1 findings #6/#8: these were declared in config.py
    # but never enforced in api/main.py. Pin the values here so a future change to either
    # is caught, since api/main.py imports these exact names.
    assert isinstance(config.ALLOWED_EXTENSIONS, set)
    assert {"png", "jpg", "jpeg", "webp"} <= config.ALLOWED_EXTENSIONS
    assert config.MAX_CONTENT_LENGTH == 16 * 1024 * 1024
