"""
config.py — Central configuration for the StyleLens retrieval-core project.

All paths and settings live here so you only need to change things in ONE place.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ── Dataset Paths ──────────────────────────────────────────────────────────────
# 👉 Override via the DATASET_BASE_PATH env var (e.g. in a .env file) for any
# machine/container where the dataset isn't at this default location.
DATASET_BASE_PATH = os.getenv(
    "DATASET_BASE_PATH",
    "/Users/anushrin/Documents/Cowork Playground/FashionRecommendation/workspace/Capstone Fashion Assistant/fashion_assistant/data/DeepFashion Database/Data/Consumer-to-shop Clothes Retrieval Benchmark"
)

# Subdirectories inside the dataset (standard DeepFashion layout)
IMG_DIR         = os.path.join(DATASET_BASE_PATH, "img")
ANNO_DIR        = os.path.join(DATASET_BASE_PATH, "Anno")
EVAL_DIR        = os.path.join(DATASET_BASE_PATH, "Eval")

# Key annotation files from DeepFashion
CATEGORY_IMG_FILE = os.path.join(ANNO_DIR, "list_item_consumer2shop.txt")
CATEGORY_CLO_FILE = os.path.join(ANNO_DIR, "list_attr_cloth.txt")
ATTR_IMG_FILE     = os.path.join(ANNO_DIR, "list_attr_items.txt")
BBOX_FILE         = os.path.join(ANNO_DIR, "list_bbox_consumer2shop.txt")
EVAL_PARTITION_FILE = os.path.join(EVAL_DIR, "list_eval_partition.txt")

# ── Model & Embedding Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR  = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
IMAGE_PATHS_FILE = os.path.join(EMBEDDINGS_DIR, "image_paths.npy")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")

# ── Model Settings ─────────────────────────────────────────────────────────────
MODEL_NAME      = "resnet50"          # CNN backbone
EMBEDDING_DIM   = 2048               # ResNet50 outputs a 2048-dim feature vector
IMAGE_SIZE      = 224                # ResNet50 expects 224x224 images

# ── Search Settings ───────────────────────────────────────────────────────────
TOP_K           = 10                 # How many similar images to return

# ── Web App Settings ──────────────────────────────────────────────────────────
UPLOAD_FOLDER   = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER   = os.path.join(BASE_DIR, "api", "static")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ── Batch Processing ──────────────────────────────────────────────────────────
# How many images to process at once when building the embeddings database.
BATCH_SIZE      = 64

# Max images to index (use None for all, or a number like 5000 for quick testing)
MAX_IMAGES      = None   # Set to e.g. 5000 while testing to speed things up

# ── Ensure output directories exist ──────────────────────────────────────────
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
