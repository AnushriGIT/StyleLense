"""
generate_embeddings.py — Generate CLIP visual embeddings for query and gallery sets.

Supports any CLIP-based checkpoint saved in the models/ directory, including all
5 ablation study runs (clip_vlm_hybrid_runA through runE).

Run:
    # Original text-image model
    .venv/bin/python scripts/generate_embeddings.py --model clip_text_image

    # Ablation study runs
    .venv/bin/python scripts/generate_embeddings.py --model clip_vlm_hybrid_runA
    .venv/bin/python scripts/generate_embeddings.py --model clip_vlm_hybrid_runE
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATASET_BASE_PATH, EVAL_PARTITION_FILE, EMBEDDINGS_DIR

# ── Model checkpoint directory mapping ────────────────────────────────────────
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

MODEL_DIRS = {
    "clip_text_image":      os.path.join(MODELS_DIR, "clip_text_image_best"),
    "clip_vlm_hybrid_runA": os.path.join(MODELS_DIR, "clip_vlm_hybrid_runA"),
    "clip_vlm_hybrid_runB": os.path.join(MODELS_DIR, "clip_vlm_hybrid_runB"),
    "clip_vlm_hybrid_runC": os.path.join(MODELS_DIR, "clip_vlm_hybrid_runC"),
    "clip_vlm_hybrid_runD": os.path.join(MODELS_DIR, "clip_vlm_hybrid_runD"),
    "clip_vlm_hybrid_runE": os.path.join(MODELS_DIR, "clip_vlm_hybrid_runE"),
    "clip_vlm_hybrid_text_attempt1": os.path.join(MODELS_DIR, "clip_vlm_hybrid_text_attempt1"),
    "clip_vlm_hybrid_text_attempt2": os.path.join(MODELS_DIR, "clip_vlm_hybrid_text_attempt2"),
    "clip_vlm_hybrid_text_attempt3": os.path.join(MODELS_DIR, "clip_vlm_hybrid_text_attempt3"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for embedding generation."""
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for evaluation.")
    parser.add_argument(
        "--model", default="clip_text_image",
        choices=list(MODEL_DIRS.keys()),
        help="Which model checkpoint to generate embeddings from.",
    )
    return parser.parse_args()


def load_partition(eval_file: str):
    """
    Load the DeepFashion test partition to determine query and gallery image paths.

    Args:
        eval_file: Path to list_eval_partition.txt.

    Returns:
        Tuple of (query_paths, gallery_paths) — lists of relative image paths.
    """
    df = pd.read_csv(eval_file, sep=r'\s+', skiprows=1)
    df.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    gallery_paths = df_test['shop_image'].unique().tolist()
    query_paths = df_test['consumer_image'].tolist()
    return query_paths, gallery_paths


def get_clip_embedding(model, processor, img_path: str, device) -> np.ndarray:
    """
    Compute a normalised L2 CLIP visual embedding for a single image.

    Loads the image at ``img_path`` relative to ``DATASET_BASE_PATH``, runs it
    through the vision encoder and visual projection head, and returns a
    1D float32 numpy vector of shape (embedding_dim,).

    Args:
        model: A CLIPModel instance.
        processor: A CLIPProcessor instance.
        img_path: Relative path to the image within the dataset directory.
        device: PyTorch device.

    Returns:
        L2-normalised float32 numpy vector of shape (embedding_dim,).
    """
    full_path = os.path.join(DATASET_BASE_PATH, img_path)
    try:
        pil_img = Image.open(full_path).convert("RGB")
    except Exception:
        pil_img = Image.new("RGB", (224, 224))

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = model.vision_model(**inputs)
        emb = model.visual_projection(vision_out.pooler_output)

    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32).flatten()


def main():
    """Entry point: load checkpoint, generate query/gallery embeddings, save as .npy files."""
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model_dir = MODEL_DIRS[args.model]
    if not os.path.exists(model_dir):
        print(f"Model checkpoint not found at {model_dir}. Train the model first.")
        return

    print(f"Loading model from: {model_dir}")
    model = CLIPModel.from_pretrained(model_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_dir)
    model.eval()

    query_paths, gallery_paths = load_partition(EVAL_PARTITION_FILE)
    print(f"Generating embeddings for {len(query_paths)} queries and {len(gallery_paths)} gallery images.")

    query_embs = []
    for path in tqdm(query_paths, desc="Queries"):
        query_embs.append(get_clip_embedding(model, processor, path, device))
    query_embs = np.array(query_embs)

    gallery_embs = []
    for path in tqdm(gallery_paths, desc="Gallery"):
        gallery_embs.append(get_clip_embedding(model, processor, path, device))
    gallery_embs = np.array(gallery_embs)

    query_out   = os.path.join(EMBEDDINGS_DIR, f"{args.model}_queries.npy")
    gallery_out = os.path.join(EMBEDDINGS_DIR, f"{args.model}_gallery.npy")
    np.save(query_out,   query_embs)
    np.save(gallery_out, gallery_embs)
    print(f"Saved embeddings.")
    print(f"  Queries : {query_out}")
    print(f"  Gallery : {gallery_out}")


if __name__ == "__main__":
    main()
