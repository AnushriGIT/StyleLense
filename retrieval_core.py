"""
retrieval_core.py — Consolidate model loading, embedding generation, and search algorithms.
"""

import os
import io
import base64
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from config import (
    DATASET_BASE_PATH, EMBEDDINGS_DIR, EVAL_PARTITION_FILE
)

'''
# ==========================================
# MODEL SETUP & INITIALIZATION
# ==========================================
1. CONFIGURE DEVICE:
    Detect hardware acceleration. Use Apple Silicon GPU (MPS) or Nvidia GPU (CUDA) if available.
    Otherwise, default to CPU.

2. EXTRACT GALLERY IMAGE PATHS:
    Read the dataset evaluation splits. Collect all unique shop image file paths designated 
    for the "test" split. These form our searchable clothing database (gallery).

3. INITIALISE RESNET-50 MODEL:
    Load a ResNet-50 model with the classification layer removed (so it outputs a 2048-dim vector instead).
    - Look for a custom fine-tuned ResNet model (`resnet50_best.pth`).
    - If found, load those custom weights.
    - If not, fall back to standard pre-trained ImageNet weights.
    - Load the pre-computed ResNet gallery database (`trained_resnet50_gallery.npy`).

4. INITIALISE CLIP MODEL:
    - Attempt to load fine-tuned CLIP weights from `models/clip_best`.
    - If not found, load OpenAI's standard pre-trained CLIP weights from Hugging Face.
    - Load the CLIP pre-computed gallery database (`trained_clip_gallery.npy` or base `clip_gallery.npy`).

5. CREATE HELPER MAPS & VOCABULARY:
    - Build a fast lookup dictionary mapping absolute file paths to gallery indexes.
    - Define a vocabulary list of descriptive clothing attributes (e.g., "striped", "denim") 
      for zero-shot tagging on user-uploaded images.
'''

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ── Device Configuration ───────────────────────────────────────────────────────
DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available()  else
    "cuda" if torch.cuda.is_available()           else "cpu"
)

# ── Gallery paths (extracted from eval split) ───────────────────────────────────
df_eval = pd.read_csv(EVAL_PARTITION_FILE, sep=r'\s+', skiprows=1)
df_eval.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
GALLERY_PATHS = df_eval[df_eval['split'] == 'test']['shop_image'].unique().tolist()

# ── ResNet-50 ──────────────────────────────────────────────────────────────────
_base = models.resnet50(weights=None)
RESNET_MODEL = nn.Sequential(*list(_base.children())[:-1], nn.Flatten())

_best = os.path.join(MODELS_DIR, "resnet50_best.pth")
_old  = os.path.join(MODELS_DIR, "resnet50_finetuned.pth")

if os.path.exists(_best):
    RESNET_MODEL.load_state_dict(torch.load(_best, map_location=DEVICE))
elif os.path.exists(_old):
    RESNET_MODEL.load_state_dict(torch.load(_old, map_location=DEVICE))
else:
    _base2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    RESNET_MODEL = nn.Sequential(*list(_base2.children())[:-1], nn.Flatten())

RESNET_MODEL = RESNET_MODEL.eval().to(DEVICE)

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_resnet_emb_path = os.path.join(EMBEDDINGS_DIR, "trained_resnet50_gallery.npy")
RESNET_GALLERY = np.load(_resnet_emb_path).astype(np.float32) if os.path.exists(_resnet_emb_path) else None

# ── CLIP ───────────────────────────────────────────────────────────────────────
CLIP_MODEL     = None
CLIP_PROCESSOR = None
CLIP_GALLERY   = None

_FINETUNED_CLIP_DIR = os.path.join(MODELS_DIR, "clip_best")
_BASE_CLIP_ID       = "openai/clip-vit-base-patch32"

try:
    # pyrefly: ignore [missing-import]
    from transformers import CLIPModel, CLIPProcessor

    if os.path.exists(_FINETUNED_CLIP_DIR):
        CLIP_MODEL = CLIPModel.from_pretrained(_FINETUNED_CLIP_DIR).eval().to(DEVICE)
    else:
        CLIP_MODEL = CLIPModel.from_pretrained(_BASE_CLIP_ID).eval().to(DEVICE)

    CLIP_PROCESSOR = CLIPProcessor.from_pretrained(_BASE_CLIP_ID)

    _trained_clip_path = os.path.join(EMBEDDINGS_DIR, "trained_clip_gallery.npy")
    _base_clip_path    = os.path.join(EMBEDDINGS_DIR, "clip_gallery.npy")

    if os.path.exists(_trained_clip_path):
        CLIP_GALLERY = np.load(_trained_clip_path).astype(np.float32)
    elif os.path.exists(_base_clip_path):
        CLIP_GALLERY = np.load(_base_clip_path).astype(np.float32)
except Exception as e:
    print(f"CLIP not available: {e}")

# ── DINOv2 ─────────────────────────────────────────────────────────────────────
DINOV2_MODEL = None
DINOV2_GALLERY = None

_dinov2_best = os.path.join(MODELS_DIR, "dinov2_best.pth")
_dinov2_emb_path = os.path.join(EMBEDDINGS_DIR, "trained_dinov2_gallery.npy")

try:
    # Load DINOv2 ViT-B/14 backbone via torch.hub
    DINOV2_MODEL = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    if os.path.exists(_dinov2_best):
        DINOV2_MODEL.load_state_dict(torch.load(_dinov2_best, map_location=DEVICE))
    DINOV2_MODEL = DINOV2_MODEL.eval().to(DEVICE)

    if os.path.exists(_dinov2_emb_path):
        DINOV2_GALLERY = np.load(_dinov2_emb_path).astype(np.float32)
except Exception as e:
    print(f"DINOv2 loading failed: {e}")

# ── Path → gallery index mapping ───────────────────────────────────────────────
PATH_TO_GALLERY_IDX: dict = {
    os.path.join(DATASET_BASE_PATH, p): i
    for i, p in enumerate(GALLERY_PATHS)
}

# ── Attribute Vocabulary ───────────────────────────────────────────────────────
ATTRIBUTE_VOCABULARY = [
    "floral print", "solid colour", "striped", "checked pattern",
    "geometric print", "animal print", "abstract print",
    "cotton fabric", "denim fabric", "silk fabric", "wool fabric",
    "linen fabric", "leather material", "synthetic fabric",
    "slim fit", "classic fit", "oversized fit", "relaxed fit",
    "tailored fit", "bodycon fit", "A-line silhouette",
    "maxi length", "midi length", "mini length", "cropped",
    "V-neck", "round neck", "collared", "off-shoulder",
    "sleeveless", "long sleeve", "short sleeve",
    "casual wear", "formal wear", "business casual",
    "evening wear", "athletic wear", "streetwear",
    "black clothing", "white clothing", "navy blue clothing",
    "red clothing", "neutral tones", "pastel colours", "bright colours",
]
ATTRIBUTE_SCORE_THRESHOLD = 0.22

# ── Core Helpers ───────────────────────────────────────────────────────────────

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """
    Scale a vector (or matrix of vectors) to have a geometric length of exactly 1.0 (Unit Length).
    
    Why: L2 normalization is the secret to fast similarity search. Once vectors are scaled to 1.0,
    the simple Dot Product (multiplication) of two vectors becomes mathematically identical to
    their Cosine Similarity. This allows us to search thousands of images instantly.
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-8)


def embed_image(pil_image: Image.Image) -> np.ndarray:
    """
    Extract a 2048-dimensional feature embedding from an image using ResNet-50.
    
    Steps:
      1. Apply preprocessing transforms (Resize, crop, normalize colors).
      2. Run the image through the convolutional layers of ResNet-50.
      3. Flatten the output into a 2048-dim vector.
      4. Normalize the vector to unit length (L2 norm).
    """
    tensor = RESNET_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = RESNET_MODEL(tensor).cpu().numpy().astype(np.float32)
    return l2_normalize(emb)


def embed_text(query: str) -> np.ndarray:
    """
    Convert a text search query into a 512-dimensional CLIP embedding vector.
    
    Steps:
      1. Tokenize the text (convert words into unique numerical IDs).
      2. Run the token IDs through the CLIP Text Encoder Transformer.
      3. Project the output into the shared CLIP multi-modal space (512 dimensions).
      4. Normalize the vector to unit length (L2 norm).
    """
    inputs = CLIP_PROCESSOR(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_out = CLIP_MODEL.text_model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE)
        )
        emb = CLIP_MODEL.text_projection(text_out.pooler_output).cpu().numpy().astype(np.float32)
    return l2_normalize(emb)


def embed_image_clip(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a clothing image into a 512-dimensional CLIP embedding vector.
    
    Steps:
      1. Use the CLIP Processor to resize, normalize, and format the image.
      2. Pass it through the CLIP Vision Encoder (Vision Transformer).
      3. Project the visual features into the shared CLIP multi-modal space (512 dimensions).
      4. Normalize the vector to unit length (L2 norm).
    """
    inputs = CLIP_PROCESSOR(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_out = CLIP_MODEL.vision_model(**inputs)
        emb = CLIP_MODEL.visual_projection(vision_out.pooler_output).cpu().numpy().astype(np.float32)
    return l2_normalize(emb)


def embed_image_dinov2(pil_image: Image.Image) -> np.ndarray:
    """
    Extract a 768-dimensional feature embedding from an image using fine-tuned DINOv2.
    """
    if DINOV2_MODEL is None:
        raise ValueError("DINOv2 model is not loaded.")
    tensor = RESNET_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = DINOV2_MODEL(tensor).cpu().numpy().astype(np.float32)
    return l2_normalize(emb)


def text_search(query: str, top_k: int = 4) -> list:
    """
    Search the clothing gallery database using only a text query.
    
    Steps:
      1. Embed the query text into a 512-dim CLIP vector.
      2. Perform a nearest-neighbor similarity search against the pre-loaded CLIP gallery.
    """
    if CLIP_GALLERY is None or CLIP_MODEL is None:
        return []
    query_emb = embed_text(query)
    return nearest_search(query_emb, CLIP_GALLERY, top_k)


def nearest_search(query_emb: np.ndarray, gallery_emb: np.ndarray, top_k: int) -> list:
    """
    Compare a query vector against all database vectors using Cosine Similarity.
    
    Steps:
      1. Compute dot product between query vector (1, 512) and gallery database (512, N).
      2. Sort the scores from highest similarity to lowest.
      3. Slice the top-K matching item indices.
      4. Convert indices back to absolute image file paths, compute similarity scores,
         encode images to Base64 (for UI loading), and return them.
    """
    scores  = (query_emb @ gallery_emb.T)[0]
    indices = np.argsort(-scores)[:top_k]
    results = []
    for rank, idx in enumerate(indices):
        full_path = os.path.join(DATASET_BASE_PATH, GALLERY_PATHS[idx])
        results.append({
            "rank":       rank + 1,
            "similarity": round(float(scores[idx]) * 100, 1),
            "image_path": full_path,
            "image_data": _image_to_base64(full_path)
        })
    return results


def _text_refine(candidates: list, user_query: str, visual_weight: float = 0.7, text_weight: float = 0.3) -> list:
    """
    Re-rank visual search results using an additional text query description.
    
    Why: If a user uploads an image of a shirt and types "red stripes", we calculate
    a combined score: 70% based on the image match + 30% based on how well the candidate
    image matches the text "red stripes" via CLIP text-to-image similarity.
    """
    try:
        text_emb = embed_text(user_query)[0]
        scored = []
        for c in candidates:
            visual_score = c["similarity"] / 100.0
            gallery_idx = PATH_TO_GALLERY_IDX.get(c.get("image_path", ""))
            if gallery_idx is not None:
                clip_emb   = CLIP_GALLERY[gallery_idx]
                text_score = float(np.dot(text_emb, clip_emb))
            else:
                text_score = 0.0

            combined = visual_weight * visual_score + text_weight * text_score
            scored.append((combined, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {**c, "rank": rank + 1, "similarity": c["similarity"]}
            for rank, (_, c) in enumerate(scored)
        ]
    except Exception:
        return candidates


def _tag_attributes(pil_image: Image.Image) -> list:
    """
    Analyze an image and automatically tag it with descriptive attributes (Zero-Shot Classification).
    
    Steps:
      1. Embed the input image into 512-dim space.
      2. Loop through all tags in ATTRIBUTE_VOCABULARY, embedding them into text vectors.
      3. Measure the cosine similarity between the image and each tag vector.
      4. Retain and return tags that exceed the similarity score threshold (e.g. 0.22).
    """
    img_emb = embed_image_clip(pil_image)
    scored = []
    for attr in ATTRIBUTE_VOCABULARY:
        score = float((img_emb @ embed_text(attr).T)[0, 0])
        if score >= ATTRIBUTE_SCORE_THRESHOLD:
            scored.append((attr, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [_clean_tag(attr) for attr, _ in scored[:6]]


def _clean_tag(attr: str) -> str:
    """
    Clean up classification labels to make them human-readable and clean.
    
    Example: Converts "cotton fabric" -> "cotton", "A-line silhouette" -> "A-line".
    This filters out redundant stopwords like "fabric", "material", "clothing", etc.
    """
    stopwords = {"fabric", "material", "print", "pattern",
                 "length", "colour", "tones", "wear", "clothing"}
    words = [w for w in attr.split() if w.lower() not in stopwords]
    return " ".join(words) if words else attr


def _image_to_base64(image_path: str, max_size: tuple = (200, 270)) -> str:
    """
    Load a local image file, resize it to a thumbnail, and encode it to a Base64 string.
    
    Why: Instead of the frontend trying to fetch image URLs from disk (which local browsers
    block for security), we embed the image directly into the JSON response as a base64 string.
    The browser can display this string instantly using `<img src="data:image/jpeg;base64,...">`.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""
