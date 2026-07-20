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

def load_partition(eval_file: str):
    df = pd.read_csv(eval_file, sep=r'\s+', skiprows=1)
    df.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    gallery_paths = df_test['shop_image'].unique().tolist()
    query_paths = df_test['consumer_image'].tolist()

    return query_paths, gallery_paths

def get_clip_embedding(model, processor, img_path, device):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'clip_text_image_best'))
    if not os.path.exists(model_dir):
        print(f"Model not found at {model_dir}")
        return
        
    model = CLIPModel.from_pretrained(model_dir).to(device)
    processor = CLIPProcessor.from_pretrained(model_dir)
    model.eval()
    
    query_paths, gallery_paths = load_partition(EVAL_PARTITION_FILE)
    print(f"Generating for {len(query_paths)} queries and {len(gallery_paths)} gallery images.")
    
    # Due to small time limit on test run, let's just generate for a small subset if requested,
    # but evaluate_retrieval.py needs the full array size or aligned paths.
    # We will generate for all to match evaluate_retrieval expectations.
    
    query_embs = []
    for path in tqdm(query_paths, desc="Queries"):
        query_embs.append(get_clip_embedding(model, processor, path, device))
    query_embs = np.array(query_embs)
    
    gallery_embs = []
    for path in tqdm(gallery_paths, desc="Gallery"):
        gallery_embs.append(get_clip_embedding(model, processor, path, device))
    gallery_embs = np.array(gallery_embs)
    
    np.save(os.path.join(EMBEDDINGS_DIR, "clip_text_image_queries.npy"), query_embs)
    np.save(os.path.join(EMBEDDINGS_DIR, "clip_text_image_gallery.npy"), gallery_embs)
    print("Saved embeddings.")

if __name__ == "__main__":
    main()
