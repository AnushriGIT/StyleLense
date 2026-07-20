"""
local_vlm_generator.py — Offline Local VLM Captioning Pipeline.

This script executes a local Vision-Language Model (Salesforce BLIP-Large)
to generate high-fidelity, instance-specific captions for the DeepFashion shop images.
It merges existing whitelisted attribute labels as a contextual hint for the prompt,
ensuring the resulting description captures both ground-truth attributes and visual details.
"""

import os
import json
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor
from fashion_ds_processing import build_item_captions

def parse_args():
    """
    Parses CLI arguments for the VLM generator.
    
    Returns:
        argparse.Namespace: Configured command line options.
    """
    parser = argparse.ArgumentParser(description="Generate shop image captions using a local VLM.")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of shop items to process (for testing/verification)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="embeddings/shop_captions.json", 
        help="Path to output JSON file."
    )
    return parser.parse_args()

def main():
    """
    Main pipeline execution:
      1. Configures execution device (MPS for Apple Silicon, CPU fallback).
      2. Loads whitelisted ground-truth template captions.
      3. Parses unique shop test images.
      4. Downloads and instantiates the local BLIP-Large VLM.
      5. Runs batch inference matching attributes with image pixels.
      6. Writes results incrementally to shop_captions.json.
    """
    args = parse_args()
    
    # 1. Device Configuration
    # Uses MPS (Metal Performance Shaders) for GPU acceleration on Mac
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # 2. Paths Configuration
    DATASET_PATH = "/Users/anushrin/Documents/Cowork Playground/FashionRecommendation/workspace/Capstone Fashion Assistant/fashion_assistant/data/DeepFashion Database/Data/Consumer-to-shop Clothes Retrieval Benchmark"
    
    # 3. Load Existing Ground-Truth Whitelisted Captions/Attributes
    # This retrieves whitelisted attributes (style, fabric, type) for each item ID
    print("Loading existing ground-truth whitelisted attributes...")
    gt_captions = build_item_captions(DATASET_PATH)
    
    # 4. Get Unique Shop Items from Test Split
    eval_path = os.path.join(DATASET_PATH, "Eval", "list_eval_partition.txt")
    df = pd.read_csv(eval_path, sep=r'\s+', skiprows=1)
    df.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
    
    # Group by item_id to get one shop image per unique item
    unique_items = df[df['split'] == 'test'][['item_id', 'shop_image']].drop_duplicates(subset=['item_id']).to_dict(orient='records')
    
    if args.limit:
        print(f"Limiting to first {args.limit} items for verification.")
        unique_items = unique_items[:args.limit]
        
    print(f"Total shop items to process: {len(unique_items)}")
    
    # 5. Load Output JSON if it already exists to allow resuming
    # This protects against script interruptions during long execution passes
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    captions_map = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                captions_map = json.load(f)
            print(f"Loaded {len(captions_map)} existing captions. Resuming...")
        except Exception as e:
            print("Failed to load existing output file, starting fresh.")
            
    # 6. Load Salesforce BLIP Model
    # Uses Salesforce's un-gated 1.8GB BLIP model which runs completely local
    model_id = "Salesforce/blip-image-captioning-large"
    print(f"Loading local BLIP model {model_id}...")
    
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    model.eval()
    
    processor = BlipProcessor.from_pretrained(model_id)
    
    # 7. Batch Inference Loop
    pbar = tqdm(unique_items, desc="Generating Captions")
    save_interval = 10
    unsaved_count = 0
    
    for idx, item in enumerate(pbar):
        item_id = str(item['item_id'])
        # Skip if already generated
        if item_id in captions_map:
            continue
            
        img_rel_path = item['shop_image']
        img_abs_path = os.path.join(DATASET_PATH, img_rel_path)
        
        if not os.path.exists(img_abs_path):
            # Fallback if image path does not exist
            continue
            
        # Get existing attributes string
        gt_attr_str = gt_captions.get(item_id, "")
        
        # Assemble Prompt incorporating the ground-truth attributes
        # We append the ground-truth as a prefix to guide the VLM generation
        if gt_attr_str:
            prompt = f"a photo of {gt_attr_str.replace('a ', '')},"
        else:
            prompt = "a photo of a garment,"
            
        try:
            # Load and process image
            image = Image.open(img_abs_path).convert("RGB")
            
            # Encode and generate
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40)
                description = processor.decode(out[0], skip_special_tokens=True)
                
            # Post-process description: strip, remove double spaces, keep it single line
            description = " ".join(description.strip().split())
            
            # Strip the prompt prefix if the model echoed it back
            for prefix in ["a photo of a ", "a photo of "]:
                if description.lower().startswith(prefix):
                    description = description[len(prefix):].strip()
                    break
            
            captions_map[item_id] = description
            unsaved_count += 1

            
        except Exception as e:
            # On rare failure, fallback to utilizing ground-truth caption
            captions_map[item_id] = gt_attr_str
            
        # Incremental save to disk to protect progress
        if unsaved_count >= save_interval:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(captions_map, f, indent=2, ensure_ascii=False)
            unsaved_count = 0
            
    # Final save of remaining captions
    if unsaved_count > 0:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(captions_map, f, indent=2, ensure_ascii=False)
            
    print(f"\nCaption generation complete! Captions written to {args.output}")

if __name__ == "__main__":
    main()
