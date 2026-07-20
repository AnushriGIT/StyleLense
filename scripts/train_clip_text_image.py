import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import sys

# Make sure we can import from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATASET_BASE_PATH, EVAL_PARTITION_FILE
from fashion_ds_processing import CLIPShopCaptionDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of training data to use")
    return parser.parse_args()

def load_data(subset_frac=1.0):
    # Depending on format of list_eval_partition.txt, we parse the dataframe
    # DeepFashion uses spaces
    try:
        df_eval = pd.read_csv(EVAL_PARTITION_FILE, sep=r'\s+', skiprows=2, names=['consumer_image', 'shop_image', 'item_id', 'split'])
    except Exception as e:
        # Fallback if first row isn't skipped
        df_eval = pd.read_csv(EVAL_PARTITION_FILE, sep=r'\s+')
        if 'split' not in df_eval.columns:
            df_eval.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
            
    df_train = df_eval[df_eval['split'] == 'train'].reset_index(drop=True)
    if subset_frac < 1.0:
        df_train = df_train.sample(frac=subset_frac, random_state=42).reset_index(drop=True)
    return df_train

def get_collate_fn(processor):
    def collate_fn(batch):
        images, texts = zip(*batch)
        # texts might contain empty strings, handle them
        texts = [t if t else "a piece of clothing" for t in texts]
        inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
        return inputs
    return collate_fn

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    df_train = load_data(args.subset)
    print(f"Train samples: {len(df_train)}")

    dataset = CLIPShopCaptionDataset(df_train, DATASET_BASE_PATH)
    
    # 2. Init Model and Processor
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    # 3. Freeze Early Layers
    # Freeze vision model except the top layers/projection
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.vision_model.encoder.layers[-2:].parameters():
        param.requires_grad = True # unfreeze last 2 layers
    
    # Freeze text model except the top layers/projection
    for param in model.text_model.parameters():
        param.requires_grad = False
    for param in model.text_model.encoder.layers[-2:].parameters():
        param.requires_grad = True # unfreeze last 2 layers
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=get_collate_fn(processor))
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Symmetric InfoNCE loss
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # 4. Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            
            # outputs.logits_per_image: (batch_size, batch_size)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            labels = torch.arange(logits_per_image.shape[0]).to(device)
            
            l_img = loss_img(logits_per_image, labels)
            l_txt = loss_txt(logits_per_text, labels)
            loss = (l_img + l_txt) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")
        
    # 5. Save Model
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'clip_text_image_best'))
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    main()
