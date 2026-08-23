"""
train_clip_vlm_hybrid.py — Advanced CLIP Fine-Tuning with Hybrid Loss & Ablation Study Support.

This script implements the full 5-strategy CLIP fine-tuning pipeline designed to improve
cross-modal retrieval on the DeepFashion Consumer-to-Shop benchmark. It supports a
controlled ablation study by exposing CLI flags that enable or disable each individual
strategy independently.

Strategies implemented:
    1. Hybrid Loss (Triplet Margin + Symmetric InfoNCE):
       L_total = lambda_triplet * L_triplet(A, P, N) + lambda_infonce * L_infonce(Image, Text)
    2. Hard Negative Mining: Constrains negative sampling to the same category folder.
    3. Structured Prompt Templates: Prepends metadata tags to VLM captions.
    4. Differential Learning Rates & Layer Unfreezing:
       - Base unfrozen layers (8-9):  LR = 1e-6
       - Top unfrozen layers (10-11): LR = 5e-6
       - Projection heads:            LR = 1e-5
    5. Dynamic Text Augmentation: Randomly drops metadata tags during training.

Usage (Ablation Study):
    # Run A — baseline InfoNCE only, no advanced strategies
    .venv/bin/python scripts/train_clip_vlm_hybrid.py \\
        --run-name runA --hybrid-loss-weight 0.0 \\
        --no-hard-negatives --no-structured-prompts --no-augment

    # Run B — add Hybrid Loss
    .venv/bin/python scripts/train_clip_vlm_hybrid.py \\
        --run-name runB --no-hard-negatives --no-structured-prompts --no-augment

    # Run C — add Hard Negative Mining
    .venv/bin/python scripts/train_clip_vlm_hybrid.py \\
        --run-name runC --no-structured-prompts --no-augment

    # Run D — add Structured Prompts
    .venv/bin/python scripts/train_clip_vlm_hybrid.py \\
        --run-name runD --no-augment

    # Run E — all strategies enabled (final full model)
    .venv/bin/python scripts/train_clip_vlm_hybrid.py --run-name runE

Outputs:
    - models/clip_vlm_hybrid_<run-name>/  — saved model checkpoint (best val loss).
    - models/clip_vlm_hybrid_<run-name>.log — training loss log per epoch/step.
"""

import os
import sys
import json
import argparse
import logging
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Ensure repo root is on the path so config.py and fashion_ds_processing.py are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_BASE_PATH,
    EVAL_PARTITION_FILE,
    CATEGORY_IMG_FILE,
)
from fashion_ds_processing import (
    AdvancedCLIPHybridDataset,
    hybrid_augmented_collate_fn,
    load_item_metadata,
    build_item_captions,
)


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID = "openai/clip-vit-base-patch32"
VLM_CAPTIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "embeddings", "shop_captions.json",
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        Parsed Namespace with all training configuration parameters.
    """
    parser = argparse.ArgumentParser(
        description="Advanced CLIP Hybrid Fine-Tuning with Ablation Study Support"
    )
    # ── Run identity ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--run-name", default="runE",
        help="Name for this ablation run (e.g. runA, runB, ..., runE). "
             "Determines checkpoint directory name.",
    )
    # ── Training hyperparameters ───────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0.05 for quick test).")
    # ── Loss weights ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--hybrid-loss-weight", type=float, default=1.0,
        help="Weight λ₁ on Triplet Margin Loss term. Set to 0.0 to use InfoNCE only (Run A).",
    )
    parser.add_argument(
        "--infonce-loss-weight", type=float, default=0.5,
        help="Weight λ₂ on Symmetric InfoNCE (text-image alignment) loss term.",
    )
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    # ── Strategy feature flags (ablation) ─────────────────────────────────────
    parser.add_argument(
        "--no-hard-negatives", action="store_true",
        help="Disable hard negative mining (Strategy 2). Use random negatives instead.",
    )
    parser.add_argument(
        "--no-structured-prompts", action="store_true",
        help="Disable structured prompt templates (Strategy 3). Use plain VLM captions.",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable dynamic text augmentation (Strategy 5).",
    )
    parser.add_argument("--augment-prob", type=float, default=0.20)
    # ── Layer unfreezing (Strategy 4) ─────────────────────────────────────────
    parser.add_argument(
        "--unfreeze-top-n", type=int, default=4,
        help="Number of top ViT transformer blocks to unfreeze in vision encoder (default 4).",
    )
    parser.add_argument(
        "--unfreeze-text-top-n", type=int, default=None,
        help="Number of top transformer blocks to unfreeze in text encoder (default: matches unfreeze-top-n).",
    )
    return parser.parse_args()


def setup_model_and_optimizer(args, device: str):
    """
    Load CLIP model, apply differential layer unfreezing (Strategy 4), and
    build an AdamW optimizer with three separate learning rate groups.

    Unfreezing groups:
        Group 1 — Base unfrozen ViT blocks (indices N-4 to N-3):  LR = 1e-6
        Group 2 — Top unfrozen ViT blocks  (indices N-2 to N-1):  LR = 5e-6
        Group 3 — Projection heads + post-layernorm:               LR = 1e-5

    Args:
        args: Parsed CLI arguments.
        device: PyTorch device string ("mps", "cuda", or "cpu").

    Returns:
        Tuple of (model, optimizer, processor, tokenizer).
    """
    model = AutoModel.from_pretrained(MODEL_ID).to(device)

    # ── Step 1: Freeze everything ──────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = False

    num_layers = 12  # ViT-B/32 has 12 transformer blocks
    vision_top_n = args.unfreeze_top_n
    text_top_n   = args.unfreeze_text_top_n if args.unfreeze_text_top_n is not None else vision_top_n

    base_params, top_params, proj_params = [], [], []

    # ── Step 2: Unfreeze & assign param groups for vision encoder ──────────────
    v_base_start = num_layers - vision_top_n
    v_base_indices = list(range(v_base_start, v_base_start + vision_top_n // 2))
    v_top_indices  = list(range(v_base_start + vision_top_n // 2, num_layers))

    for i in v_base_indices:
        for p in model.vision_model.encoder.layers[i].parameters():
            p.requires_grad = True
            base_params.append(p)
    for i in v_top_indices:
        for p in model.vision_model.encoder.layers[i].parameters():
            p.requires_grad = True
            top_params.append(p)

    ln_v = getattr(model.vision_model, "post_layernorm", None) or getattr(model.vision_model, "final_layer_norm", None)
    if ln_v is not None:
        for p in ln_v.parameters():
            p.requires_grad = True
            proj_params.append(p)

    # ── Step 3: Unfreeze & assign param groups for text encoder ───────────────
    t_base_start = num_layers - text_top_n
    t_base_indices = list(range(t_base_start, t_base_start + text_top_n // 2))
    t_top_indices  = list(range(t_base_start + text_top_n // 2, num_layers))

    for i in t_base_indices:
        for p in model.text_model.encoder.layers[i].parameters():
            p.requires_grad = True
            base_params.append(p)
    for i in t_top_indices:
        for p in model.text_model.encoder.layers[i].parameters():
            p.requires_grad = True
            top_params.append(p)

    ln_t = getattr(model.text_model, "post_layernorm", None) or getattr(model.text_model, "final_layer_norm", None)
    if ln_t is not None:
        for p in ln_t.parameters():
            p.requires_grad = True
            proj_params.append(p)

    # ── Step 3: Unfreeze projection heads ─────────────────────────────────────
    model.visual_projection.requires_grad_(True)
    model.text_projection.requires_grad_(True)
    proj_params += list(model.visual_projection.parameters())
    proj_params += list(model.text_projection.parameters())

    # ── Step 4: Build differential optimizer ──────────────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": 1e-6, "weight_decay": 0.01},
            {"params": top_params,  "lr": 5e-6, "weight_decay": 0.01},
            {"params": proj_params, "lr": 1e-5, "weight_decay": 0.01},
        ]
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.1f}%) — "
        f"vision_unfreeze={v_base_indices}+{v_top_indices}, text_unfreeze={t_base_indices}+{t_top_indices}"
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, optimizer, processor, tokenizer


def compute_hybrid_loss(
    model,
    a_pixels: torch.Tensor,
    p_pixels: torch.Tensor,
    n_pixels: torch.Tensor,
    text_inputs: dict,
    device: str,
    triplet_margin: float,
    lambda_triplet: float,
    lambda_infonce: float,
):
    """
    Compute the combined Hybrid Loss for one training batch.

    L_Total = λ₁ × L_Triplet(anchor, positive, negative)
            + λ₂ × L_InfoNCE_symmetric(positive_image, text)

    Triplet Margin Loss (Strategy 1 — visual discrimination):
        Maximises the distance between (anchor, negative) relative to (anchor, positive).
        Uses L2-squared distances on L2-normalised embeddings so all distances
        live in [0, 4] and the margin is well-conditioned.

    Symmetric InfoNCE / Contrastive Loss (Strategy 1 — text-image alignment):
        Forces the CLIP vision and text embeddings to stay jointly calibrated.
        Image→Text and Text→Image cross-entropy losses are averaged.

    Args:
        model: The CLIP AutoModel instance.
        a_pixels, p_pixels, n_pixels: ``pixel_values`` tensors of shape (B, 3, 224, 224).
        text_inputs: Dict with ``input_ids`` and ``attention_mask`` tensors (B, L).
        device: PyTorch device string.
        triplet_margin: Scalar margin m for Triplet Margin Loss.
        lambda_triplet: Loss weight for triplet term (set to 0 for InfoNCE-only Run A).
        lambda_infonce: Loss weight for InfoNCE term.

    Returns:
        Dict with keys:
            "total":   Combined scalar loss (requires_grad=True).
            "triplet": Triplet margin loss value (float).
            "infonce": Symmetric InfoNCE loss value (float).
    """
    # ── Extract visual embeddings for the triplet ──────────────────────────────
    emb_a = model.get_image_features(pixel_values=a_pixels.to(device))
    emb_p = model.get_image_features(pixel_values=p_pixels.to(device))
    emb_n = model.get_image_features(pixel_values=n_pixels.to(device))

    if hasattr(emb_a, "pooler_output") and emb_a.pooler_output is not None:
        emb_a = emb_a.pooler_output
    elif hasattr(emb_a, "logits"):
        emb_a = emb_a.logits
    if hasattr(emb_p, "pooler_output") and emb_p.pooler_output is not None:
        emb_p = emb_p.pooler_output
    elif hasattr(emb_p, "logits"):
        emb_p = emb_p.logits
    if hasattr(emb_n, "pooler_output") and emb_n.pooler_output is not None:
        emb_n = emb_n.pooler_output
    elif hasattr(emb_n, "logits"):
        emb_n = emb_n.logits

    # L2-normalise so distances live in [0, 4]
    emb_a = F.normalize(emb_a, p=2, dim=-1)
    emb_p = F.normalize(emb_p, p=2, dim=-1)
    emb_n = F.normalize(emb_n, p=2, dim=-1)

    # Triplet Margin Loss: max(0, d(A,P)^2 - d(A,N)^2 + margin)
    dist_ap = torch.sum((emb_a - emb_p) ** 2, dim=-1)  # (B,)
    dist_an = torch.sum((emb_a - emb_n) ** 2, dim=-1)  # (B,)
    loss_triplet = F.relu(dist_ap - dist_an + triplet_margin).mean()

    # ── Symmetric InfoNCE (text-image alignment) ───────────────────────────────
    emb_t = model.get_text_features(
        input_ids=text_inputs["input_ids"].to(device),
        attention_mask=text_inputs["attention_mask"].to(device),
    )
    if hasattr(emb_t, "pooler_output") and emb_t.pooler_output is not None:
        emb_t = emb_t.pooler_output
    elif hasattr(emb_t, "logits"):
        emb_t = emb_t.logits
    emb_t = F.normalize(emb_t, p=2, dim=-1)

    # Temperature-scaled logit matrix between positive images and their captions
    logit_scale = model.logit_scale.exp()
    logits_v2t = logit_scale * emb_p @ emb_t.T   # (B, B)
    logits_t2v = logit_scale * emb_t @ emb_p.T   # (B, B)

    labels = torch.arange(emb_p.size(0), device=device)
    loss_infonce = 0.5 * (
        F.cross_entropy(logits_v2t, labels) + F.cross_entropy(logits_t2v, labels)
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    loss_total = lambda_triplet * loss_triplet + lambda_infonce * loss_infonce

    return {
        "total":   loss_total,
        "triplet": loss_triplet.item(),
        "infonce": loss_infonce.item(),
    }


def load_training_data(args):
    """
    Load DeepFashion training split, item metadata, captions, and VLM captions.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Tuple of (df_train, item_to_clothes_type, item_to_category_folder,
                  item_captions, vlm_captions).
    """
    import pandas as pd
    import re

    logging.info("Loading data from dataset...")

    # ── Parse the evaluation partition file ───────────────────────────────────
    df_eval = pd.read_csv(EVAL_PARTITION_FILE, sep=r"\s+", skiprows=1)
    df_eval.columns = ["consumer_image", "shop_image", "item_id", "split"]

    df_train = df_eval[df_eval["split"] == "train"].reset_index(drop=True)

    if args.data_fraction < 1.0:
        df_train = df_train.sample(frac=args.data_fraction, random_state=42).reset_index(drop=True)
        logging.info(f"Using {args.data_fraction*100:.0f}% subset: {len(df_train)} rows.")
    else:
        logging.info(f"Full training set: {len(df_train)} rows.")

    # ── Load metadata and captions ─────────────────────────────────────────────
    item_to_clothes_type, item_to_category_folder = load_item_metadata(DATASET_BASE_PATH)
    item_captions = build_item_captions(DATASET_BASE_PATH)

    vlm_captions = {}
    if os.path.exists(VLM_CAPTIONS_FILE):
        with open(VLM_CAPTIONS_FILE, "r") as f:
            vlm_captions = json.load(f)
        logging.info(f"Loaded {len(vlm_captions)} VLM captions from shop_captions.json.")
    else:
        logging.warning("shop_captions.json not found — falling back to attribute-only captions.")

    return df_train, item_to_clothes_type, item_to_category_folder, item_captions, vlm_captions


def run_training(args):
    """
    Main training loop for the ablation study experiment.

    Orchestrates data loading, model setup, training, and checkpoint saving.
    Logs per-step and per-epoch losses (total, triplet, infonce) to a file
    rather than stdout to comply with project Context Budgeting rules.

    Args:
        args: Parsed CLI arguments from parse_args().
    """
    # ── Device selection ───────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info(f"Using device: {device}")

    # ── Output paths ──────────────────────────────────────────────────────────
    output_dir = os.path.join(MODELS_DIR, f"clip_vlm_hybrid_{args.run_name}")
    os.makedirs(output_dir, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────
    df_train, item_to_clothes_type, item_to_category_folder, item_captions, vlm_captions = (
        load_training_data(args)
    )

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    # Load processor and tokenizer early so we can pass them to the collate fn
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    dataset = AdvancedCLIPHybridDataset(
        df=df_train,
        dataset_root=DATASET_BASE_PATH,
        item_captions=item_captions,
        item_to_clothes_type=item_to_clothes_type,
        item_to_category_folder=item_to_category_folder,
        vlm_captions=vlm_captions,
        use_hard_negatives=not args.no_hard_negatives,
        use_structured_prompts=not args.no_structured_prompts,
        augment_text=not args.no_augment,
        augment_prob=args.augment_prob,
    )

    collate = partial(hybrid_augmented_collate_fn, processor=processor, tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    model, optimizer, processor, tokenizer = setup_model_and_optimizer(args, device)

    # ── Log run configuration ─────────────────────────────────────────────────
    config_summary = {
        "run_name": args.run_name,
        "strategies": {
            "1_hybrid_loss": args.hybrid_loss_weight > 0,
            "2_hard_negatives": not args.no_hard_negatives,
            "3_structured_prompts": not args.no_structured_prompts,
            "4_differential_lr": True,    # always applied
            "5_text_augmentation": not args.no_augment,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lambda_triplet": args.hybrid_loss_weight,
            "lambda_infonce": args.infonce_loss_weight,
            "margin": args.triplet_margin,
            "unfreeze_top_n": args.unfreeze_top_n,
        },
        "training_samples": len(dataset),
        "device": device,
    }
    logging.info(f"Run config: {json.dumps(config_summary, indent=2)}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    best_avg_loss = float("inf")
    total_steps = len(loader)

    for epoch in range(1, args.epochs + 1):
        epoch_loss_total   = 0.0
        epoch_loss_triplet = 0.0
        epoch_loss_infonce = 0.0

        for step, (a, p, n, t, _captions) in enumerate(loader):
            optimizer.zero_grad()

            losses = compute_hybrid_loss(
                model=model,
                a_pixels=a["pixel_values"],
                p_pixels=p["pixel_values"],
                n_pixels=n["pixel_values"],
                text_inputs=t,
                device=device,
                triplet_margin=args.triplet_margin,
                lambda_triplet=args.hybrid_loss_weight,
                lambda_infonce=args.infonce_loss_weight,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            epoch_loss_total   += losses["total"].item()
            epoch_loss_triplet += losses["triplet"]
            epoch_loss_infonce += losses["infonce"]

            # Log every 10 steps (not every step to avoid log spam)
            if step % 10 == 0:
                logging.info(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{total_steps}] "
                    f"Total: {losses['total'].item():.4f} "
                    f"Triplet: {losses['triplet']:.4f} "
                    f"InfoNCE: {losses['infonce']:.4f}"
                )

        avg_total   = epoch_loss_total   / total_steps
        avg_triplet = epoch_loss_triplet / total_steps
        avg_infonce = epoch_loss_infonce / total_steps

        logging.info(
            f"Epoch [{epoch}/{args.epochs}] AVG — "
            f"Total: {avg_total:.4f} | Triplet: {avg_triplet:.4f} | InfoNCE: {avg_infonce:.4f}"
        )

        # Save best checkpoint based on total loss
        if avg_total < best_avg_loss:
            best_avg_loss = avg_total
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logging.info(f"✅ New best model saved to {output_dir} (loss={avg_total:.4f})")

    logging.info(f"Training complete. Best loss: {best_avg_loss:.4f}")
    logging.info(f"Model saved to {output_dir}")


def main():
    """Entry point: parse args, set up logging, run training."""
    args = parse_args()

    # Pipe ALL output to a log file (Rule #3: Context Budgeting — no verbose stdout)
    log_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        f"clip_vlm_hybrid_{args.run_name}.log",
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),  # also echo to stdout for live monitoring
        ],
    )
    logging.info(f"Logging to: {log_file}")

    run_training(args)


if __name__ == "__main__":
    main()
