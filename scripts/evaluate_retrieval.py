"""
evaluate_retrieval.py — Recall@K evaluation for consumer-to-shop retrieval.

Uses precomputed embeddings (no model inference needed).
Evaluates: trained_clip, clip (base), trained_dinov2, trained_resnet50, trained_efficientnet

Run:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --model trained_clip
    python evaluate_retrieval.py --model trained_dinov2 --ks 1 5 10 20
"""

import os
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate

# ── Paths ─────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVAL_PARTITION_FILE, EMBEDDINGS_DIR
EVAL_FILE = EVAL_PARTITION_FILE

# ── Available model embeddings ─────────────────────────────────────────────────
MODELS = {
    "clip_base":           ("clip_queries.npy",                 "clip_gallery.npy"),
    "trained_clip":        ("trained_clip_queries.npy",         "trained_clip_gallery.npy"),
    "trained_dinov2":      ("trained_dinov2_queries.npy",       "trained_dinov2_gallery.npy"),
    "trained_resnet50":    ("trained_resnet50_queries.npy",     "trained_resnet50_gallery.npy"),
    "trained_efficientnet":("trained_efficientnet_queries.npy", "trained_efficientnet_gallery.npy"),
    "trained_siamese":     ("trained_siamese_queries.npy",      "trained_siamese_gallery.npy"),
    "trained_siamese_clip":("trained_siamese_clip_queries.npy", "trained_siamese_clip_gallery.npy"),
}


def load_partition(eval_file: str):
    """
    Loads the DeepFashion eval partition.
    Returns (query_paths, gallery_paths, query_to_gallery_idx)
    where query_to_gallery_idx[i] = index in gallery_paths that matches query i.
    """
    df = pd.read_csv(eval_file, sep=r'\s+', skiprows=1)
    df.columns = ['consumer_image', 'shop_image', 'item_id', 'split']
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    gallery_paths = df_test['shop_image'].unique().tolist()
    gallery_idx   = {p: i for i, p in enumerate(gallery_paths)}

    query_paths          = df_test['consumer_image'].tolist()
    query_gt_shop        = df_test['shop_image'].tolist()
    query_to_gallery_idx = [gallery_idx[s] for s in query_gt_shop]

    return query_paths, gallery_paths, query_to_gallery_idx


def recall_at_k(query_emb: np.ndarray, gallery_emb: np.ndarray,
                ground_truth: list, ks: list) -> dict:
    """
    Computes Recall@K for all K values in ks.

    For each query, ranks all gallery items by cosine similarity.
    Checks if the ground truth gallery index is in the top K.
    """
    # Cosine similarity matrix: (n_queries, n_gallery)
    # Both embeddings are L2-normalised so dot product = cosine similarity
    scores  = query_emb @ gallery_emb.T        # (Q, G)
    ranks   = np.argsort(-scores, axis=1)      # descending

    results = {}
    for k in ks:
        top_k  = ranks[:, :k]                  # (Q, k)
        # Check if ground truth index appears in top K for each query
        hits   = [gt in top_k[i] for i, gt in enumerate(ground_truth)]
        results[k] = np.mean(hits) * 100       # percentage

    return results


def l2_normalize(emb: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norm, 1e-8)


def evaluate_model(model_name: str, query_paths: list, gallery_paths: list,
                   ground_truth: list, ks: list) -> dict | None:
    query_file, gallery_file = MODELS[model_name]
    qpath = os.path.join(EMBEDDINGS_DIR, query_file)
    gpath = os.path.join(EMBEDDINGS_DIR, gallery_file)

    if not os.path.exists(qpath) or not os.path.exists(gpath):
        return None

    query_emb   = l2_normalize(np.load(qpath).astype(np.float32))
    gallery_emb = l2_normalize(np.load(gpath).astype(np.float32))

    # Sanity check: embedding count must match partition
    if query_emb.shape[0] != len(query_paths):
        print(f"  WARNING {model_name}: query embedding count {query_emb.shape[0]} "
              f"!= partition query count {len(query_paths)} — skipping")
        return None
    if gallery_emb.shape[0] != len(gallery_paths):
        print(f"  WARNING {model_name}: gallery embedding count {gallery_emb.shape[0]} "
              f"!= partition gallery count {len(gallery_paths)} — skipping")
        return None

    return recall_at_k(query_emb, gallery_emb, ground_truth, ks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="Model to evaluate, or 'all' (default)")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20],
                        help="K values for Recall@K (default: 1 5 10 20)")
    args = parser.parse_args()

    print("Loading eval partition...")
    query_paths, gallery_paths, ground_truth = load_partition(EVAL_FILE)
    print(f"  Queries:  {len(query_paths)}")
    print(f"  Gallery:  {len(gallery_paths)}")
    print(f"  K values: {args.ks}")
    print()

    models_to_eval = list(MODELS.keys()) if args.model == "all" else [args.model]

    rows = []
    for model_name in models_to_eval:
        print(f"Evaluating {model_name}...")
        result = evaluate_model(model_name, query_paths, gallery_paths, ground_truth, args.ks)
        if result is None:
            print("  Skipped (embeddings not found)\n")
            continue
        row = [model_name] + [f"{result[k]:.1f}%" for k in args.ks]
        rows.append(row)
        print("  " + "  ".join(f"R@{k}: {result[k]:.1f}%" for k in args.ks))
        print()

    if rows:
        headers = ["Model"] + [f"Recall@{k}" for k in args.ks]
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        try:
            print(tabulate(rows, headers=headers, tablefmt="github"))
        except ImportError:
            print("\t".join(headers))
            for r in rows:
                print("\t".join(r))
        print()
        print("Recall@K = % of queries where correct shop image is in top K results")


if __name__ == "__main__":
    main()
