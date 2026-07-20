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
import json
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from tabulate import tabulate

# ── Paths ─────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EVAL_PARTITION_FILE, EMBEDDINGS_DIR
EVAL_FILE = EVAL_PARTITION_FILE

# Persisted run history, used to detect Recall@K regressions between runs.
HISTORY_FILE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_results", "history.json")
)

# ── Available model embeddings ─────────────────────────────────────────────────
MODELS = {
    "clip_base":           ("clip_queries.npy",                 "clip_gallery.npy"),
    "trained_clip":        ("trained_clip_queries.npy",         "trained_clip_gallery.npy"),
    "trained_dinov2":      ("trained_dinov2_queries.npy",       "trained_dinov2_gallery.npy"),
    "trained_resnet50":    ("trained_resnet50_queries.npy",     "trained_resnet50_gallery.npy"),
    "trained_efficientnet":("trained_efficientnet_queries.npy", "trained_efficientnet_gallery.npy"),
    "trained_siamese":     ("trained_siamese_queries.npy",      "trained_siamese_gallery.npy"),
    "trained_siamese_clip":("trained_siamese_clip_queries.npy", "trained_siamese_clip_gallery.npy"),
    "clip_text_image":     ("clip_text_image_queries.npy",      "clip_text_image_gallery.npy"),
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


def load_history() -> list:
    """All previously recorded eval runs, oldest first. Empty list if none yet."""
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def save_history(history: list) -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def last_result_for(model_name: str, history: list) -> dict | None:
    """Most recent recorded result for this model, or None if it's never been run before."""
    matches = [entry for entry in history if entry["model"] == model_name]
    return matches[-1] if matches else None


def check_regression(model_name: str, results: dict, previous: dict | None, tolerance: float) -> list:
    """
    Compares this run's Recall@K against the last recorded run for the same model.
    Returns human-readable regression messages; empty list if none (or no prior baseline exists yet).
    """
    if previous is None:
        return []
    messages = []
    for k, current_value in results.items():
        prev_value = previous["recall"].get(str(k))
        if prev_value is None:
            continue
        drop = prev_value - current_value
        if drop > tolerance:
            messages.append(
                f"{model_name} Recall@{k} dropped {drop:.1f}pp "
                f"(from {prev_value:.1f}% to {current_value:.1f}%, tolerance {tolerance:.1f}pp)"
            )
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="Model to evaluate, or 'all' (default)")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20],
                        help="K values for Recall@K (default: 1 5 10 20)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Max acceptable Recall@K drop in percentage points vs. the last "
                             "recorded run before this counts as a regression (default: 2.0)")
    args = parser.parse_args()

    print("Loading eval partition...")
    query_paths, gallery_paths, ground_truth = load_partition(EVAL_FILE)
    print(f"  Queries:  {len(query_paths)}")
    print(f"  Gallery:  {len(gallery_paths)}")
    print(f"  K values: {args.ks}")
    print()

    models_to_eval = list(MODELS.keys()) if args.model == "all" else [args.model]

    history = load_history()
    new_entries = []
    regressions = []

    rows = []
    for model_name in models_to_eval:
        print(f"Evaluating {model_name}...")
        result = evaluate_model(model_name, query_paths, gallery_paths, ground_truth, args.ks)
        if result is None:
            print("  Skipped (embeddings not found)\n")
            continue

        previous = last_result_for(model_name, history)
        regressions.extend(check_regression(model_name, result, previous, args.tolerance))
        new_entries.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "recall": {str(k): result[k] for k in args.ks},
        })

        row = [model_name] + [f"{result[k]:.1f}%" for k in args.ks]
        rows.append(row)
        print("  " + "  ".join(f"R@{k}: {result[k]:.1f}%" for k in args.ks))
        print()

    if new_entries:
        save_history(history + new_entries)

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
        print(f"Results recorded to {os.path.relpath(HISTORY_FILE)}")

    print()
    if regressions:
        print("=" * 60)
        print("REGRESSION(S) DETECTED vs. last recorded run:")
        print("=" * 60)
        for msg in regressions:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print("No regressions vs. last recorded run (or no prior baseline yet).")
        sys.exit(0)


if __name__ == "__main__":
    main()
