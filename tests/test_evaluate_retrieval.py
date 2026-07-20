"""
Tests for scripts/evaluate_retrieval.py — the Recall@K math and the v5 regression-tracking
additions (history persistence, baseline comparison, tolerance gating).
"""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import evaluate_retrieval as er


# ── recall_at_k ───────────────────────────────────────────────────────────────

def test_recall_at_k_perfect_match():
    # Query i's embedding is identical to gallery row i => ground truth always rank 1.
    query_emb = np.eye(3)
    gallery_emb = np.eye(3)
    ground_truth = [0, 1, 2]
    result = er.recall_at_k(query_emb, gallery_emb, ground_truth, ks=[1])
    assert result[1] == 100.0


def test_recall_at_k_zero_when_never_in_top_k():
    query_emb = np.array([[1.0, 0.0, 0.0]])
    # Ground truth is index 2, but it's the worst match to the query.
    gallery_emb = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    result = er.recall_at_k(query_emb, gallery_emb, ground_truth=[2], ks=[1])
    assert result[1] == 0.0


# ── check_regression (v5) ─────────────────────────────────────────────────────

def test_check_regression_no_previous_baseline_returns_empty():
    assert er.check_regression("clip_base", {1: 50.0}, previous=None, tolerance=2.0) == []


def test_check_regression_within_tolerance_returns_empty():
    previous = {"recall": {"1": 50.0}}
    messages = er.check_regression("clip_base", {1: 49.0}, previous, tolerance=2.0)
    assert messages == []


def test_check_regression_beyond_tolerance_flags_it():
    previous = {"recall": {"1": 50.0}}
    messages = er.check_regression("clip_base", {1: 40.0}, previous, tolerance=2.0)
    assert len(messages) == 1
    assert "clip_base" in messages[0]
    assert "Recall@1" in messages[0]


def test_check_regression_improvement_is_not_flagged():
    previous = {"recall": {"1": 40.0}}
    messages = er.check_regression("clip_base", {1: 90.0}, previous, tolerance=2.0)
    assert messages == []


# ── last_result_for ────────────────────────────────────────────────────────────

def test_last_result_for_returns_most_recent_matching_model():
    history = [
        {"model": "clip_base", "recall": {"1": 10.0}},
        {"model": "trained_dinov2", "recall": {"1": 20.0}},
        {"model": "clip_base", "recall": {"1": 30.0}},
    ]
    result = er.last_result_for("clip_base", history)
    assert result["recall"]["1"] == 30.0


def test_last_result_for_no_match_returns_none():
    assert er.last_result_for("nonexistent_model", []) is None


# ── load_history / save_history round-trip (v5) ───────────────────────────────

def test_history_round_trips_through_save_and_load(tmp_path, monkeypatch):
    fake_history_file = tmp_path / "eval_results" / "history.json"
    monkeypatch.setattr(er, "HISTORY_FILE", str(fake_history_file))

    assert er.load_history() == []  # no file yet

    entries = [{"model": "clip_base", "timestamp": "t1", "recall": {"1": 42.0}}]
    er.save_history(entries)

    assert fake_history_file.exists()
    assert er.load_history() == entries


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
