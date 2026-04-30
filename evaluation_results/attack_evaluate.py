#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(epsilon,q)-RSI

This script evaluates whether generated answers reveal which retrieved chunks 
were used as supporting context.
Notes:
  - This script expects your project provides `retrieval_database.get_encoder`.
  - Recommended for GitHub: install project in editable mode: pip install -e .
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from retrieval_database import get_encoder

def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = " ".join(s.split())
    return s

def normalize_vec(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize_vec(a), normalize_vec(b)))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL at line {ln}: {path}") from e

    return rows


def load_json_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")

    return data

def group_anchor_boundary(
    prompts: List[Dict[str, Any]],
    answers: List[str],
) -> Dict[int, Dict[str, Any]]:
    if len(prompts) != len(answers):
        raise ValueError(
            f"Prompts and outputs must align: prompts={len(prompts)} vs outputs={len(answers)}"
        )

    by_q: Dict[int, Dict[str, Any]] = {}

    for p, ans in zip(prompts, answers):
        if "query_index" not in p or "type" not in p:
            raise KeyError("Each prompt must contain 'query_index' and 'type' fields.")

        qi = int(p["query_index"])
        typ = str(p["type"]).strip()

        if typ not in ("anchor", "boundary"):
            continue

        contexts = p.get("contexts", [])
        if not isinstance(contexts, list):
            raise ValueError(
                f"Prompt contexts must be a list, got {type(contexts)} "
                f"(query_index={qi})"
            )

        by_q.setdefault(qi, {})
        by_q[qi][typ] = {
            "contexts": [normalize_text(x) for x in contexts],
            "answer": normalize_text(ans),
            "question": p.get("question", ""),
        }

    return by_q

def safe_auc(y_true: List[int], y_score: List[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def best_acc_from_scores(
    y_true: List[int],
    y_score: List[float],
) -> Tuple[float, float]:
    """
    Choose the threshold that gives the best attack accuracy.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    unique_scores = np.unique(y_score)
    if len(unique_scores) == 0:
        return float("nan"), float("nan")

    best_acc = -1.0
    best_threshold = unique_scores[0]

    for threshold in unique_scores:
        y_pred = (y_score >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    return float(best_acc), float(best_threshold)


def tpr_at_fpr(
    y_true: List[int],
    y_score: List[float],
    target_fpr: float,
) -> float:
    """
    Compute the maximum TPR achievable under FPR <= target_fpr.
    """
    if len(set(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid_tpr = tpr[fpr <= target_fpr]

    if len(valid_tpr) == 0:
        return 0.0

    return float(np.max(valid_tpr))


def build_chunk_mia_dataset(
    by_q: Dict[int, Dict[str, Any]],
    encoder,
) -> Tuple[List[int], List[float], List[Dict[str, Any]], Dict[str, int]]:

    text_embedding_cache: Dict[str, np.ndarray] = {}

    def embed_text(text: str) -> np.ndarray:
        if text not in text_embedding_cache:
            text_embedding_cache[text] = encoder([text])[0]
        return text_embedding_cache[text]

    all_labels: List[int] = []
    all_scores: List[float] = []
    per_query_results: List[Dict[str, Any]] = []

    num_total_queries = len(by_q)
    num_valid_queries = 0
    num_skipped_queries = 0
    num_total_candidates = 0

    for qi in sorted(by_q.keys()):
        item = by_q[qi]

        if "anchor" not in item or "boundary" not in item:
            num_skipped_queries += 1
            continue

        anchor_contexts = item["anchor"]["contexts"]
        boundary_contexts = item["boundary"]["contexts"]

        anchor_answer = item["anchor"]["answer"]
        boundary_answer = item["boundary"]["answer"]

        anchor_set = set(anchor_contexts)
        boundary_set = set(boundary_contexts)

        anchor_only = list(anchor_set - boundary_set)
        boundary_only = list(boundary_set - anchor_set)

        if len(anchor_only) == 0 or len(boundary_only) == 0:
            num_skipped_queries += 1
            continue

        num_valid_queries += 1

        anchor_answer_vec = embed_text(anchor_answer)
        boundary_answer_vec = embed_text(boundary_answer)

        query_detail = {
            "query_index": qi,
            "question": item["anchor"].get("question", ""),
            "num_anchor_only": len(anchor_only),
            "num_boundary_only": len(boundary_only),
            "anchor_samples": [],
            "boundary_samples": [],
        }

        for chunk in anchor_only:
            score = cosine(embed_text(chunk), anchor_answer_vec)

            all_labels.append(1)
            all_scores.append(score)
            num_total_candidates += 1

            query_detail["anchor_samples"].append({
                "label": 1,
                "score": score,
                "chunk": chunk,
            })

        for chunk in boundary_only:
            score = cosine(embed_text(chunk), anchor_answer_vec)

            all_labels.append(0)
            all_scores.append(score)
            num_total_candidates += 1

            query_detail["anchor_samples"].append({
                "label": 0,
                "score": score,
                "chunk": chunk,
            })

        for chunk in boundary_only:
            score = cosine(embed_text(chunk), boundary_answer_vec)

            all_labels.append(1)
            all_scores.append(score)
            num_total_candidates += 1

            query_detail["boundary_samples"].append({
                "label": 1,
                "score": score,
                "chunk": chunk,
            })

        for chunk in anchor_only:
            score = cosine(embed_text(chunk), boundary_answer_vec)

            all_labels.append(0)
            all_scores.append(score)
            num_total_candidates += 1

            query_detail["boundary_samples"].append({
                "label": 0,
                "score": score,
                "chunk": chunk,
            })

        per_query_results.append(query_detail)

    stats = {
        "num_total_queries": num_total_queries,
        "num_valid_queries": num_valid_queries,
        "num_skipped_queries": num_skipped_queries,
        "num_total_candidates": num_total_candidates,
    }

    return all_labels, all_scores, per_query_results, stats


def summarize_attack(
    labels: List[int],
    scores: List[float],
    stats: Dict[str, int],
) -> Dict[str, Any]:
    if len(labels) == 0:
        raise RuntimeError("No valid membership inference candidates found.")

    auc = safe_auc(labels, scores)
    best_acc, best_threshold = best_acc_from_scores(labels, scores)

    member_scores = [s for y, s in zip(labels, scores) if y == 1]
    nonmember_scores = [s for y, s in zip(labels, scores) if y == 0]

    summary = {
        **stats,
        "auc": auc,
        "best_acc": best_acc,
        "best_threshold": best_threshold,
        "tpr_at_fpr_0.01": tpr_at_fpr(labels, scores, 0.01),
        "positive_fraction": float(np.mean(labels)),
        "mean_score_member": (
            float(np.mean(member_scores)) if len(member_scores) > 0 else float("nan")
        ),
        "mean_score_nonmember": (
            float(np.mean(nonmember_scores)) if len(nonmember_scores) > 0 else float("nan")
        ),
    }

    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Retrieved-Chunk Membership Inference Attack."
    )

    p.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to prompts.jsonl.",
    )

    p.add_argument(
        "--outputs_file",
        type=str,
        required=True,
        help="Path to outputs.json, a JSON list aligned with prompts.",
    )

    p.add_argument(
        "--encoder_model",
        type=str,
        default="bge-large-en-v1.5",
        help="Encoder model name or local path.",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for encoder inference, e.g., cpu or cuda.",
    )

    p.add_argument(
        "--out_json",
        type=str,
        default="",
        help="Optional: save summary and per-query details to this JSON file.",
    )

    p.add_argument(
        "--save_detail",
        action="store_true",
        help="Save per-query chunk-level attack details when --out_json is provided.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    prompts_path = Path(args.prompts_file)
    outputs_path = Path(args.outputs_file)

    if not prompts_path.is_file():
        raise FileNotFoundError(f"prompts_file not found: {prompts_path}")

    if not outputs_path.is_file():
        raise FileNotFoundError(f"outputs_file not found: {outputs_path}")

    print("Loading prompts & outputs...")
    prompts = load_jsonl(prompts_path)
    answers = load_json_list(outputs_path)

    by_q = group_anchor_boundary(prompts, answers)

    print(f"Loading encoder: {args.encoder_model} on {args.device} ...")
    encoder = get_encoder(args.encoder_model, device=args.device)

    labels, scores, detail, stats = build_chunk_mia_dataset(
        by_q=by_q,
        encoder=encoder,
    )

    summary = summarize_attack(labels, scores, stats)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out_json.strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "summary": summary,
            "encoder_model": args.encoder_model,
            "device": args.device,
            "prompts_file": str(prompts_path),
            "outputs_file": str(outputs_path),
        }

        if args.save_detail:
            payload["detail"] = detail

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()