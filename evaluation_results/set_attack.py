#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Set-distinguishability attack.

Notes:
  - This script expects your project provides `retrieval_database.get_encoder`.
  - Recommended for GitHub: install project in editable mode:
      pip install -e .
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

# project import
from retrieval_database import get_encoder


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


def group_anchor_boundary(prompts: List[Dict[str, Any]], answers: List[str]) -> Dict[int, Dict[str, Any]]:
    if len(prompts) != len(answers):
        raise ValueError(f"Prompts and outputs must align: prompts={len(prompts)} vs outputs={len(answers)}")

    by_q: Dict[int, Dict[str, Any]] = {}
    for p, ans in zip(prompts, answers):
        if "query_index" not in p or "type" not in p:
            raise KeyError("Each prompt must contain 'query_index' and 'type' fields.")
        qi = int(p["query_index"])
        typ = str(p["type"]).strip()
        if typ not in ("anchor", "boundary"):
            # ignore other types if exist
            continue
        ctxs = p.get("contexts", [])
        if not isinstance(ctxs, list):
            raise ValueError(f"prompt contexts must be a list, got {type(ctxs)} (query_index={qi})")

        by_q.setdefault(qi, {})
        by_q[qi][typ] = {"contexts": ctxs, "answer": ans}

    return by_q


def bootstrap_adv(success: np.ndarray, n: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    advs = []
    for _ in range(n):
        s = rng.choice(success, size=len(success), replace=True)
        advs.append(float(s.mean() - 0.5))
    lo, med, hi = np.percentile(np.array(advs), [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def parse_args():
    p = argparse.ArgumentParser(description="Set Distinguishability Attack (context-answer matching).")
    p.add_argument("--prompts_file", type=str, required=True, help="Path to prompts.jsonl")
    p.add_argument("--outputs_file", type=str, required=True, help="Path to outputs.json (list aligned with prompts)")
    p.add_argument("--encoder_model", type=str, default="bge-large-en-v1.5", help="Encoder model name/path")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for VDPM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", type=str, default="", help="Optional: save metrics JSON here")
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

    # Load embedding model
    print(f"Loading encoder: {args.encoder_model} on {args.device} ...")
    encoder = get_encoder(args.encoder_model, device=args.device)

    success = []
    deltas = []
    used_qids = 0

    for qi in sorted(by_q.keys()):
        item = by_q[qi]
        if "anchor" not in item or "boundary" not in item:
            continue

        C0_texts = item["anchor"]["contexts"]
        C1_texts = item["boundary"]["contexts"]
        r0 = item["anchor"]["answer"] or ""
        r1 = item["boundary"]["answer"] or ""

        if len(C0_texts) == 0 or len(C1_texts) == 0:
            continue

        # Embed context sets (mean of chunks)
        C0_vec = encoder(C0_texts).mean(axis=0)
        C1_vec = encoder(C1_texts).mean(axis=0)

        # Embed answers
        r0_vec = encoder([r0])[0]
        r1_vec = encoder([r1])[0]

        # Similarities
        s00 = cosine(C0_vec, r0_vec)
        s11 = cosine(C1_vec, r1_vec)
        s01 = cosine(C0_vec, r1_vec)
        s10 = cosine(C1_vec, r0_vec)

        ok0 = (s00 > s10)
        ok1 = (s11 > s01)
        success.append(0.5 * (float(ok0) + float(ok1)))

        Delta = 0.5 * (s00 + s11) - 0.5 * (s01 + s10)
        deltas.append(float(Delta))
        used_qids += 1

    if used_qids == 0:
        raise RuntimeError("No valid (anchor, boundary) pairs found. Check prompts types and query_index alignment.")

    success = np.array(success, dtype=float)
    deltas = np.array(deltas, dtype=float)

    acc = float(success.mean())
    adv = float(2*acc - 1.0)
    mean_delta = float(deltas.mean())
    frac_delta_pos = float((deltas > 0).mean())

    ci_lo, ci_med, ci_hi = bootstrap_adv(success, n=int(args.bootstrap), seed=int(args.seed))

    print("======== Set Distinguishability Attack ========")
    print("Number of queries:", int(len(success)))
    print("Attack accuracy:", acc)
    print("Fraction Δ > 0:", frac_delta_pos)
    print("Advantage 95% CI:", [ci_lo, ci_med, ci_hi])

    if args.out_json.strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_queries": int(len(success)),
            "attack_accuracy": acc,
            "attack_advantage": adv,
            "mean_delta": mean_delta,
            "frac_delta_gt_0": frac_delta_pos,
            "advantage_ci95": [ci_lo, ci_med, ci_hi],
            "encoder_model": args.encoder_model,
            "device": args.device,
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
            "prompts_file": str(prompts_path),
            "outputs_file": str(outputs_path),
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved metrics to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
