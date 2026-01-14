#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate generation utility by comparing method answers against reference answers.

Metrics:
  - BLEU-1 (sentence-level, averaged)
  - ROUGE-L (F1, averaged)

Input files are JSON lists of strings:
  - reference: ["...", "...", ...]
  - method:    ["...", "...", ...]
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def simple_tokenize(text: str) -> List[str]:
    return (text or "").strip().split()


def load_answers(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    out = []
    for i, x in enumerate(data):
        if x is None:
            out.append("")
        elif isinstance(x, str):
            out.append(x)
        else:
            out.append(str(x))
    return out


def eval_bleu_rouge(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    if len(refs) != len(hyps):
        raise ValueError(
            f"ref and hyp lengths differ: ref={len(refs)} vs hyp={len(hyps)}. "
            "They may be misaligned."
        )

    smooth = SmoothingFunction().method3
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu_scores = []
    rougeL_scores = []

    for ref, hyp in zip(refs, hyps):
        ref_tokens = simple_tokenize(ref)
        hyp_tokens = simple_tokenize(hyp)

        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            bleu_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue

        bleu1 = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smooth,
        )
        bleu_scores.append(float(bleu1))

        rouge = scorer.score(ref, hyp)["rougeL"]
        rougeL_scores.append(float(rouge.fmeasure))

    mean_bleu1 = sum(bleu_scores) / max(1, len(bleu_scores))
    mean_rougeL = sum(rougeL_scores) / max(1, len(rougeL_scores))

    return {"BLEU1": mean_bleu1, "ROUGE_L": mean_rougeL}


def parse_kv_list(items: List[str]) -> Dict[str, Path]:
    """
    Parse ["name=path", "name2=path2"] into dict.
    """
    out: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad --method format: '{it}'. Use name=path")
        name, p = it.split("=", 1)
        name = name.strip()
        p = p.strip()
        if not name:
            raise ValueError(f"Empty method name in: '{it}'")
        out[name] = Path(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLEU-1 and ROUGE-L for multiple methods.")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference answers JSON list")
    parser.add_argument(
        "--method",
        type=str,
        action="append",
        required=True,
        help="Method in the format name=path_to_json_list (repeatable).",
    )
    parser.add_argument("--out_json", type=str, default="", help="Optional: write results to this JSON file")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")

    methods = parse_kv_list(args.method)
    for name, p in methods.items():
        if not p.is_file():
            raise FileNotFoundError(f"Method file not found for '{name}': {p}")

    print("Loading reference answers...")
    refs = load_answers(ref_path)
    print(f"Reference count: {len(refs)}")

    all_results: Dict[str, Dict[str, float]] = {}

    for name, path in methods.items():
        print(f"\nEvaluating: {name}")
        hyps = load_answers(path)
        print(f"{name} count: {len(hyps)}")

        metrics = eval_bleu_rouge(refs, hyps)
        all_results[name] = metrics

        print(f"{name} - mean BLEU-1  : {metrics['BLEU1']:.4f}")
        print(f"{name} - mean ROUGE-L: {metrics['ROUGE_L']:.4f}")

    if args.out_json.strip():
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
