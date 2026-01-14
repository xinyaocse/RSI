# -*- coding: utf-8 -*-
"""
VDPM protection on neighborhoods.
"""

import argparse
import json
import os
from typing import List, Tuple, Optional, Dict

import numpy as np

def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed to parse JSONL at line {ln}: {path}") from e
    return rows

def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norm, eps)

def load_chunk_store(path: str):
    ids: List[object] = []
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            cid = r.get("chunk_id", r.get("id", r.get("chunkid", None)))
            txt = r.get("text", r.get("chunk", r.get("content", "")))
            ids.append(cid)
            texts.append(txt)
    return ids, texts

def noise_stats(x: np.ndarray, noisy: np.ndarray) -> Dict[str, float]:
    eta = noisy - x
    l2 = np.linalg.norm(eta, axis=1)
    l2_sq = l2 ** 2
    x_l2_sq = np.sum(x * x, axis=1) + 1e-12
    return {
        "mean_L2": float(np.mean(l2)),
        "mean_L2_sq": float(np.mean(l2_sq)),
        "mean_SNR": float(np.mean(l2_sq / x_l2_sq)),
    }

def _sample_vmf_unit(mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    dim = mu.shape[0]
    mu = mu / (np.linalg.norm(mu) + 1e-12)

    if kappa <= 1e-8:
        v = rng.normal(size=dim)
        return v / (np.linalg.norm(v) + 1e-12)

    v = rng.normal(size=dim)
    v = v - v.dot(mu) * mu
    v = v / (np.linalg.norm(v) + 1e-12)

    u = rng.uniform(0.0, 1.0)
    w = 1.0 + np.log(u + 1e-12) / (kappa + 1e-12)
    w = np.clip(w, -1.0, 1.0)
    s = np.sqrt(max(0.0, 1.0 - w ** 2))
    return w * mu + s * v

def add_vmf_noise(embeddings: np.ndarray, kappa: float, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
    X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        out[i] = _sample_vmf_unit(X[i], kappa, rng)
    return out

def map_noisy_to_corpus(
    noisy_embs: np.ndarray,
    doc_emb: np.ndarray,
    epsilon_exp: float,
    top_M: int = 50,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    N = doc_emb.shape[0]
    indices: List[int] = []
    noisy_embs = l2_normalize(noisy_embs)

    for x in noisy_embs:
        sims = doc_emb @ x  # dot = cosine (both normalized)

        if top_M is not None and top_M < N:
            cand_idx = np.argpartition(-sims, top_M)[:top_M]
            cand_sims = sims[cand_idx]
        else:
            cand_idx = np.arange(N)
            cand_sims = sims

        scores = epsilon_exp * cand_sims
        scores = scores - scores.max()
        w = np.exp(scores)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w_sum

        chosen = rng.choice(cand_idx, p=w)
        indices.append(int(chosen))

    return np.array(indices, dtype=int)

def ci_perturb_context_indices(
    ctx_rows: List[int],
    doc_emb: np.ndarray,
    kappa: float,
    epsilon_exp: float,
    top_M: int = 50,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    if len(ctx_rows) == 0:
        return np.array([], dtype=int), None

    ctx_embs = doc_emb[np.array(ctx_rows, dtype=int)]
    noisy = add_vmf_noise(ctx_embs, kappa=kappa, seed=seed)
    stats = noise_stats(ctx_embs, noisy)

    perturbed_rows = map_noisy_to_corpus(
        noisy_embs=noisy,
        doc_emb=doc_emb,
        epsilon_exp=epsilon_exp,
        top_M=top_M,
        seed=seed,
    )
    return perturbed_rows, stats

def ci_perturb_pair(
    C0_rows: List[int],
    C1_rows: List[int],
    doc_emb: np.ndarray,
    kappa: float,
    epsilon_exp: float,
    top_M: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]], Optional[Dict[str, float]]]:
    seed0 = base_seed
    seed1 = base_seed + 1

    C0_perturbed, stats0 = ci_perturb_context_indices(
        ctx_rows=C0_rows, doc_emb=doc_emb, kappa=kappa, epsilon_exp=epsilon_exp, top_M=top_M, seed=seed0
    )
    C1_perturbed, stats1 = ci_perturb_context_indices(
        ctx_rows=C1_rows, doc_emb=doc_emb, kappa=kappa, epsilon_exp=epsilon_exp, top_M=top_M, seed=seed1
    )
    return C0_perturbed, C1_perturbed, stats0, stats1

def parse_args():
    p = argparse.ArgumentParser(description="Apply VDPM protection to neighborhoods.")
    p.add_argument("--neighborhoods_file", type=str, required=True)
    p.add_argument("--embeddings_file", type=str, required=True)
    p.add_argument("--chunk_store_file", type=str, required=True)

    p.add_argument("--eps", type=float, required=True,
                   help="Nominal privacy parameter")
    p.add_argument("--kappa", type=float, default=None,
                   help="Optional override for vMF kappa.")
    p.add_argument("--epsilon_exp", type=float, default=None,
                   help="Optional override for exp mechanism epsilon.")

    p.add_argument("--top_M", type=int, default=50,
                   help="Exp mechanism samples from top_M most similar candidates (default=50).")

    p.add_argument("--seed_offset", type=int, default=1234,
                   help="Seed = query_index + seed_offset for reproducibility.")

    p.add_argument("--out_file", type=str, default="",
                   help="Output jsonl path.")

    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.neighborhoods_file):
        raise FileNotFoundError(f"neighborhoods_file not found: {args.neighborhoods_file}")
    if not os.path.isfile(args.embeddings_file):
        raise FileNotFoundError(f"embeddings_file not found: {args.embeddings_file}")
    if not os.path.isfile(args.chunk_store_file):
        raise FileNotFoundError(f"chunk_store_file not found: {args.chunk_store_file}")

    kappa = float(args.eps if args.kappa is None else args.kappa)
    epsilon_exp = float(args.eps if args.epsilon_exp is None else args.epsilon_exp)

    out_file = args.out_file.strip()
    if not out_file:
        # auto name
        base = os.path.splitext(os.path.basename(args.neighborhoods_file))[0]
        out_file = os.path.join(
            os.path.dirname(args.neighborhoods_file) or ".",
            f"{base}_ci_eps{args.eps}_top{args.top_M}.jsonl"
        )

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    # 1) load doc embeddings
    doc_emb = np.load(args.embeddings_file)
    if doc_emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {doc_emb.shape}")
    doc_emb = l2_normalize(doc_emb.astype(np.float32))

    # 2) load chunk store
    chunk_ids, chunk_texts = load_chunk_store(args.chunk_store_file)
    if len(chunk_ids) != doc_emb.shape[0]:
        raise ValueError(
            f"chunk_store rows ({len(chunk_ids)}) != embeddings rows ({doc_emb.shape[0]}). They must be aligned."
        )

    # 3) load neighborhoods
    records = read_jsonl(args.neighborhoods_file)

    n_out = 0
    with open(out_file, "w", encoding="utf-8") as wf:
        for rec in records:
            qi = int(rec["query_index"])
            q_text = rec["query"]

            C0_rows = [int(c["row"]) for c in rec["C0"]]
            C1_rows = [int(c["row"]) for c in rec["C1"]]

            base_seed = qi + int(args.seed_offset)

            C0_noisy_rows, C1_noisy_rows, stats0, stats1 = ci_perturb_pair(
                C0_rows=C0_rows,
                C1_rows=C1_rows,
                doc_emb=doc_emb,
                kappa=kappa,
                epsilon_exp=epsilon_exp,
                top_M=int(args.top_M),
                base_seed=base_seed,
            )

            C0_new = [{"row": int(r), "chunk_id": chunk_ids[int(r)], "text": chunk_texts[int(r)]}
                      for r in C0_noisy_rows.tolist()]
            C1_new = [{"row": int(r), "chunk_id": chunk_ids[int(r)], "text": chunk_texts[int(r)]}
                      for r in C1_noisy_rows.tolist()]

            out_rec = {
                "query_index": qi,
                "query": q_text,
                "k": rec.get("k"),
                "m": rec.get("m"),
                "kprime": rec.get("kprime"),
                "gamma": rec.get("gamma"),
                "tau_max": rec.get("tau_max"),
                "Delta_tau": rec.get("Delta_tau"),
                "delta": rec.get("delta"),
                "C0": C0_new,
                "C1": C1_new,
                "ci_noise": {"C0": stats0, "C1": stats1},
                "ci_params": {
                    "eps": float(args.eps),
                    "kappa": float(kappa),
                    "epsilon_exp": float(epsilon_exp),
                    "top_M": int(args.top_M),
                    "seed": int(base_seed),
                },
            }
            if "bucket" in rec:
                out_rec["bucket"] = rec["bucket"]

            wf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Done. Wrote {n_out} records to {out_file}")


if __name__ == "__main__":
    main()
