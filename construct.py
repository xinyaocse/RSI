#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Neighboring Retrieval Sets for each query.
Notes:
  - Uses normalized cosine similarity (dot product on L2-normalized vectors).
  - Uses exact 2-Wasserstein (uniform measures) via optimal matching (permutations).
  - Retrieval uses FAISS if installed; otherwise falls back to brute-force dot products.
"""
import os
import json
import math
import time
import random
import itertools
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


@dataclass
class Params:
    k: int = 3
    m: int = 3
    kprime: int = 200
    M: int = 5
    A: int = 50
    N_tau: int = 200
    gamma_quantile: float = 0.80
    tau_quantile: float = 0.95
    delta_tau_frac: float = 0.10
    bucket_thresholds: Optional[List[float]] = None
    seed: int = 42


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norm, eps)


def extract_question(obj: Dict[str, Any]) -> str:
    for key in ["question", "query", "text", "prompt", "instruction"]:
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            return obj[key].strip()
    for _, v in obj.items():
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise ValueError(f"Cannot find question text in: {obj}")


def exact_w2_uniform_by_matching(A: np.ndarray, B: np.ndarray) -> float:
    k = A.shape[0]
    assert B.shape[0] == k
    D2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    best = float("inf")
    for perm in itertools.permutations(range(k)):
        s = 0.0
        for i, j in enumerate(perm):
            s += D2[i, j]
            if s >= best:
                break
        if s < best:
            best = s
    return math.sqrt(best / k)


def choose_gamma_from_pool_sims(pool_sims: np.ndarray, q: float) -> float:
    return float(np.quantile(pool_sims, q))


def bucket_index(delta: float, thresholds: List[float]) -> int:
    prev = -float("inf")
    for i, t in enumerate(thresholds, start=1):
        if prev < delta <= t:
            return i
        prev = t
    return len(thresholds) + 1


def embed_queries_with_local_model(queries: List[str], model_path: str, batch_size: int = 16) -> np.ndarray:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        raise RuntimeError(
            "transformers/torch not available. Install them (pip install torch transformers) "
            "or provide precomputed query embeddings."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch = queries[i: i + batch_size]
            enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**enc)

            attn = enc["attention_mask"].unsqueeze(-1)
            x = out.last_hidden_state * attn
            summed = x.sum(dim=1)
            denom = attn.sum(dim=1).clamp(min=1)
            emb = summed / denom
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_vecs.append(emb.cpu().numpy())
    return np.vstack(all_vecs)


class Retriever:
    def __init__(self, doc_emb: np.ndarray):
        self.doc_emb = doc_emb.astype(np.float32)
        self.N, self.dim = self.doc_emb.shape
        self.use_faiss = False
        self.index = None

        try:
            import faiss
            self.use_faiss = True
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.doc_emb)
        except Exception:
            self.use_faiss = False
            self.index = None

    def topk(self, q_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = q_emb.astype(np.float32).reshape(1, -1)
        if self.use_faiss:
            scores, idx = self.index.search(q, k)
            return scores[0], idx[0]

        scores = (self.doc_emb @ q.T).reshape(-1)
        idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return scores[idx], idx


def build_boundary_neighbors_for_query(
    q_text: str,
    q_emb: np.ndarray,
    retriever: Retriever,
    doc_emb: np.ndarray,
    chunk_ids: List[Any],
    chunk_texts: List[str],
    params: Params,
) -> Dict[str, Any]:
    pool_scores, pool_idx = retriever.topk(q_emb, params.kprime)
    C0_idx = pool_idx[: params.k]
    C0_scores = pool_scores[: params.k]

    gamma = choose_gamma_from_pool_sims(pool_scores, params.gamma_quantile)

    C0_set = set(C0_idx.tolist())
    nonC0_pool_idx = [i for i in pool_idx.tolist() if i not in C0_set]

    C0_vec = doc_emb[C0_idx]  # [k, dim]

    def select_T_boundary_seeking(m: int, L: int = 50) -> List[int]:
        score_map = {int(i): float(s) for i, s in zip(pool_idx.tolist(), pool_scores.tolist())}
        cand = [i for i in nonC0_pool_idx if score_map.get(int(i), -1.0) >= gamma]
        if len(cand) < m:
            cand = nonC0_pool_idx[:]

        cand_vec = doc_emb[cand]
        sims = cand_vec @ C0_vec.T
        conflict = sims.min(axis=1)
        order = np.argsort(conflict)

        L_eff = min(L, len(order))
        topL = order[:L_eff]

        chosen_pos = np.random.choice(topL, size=m, replace=False)
        return [cand[i] for i in chosen_pos]

    # Estimate tau_max
    deltas = []
    for _ in range(params.N_tau):
        if params.m >= params.k:
            S_idx = C0_idx.copy()
        else:
            S_idx = np.array(random.sample(C0_idx.tolist(), params.m), dtype=int)

        T_idx = np.array(select_T_boundary_seeking(len(S_idx)), dtype=int)

        if len(S_idx) == params.k:
            C1_idx = T_idx
        else:
            remaining = [i for i in C0_idx.tolist() if i not in set(S_idx.tolist())]
            C1_idx = np.array(remaining + T_idx.tolist(), dtype=int)

        sims_q_C1 = doc_emb[C1_idx] @ q_emb.reshape(-1, 1)
        if float(sims_q_C1.min()) < gamma:
            continue

        delta = exact_w2_uniform_by_matching(doc_emb[C0_idx], doc_emb[C1_idx])
        deltas.append(delta)

    if len(deltas) < 10:
        deltas = []
        for _ in range(max(params.N_tau, 200)):
            if params.m >= params.k:
                if len(nonC0_pool_idx) >= params.k:
                    C1_idx = np.array(random.sample(nonC0_pool_idx, params.k), dtype=int)
                else:
                    C1_idx = np.array(pool_idx[:params.k], dtype=int)
            else:
                S_idx = np.array(random.sample(C0_idx.tolist(), params.m), dtype=int)
                if len(nonC0_pool_idx) >= params.m:
                    T_idx = np.array(random.sample(nonC0_pool_idx, params.m), dtype=int)
                else:
                    T_idx = np.array(pool_idx[params.k: params.k + params.m], dtype=int)
                remaining = [i for i in C0_idx.tolist() if i not in set(S_idx.tolist())]
                C1_idx = np.array(remaining + T_idx.tolist(), dtype=int)

            delta = exact_w2_uniform_by_matching(doc_emb[C0_idx], doc_emb[C1_idx])
            deltas.append(delta)

    tau_max = float(np.quantile(np.array(deltas, dtype=float), params.tau_quantile))
    Delta_tau = float(params.delta_tau_frac * tau_max)

    pairs = []
    accepted = 0

    for attempt in range(params.A):
        if accepted >= params.M:
            break

        if params.m >= params.k:
            S_idx = C0_idx.copy()
        else:
            S_idx = np.array(random.sample(C0_idx.tolist(), params.m), dtype=int)

        T_idx = np.array(select_T_boundary_seeking(len(S_idx)), dtype=int)

        if len(S_idx) == params.k:
            C1_idx = T_idx
        else:
            remaining = [i for i in C0_idx.tolist() if i not in set(S_idx.tolist())]
            C1_idx = np.array(remaining + T_idx.tolist(), dtype=int)

        if len(C1_idx) != params.k or len(set(C1_idx.tolist())) != params.k:
            continue

        sims_q_C1 = doc_emb[C1_idx] @ q_emb.reshape(-1, 1)
        if float(sims_q_C1.min()) < gamma:
            continue

        delta = exact_w2_uniform_by_matching(doc_emb[C0_idx], doc_emb[C1_idx])
        if not (tau_max - Delta_tau <= delta <= tau_max):
            continue

        rec = {
            "q": q_text,
            "k": params.k,
            "m": params.m,
            "kprime": params.kprime,
            "gamma": gamma,
            "tau_max": tau_max,
            "Delta_tau": Delta_tau,
            "delta": float(delta),
            "C0": [
                {"row": int(i), "chunk_id": chunk_ids[int(i)], "text": chunk_texts[int(i)], "sim_q": float(s)}
                for i, s in zip(C0_idx.tolist(), C0_scores.tolist())
            ],
            "C1": [
                {
                    "row": int(i),
                    "chunk_id": chunk_ids[int(i)],
                    "text": chunk_texts[int(i)],
                    "sim_q": float((doc_emb[int(i)] @ q_emb).item()),
                }
                for i in C1_idx.tolist()
            ],
        }

        if params.bucket_thresholds:
            rec["bucket"] = int(bucket_index(float(delta), params.bucket_thresholds))

        pairs.append(rec)
        accepted += 1

    return {
        "q": q_text,
        "anchor": {
            "C0_rows": C0_idx.tolist(),
            "C0_chunk_ids": [chunk_ids[int(i)] for i in C0_idx.tolist()],
        },
        "accepted": accepted,
        "attempted": min(params.A, attempt + 1 if params.A > 0 else 0),
        "gamma": gamma,
        "tau_max": tau_max,
        "Delta_tau": Delta_tau,
        "pairs": pairs,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Build boundary-shell neighbors and dump accepted (C0, C1) pairs.")
    p.add_argument("--questions_file", type=str, required=True)
    p.add_argument("--embeddings_file", type=str, required=True)
    p.add_argument("--chunk_store_file", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)

    # params
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--kprime", type=int, default=200)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--A", type=int, default=50)
    p.add_argument("--N_tau", type=int, default=200)
    p.add_argument("--gamma_quantile", type=float, default=0.80)
    p.add_argument("--tau_quantile", type=float, default=0.95)
    p.add_argument("--delta_tau_frac", type=float, default=0.10)
    p.add_argument("--bucket_thresholds", type=str, default="",
                   help="Comma-separated thresholds, e.g. '0.02,0.05,0.10'. Leave empty to disable.")
    p.add_argument("--seed", type=int, default=42)

    # embedding
    p.add_argument("--batch_size", type=int, default=16)

    return p.parse_args()


def main():
    args = parse_args()

    bucket_thresholds = None
    if args.bucket_thresholds.strip():
        bucket_thresholds = [float(x) for x in args.bucket_thresholds.split(",") if x.strip()]

    params = Params(
        k=args.k,
        m=args.m,
        kprime=args.kprime,
        M=args.M,
        A=args.A,
        N_tau=args.N_tau,
        gamma_quantile=args.gamma_quantile,
        tau_quantile=args.tau_quantile,
        delta_tau_frac=args.delta_tau_frac,
        bucket_thresholds=bucket_thresholds,
        seed=args.seed,
    )

    random.seed(params.seed)
    np.random.seed(params.seed)

    # Load chunk store
    chunk_rows = read_jsonl(args.chunk_store_file)
    chunk_ids, chunk_texts = [], []
    for r in chunk_rows:
        cid = r.get("chunk_id", r.get("id", r.get("chunkid", None)))
        txt = r.get("text", r.get("chunk", r.get("content", "")))
        chunk_ids.append(cid)
        chunk_texts.append(txt)

    # Load embeddings
    doc_emb = np.load(args.embeddings_file)
    if doc_emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array, got shape {doc_emb.shape}")
    if len(chunk_ids) != doc_emb.shape[0]:
        raise ValueError(
            f"chunk_store rows ({len(chunk_ids)}) != embeddings rows ({doc_emb.shape[0]}). They must be aligned."
        )

    doc_emb = l2_normalize(doc_emb.astype(np.float32))

    # Build retriever
    retriever = Retriever(doc_emb)

    # Load questions
    q_rows = read_jsonl(args.questions_file)
    questions = [extract_question(r) for r in q_rows]

    # Embed questions
    q_embs = embed_queries_with_local_model(questions, args.model_path, batch_size=args.batch_size)
    q_embs = l2_normalize(q_embs.astype(np.float32))

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    t0 = time.time()
    with open(args.out_jsonl, "w", encoding="utf-8") as wf:
        for qi, (q_text, q_emb) in enumerate(zip(questions, q_embs)):
            res = build_boundary_neighbors_for_query(
                q_text=q_text,
                q_emb=q_emb,
                retriever=retriever,
                doc_emb=doc_emb,
                chunk_ids=chunk_ids,
                chunk_texts=chunk_texts,
                params=params,
            )
            for p in res["pairs"]:
                record = {
                    "query_index": qi,
                    "query": q_text,
                    "k": p["k"],
                    "m": p["m"],
                    "kprime": p["kprime"],
                    "gamma": p["gamma"],
                    "tau_max": p["tau_max"],
                    "Delta_tau": p["Delta_tau"],
                    "delta": p["delta"],
                    "C0": p["C0"],
                    "C1": p["C1"],
                }
                if "bucket" in p:
                    record["bucket"] = p["bucket"]
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[{qi+1:03d}/{len(questions)}] accepted={res['accepted']:3d} "
                f"attempted={res['attempted']:4d} gamma={res['gamma']:.4f} "
                f"tau_max={res['tau_max']:.4f} Delta_tau={res['Delta_tau']:.4f}"
            )

    print(f"Done. Wrote: {args.out_jsonl}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
