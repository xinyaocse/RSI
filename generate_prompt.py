# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple

from retrieval_database import (
    load_retrieval_database_from_parameter,
    get_encoder,
    cosine_sim,
)

def _get_topk_contexts(
    query_vec: np.ndarray,
    db_vecs: np.ndarray,
    db_texts: List[str],
    k: int = 3
) -> List[Tuple[int, float, str]]:
    q = query_vec.reshape(1, -1)
    sims = cosine_sim(q, db_vecs).reshape(-1)
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), db_texts[i]) for i in idx]

def get_contexts(
    questions: List[str],
    data_name: str,
    encoder_model_name: str,
    top_k: int = 3,
    device: str = "cpu",
    base_dir: str = "RetrievalBase",
) -> Tuple[List[Dict], List[Dict]]:
    
    db = load_retrieval_database_from_parameter(
        data_name=data_name,
        encoder_model_name=encoder_model_name,
        base_dir=base_dir,
    )
    db_vecs = db["embeddings"]
    db_texts = db["texts"]

    encoder = get_encoder(encoder_model_name, device=device)
    q_vecs = encoder(questions)

    contexts_pack = []
    retrieval_logs = []
    for qi, q in enumerate(questions):
        triples = _get_topk_contexts(q_vecs[qi], db_vecs, db_texts, k=top_k)
        contexts_pack.append({
            "question": q,
            "contexts": [{"idx": i, "score": s, "text": t} for (i, s, t) in triples]
        })
        retrieval_logs.append({
            "qid": qi,
            "question": q,
            "topk": [{"chunk_id": i, "score": s} for (i, s, _) in triples]
        })
    return contexts_pack, retrieval_logs

def _default_template(question: str, contexts: List[str]) -> str:
    ctx_block = "\n\n".join([f"[Doc{i+1}]\n{c}" for i, c in enumerate(contexts)])
    return (
        "You are a helpful medical assistant.\n"
        "Use the following retrieved documents to answer the question.\n"
        "If there is insufficient information in the context, you may say 'I don't know.'\n"
        "Do not rely on prior knowledge beyond the context.\n\n"
        f"{ctx_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def get_prompt(
    questions: List[str],
    contexts_pack: List[Dict],
    template: str = "default"
) -> List[Dict]:
    outputs = []
    for i, q in enumerate(questions):
        ctx_texts = [c["text"] for c in contexts_pack[i]["contexts"]]
       
        prompt = _default_template(q, ctx_texts)
        outputs.append({
            "question": q,
            "prompt": prompt,
            "contexts": ctx_texts
        })
    return outputs

def get_executable_file(
    prompts: List[Dict],
    save_dir: str,
    llm_model_name: str = "llama-2-7b-chat",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_gen_len: int = 256
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    p_path = os.path.join(save_dir, "prompts.jsonl")
    with open(p_path, "w", encoding="utf-8") as f:
        for obj in prompts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    cfg = {
        "llm_model_name": llm_model_name,
        "temperature": temperature,
        "top_p": top_p,
        "max_seq_len": max_seq_len,
        "max_gen_len": max_gen_len,
        "prompts_file": p_path
    }
    cfg_path = os.path.join(save_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return save_dir

def main():
    parser = argparse.ArgumentParser(description="Generate prompts with retrieval contexts.")

    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--encoder_model_name", type=str, default="bge-large-en-v1.5")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--base_dir", type=str, default="RetrievalBase")

    parser.add_argument("--questions_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--llm_model_name", type=str, default="llama-2-7b-chat")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--max_gen_len", type=int, default=256)
    args = parser.parse_args()

    questions = []
    with open(args.questions_file, "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            q = obj.get("question") or obj.get("query") or str(obj)
            questions.append(q)

    ctxs, retrieval_logs = get_contexts(
        questions=questions,
        data_name=args.data_name,
        encoder_model_name=args.encoder_model_name,
        top_k=args.top_k,
        device=args.device,
        base_dir=args.base_dir,
    )

    prompts = get_prompt(questions, ctxs, template="default")

    out_dir = get_executable_file(
        prompts=prompts,
        save_dir=args.save_dir,
        llm_model_name=args.llm_model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        max_gen_len=args.max_gen_len
    )

    log_path = os.path.join(args.save_dir, "retrieval_log.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for log in retrieval_logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[OK] prompts & run_config saved to: {out_dir}")
    print(f"[OK] retrieval log saved to: {log_path}")

if __name__ == "__main__":
    main()
