import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Callable
from datetime import datetime


def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None


def _try_import_flag_embedding():
    try:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel
    except Exception:
        return None


def get_encoding_of_file(path: str) -> str:
    return "utf-8"


def get_encoder(encoder_model_name: str, device: str = "cpu"):
    ST = _try_import_sentence_transformers()
    BGEM = _try_import_flag_embedding()

    alias_map = {
        "bge-large-en-v1.5": os.path.join("Model", "bge-large-en-v1.5"),
        "e5-base-v2": os.path.join("Model", "e5-base-v2"),
    }
    model_path = alias_map.get(encoder_model_name, encoder_model_name)

    if ST is not None:
        try:
            model = ST(model_path, device=device)
            return lambda texts: np.asarray(model.encode(texts, normalize_embeddings=True))
        except Exception:
            pass

    if BGEM is not None and "bge" in encoder_model_name:
        try:
            model = BGEM(model_path, use_fp16=True)

            def _enc(texts: List[str]) -> np.ndarray:
                vecs = model.encode(texts)["dense_vecs"]
                vecs = np.asarray(vecs)
                vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
                return vecs

            return _enc
        except Exception:
            pass

    
    def _cheap_hash_vec(t: str, d: int = 256) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        v = rng.normal(size=d)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v

    return lambda texts: np.stack([_cheap_hash_vec(t) for t in texts], axis=0)


def _load_raw_texts_from_data_root(data_name: str) -> List[str]:
    path = os.path.join("Data", f"{data_name}", "chatdoctor.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r", encoding=get_encoding_of_file(path)) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    merged_texts = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.lower().startswith("input:") and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.lower().startswith("output:"):
                merged = f"{line}\n{next_line}"
                merged_texts.append(merged)
                i += 2
                continue
        merged_texts.append(line)
        i += 1

    
    return list(dict.fromkeys(merged_texts))


def _build_save_dir(
    base_dir: str,
    data_name: str,
    encoder_model_name: str,
) -> str:
    
    sub = "original"
    save_dir = os.path.join(base_dir, sub, data_name, encoder_model_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _save_index(save_dir: str, embeddings: np.ndarray, texts: List[str], meta: Dict):
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings.astype("float32"))

    with open(os.path.join(save_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_index(load_dir: str) -> Tuple[np.ndarray, List[str], Dict]:
    embs = np.load(os.path.join(load_dir, "embeddings.npy"))

    texts = []
    with open(os.path.join(load_dir, "texts.jsonl"), "r", encoding="utf-8") as f:
        for ln in f:
            obj = json.loads(ln)
            texts.append(obj["text"])

    meta = {}
    meta_p = os.path.join(load_dir, "meta.json")
    if os.path.isfile(meta_p):
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.loads(f.read())

    return embs, texts, meta


def construct_retrieval_database(
    data_name: str,
    encoder_model_name: str,
    device: str = "cpu",
    seed: int = 42,
    base_dir: str = "RetrievalBase",
) -> str:
    # Note: seed kept only for meta reproducibility
    texts = _load_raw_texts_from_data_root(data_name)
    encoder = get_encoder(encoder_model_name, device=device)
    embeddings = encoder(texts)

    save_dir = _build_save_dir(base_dir, data_name, encoder_model_name)

    meta = {
        "data_name": data_name,
        "encoder_model_name": encoder_model_name,
        "device": device,
        "perturb_type": "none",
        "seed": seed,
        "built_at": datetime.now().isoformat(),
        "num_chunks": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
    }

    _save_index(save_dir, embeddings, texts, meta)
    return save_dir


def load_retrieval_database_from_address(address: str):
    embs, texts, meta = _load_index(address)
    return {"embeddings": embs, "texts": texts, "meta": meta, "path": address}


def load_retrieval_database_from_parameter(
    data_name: str,
    encoder_model_name: str,
    base_dir: str = "RetrievalBase",
):
    load_dir = _build_save_dir(base_dir, data_name, encoder_model_name)
    return load_retrieval_database_from_address(load_dir)


def build_query_transform_from_meta(meta: Dict, base_path: str) -> Callable[[np.ndarray], np.ndarray]:
    
    return lambda X: X


def main():
    parser = argparse.ArgumentParser(description="Build retrieval base.")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--encoder_model_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_dir", type=str, default="RetrievalBase")
    args = parser.parse_args()

    save_dir = construct_retrieval_database(
        data_name=args.data_name,
        encoder_model_name=args.encoder_model_name,
        device=args.device,
        seed=args.seed,
        base_dir=args.base_dir,
    )
    print(f"Baseline index built and saved to {save_dir}")


if __name__ == "__main__":
    main()
