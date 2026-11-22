from pathlib import Path
import json,os
from recare_retrieval.common import *
from scripts.reproduce_official_recare import stage_local_model_code
MODEL=Path("/root/user/cs_tcci_zhouyulin/LLM_list/jina-embeddings-v3")


def load_native(model_path=MODEL):
    import torch
    from transformers import AutoModel
    stage_local_model_code(model_path)
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if model_path != MODEL or "--" not in model_config.get("auto_map", {}).get("AutoModel", ""):
        from recare_baselines.finetuned_dense import _prime_local_remote_code_cache
        _prime_local_remote_code_cache(model_path)
    model, loading = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16,
        use_flash_attn=True, output_loading_info=True,
    )
    if model_path != MODEL and any(loading.get(key) for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs",
    )):
        raise ValueError(f"Fine-tuned checkpoint did not reload exactly: {loading}")
    model = model.eval().cuda().half()
    return model


def native_vectors(model, rows, folder, *, is_query=True):
    import numpy as np
    task = "retrieval.query" if is_query else "retrieval.passage"
    folder.mkdir(parents=True, exist_ok=True)
    output = []
    for index, row in enumerate(rows):
        path = folder / (fingerprint({"id": row["_id"], "text": row["text"]}) + ".npy")
        if path.exists():
            embedding = np.load(path)
        else:
            embedding = np.asarray(model.encode(
                [row["text"]], task=task, max_length=8192,
                truncate_dim=None, show_progress_bar=False,
            ), dtype=np.float32)
            if embedding.shape != (1, 1024) or not np.isfinite(embedding).all():
                raise ValueError("Invalid native Jina vector")
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, embedding)
            temporary.replace(path)
        if not np.allclose(np.linalg.norm(embedding, axis=1), 1, atol=1e-3):
            raise ValueError("Native vector is not normalized")
        output.append(embedding[0])
        if (index+1) % 20 == 0 or index+1 == len(rows):
            print(f"native {task}: {index+1}/{len(rows)}", flush=True)
    return np.stack(output)


def rank(qids, queries, vectors, ids, task, *, top_k=200):
    from recare_baselines.dense import _rank_inner_product
    output = []
    for start in range(0, len(qids), 32):
        batch = _rank_inner_product(
            qids[start:start+32], queries[start:start+32], vectors, ids, top_k=top_k+1,
        )
        for qid, scores in batch.items():
            pairs = sorted(scores, key=lambda p: (p[1], p[0]), reverse=True)
            ranked = [doc for doc, score in pairs if task != "rev2rev-en" or doc != qid][:top_k]
            if len(ranked) != top_k or len(set(ranked)) != top_k:
                raise ValueError("Invalid native retrieval candidate set")
            output.append({"qid": qid, "ranked": ranked})
    return output

