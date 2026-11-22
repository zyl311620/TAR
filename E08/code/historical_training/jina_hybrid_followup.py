from pathlib import Path
import json,os,math,sys
from dataclasses import asdict
from contextlib import contextmanager
from recare_retrieval.common import *
from recare_retrieval.lexical import BM25,reciprocal_rank_fusion
from scripts.joint_jina_gpu import MODEL,load_native,native_vectors,rank
WORKSPACE=Path(__file__).resolve().parents[3]
RECOVERY=WORKSPACE/'runs/finetune/jina_v3_recovery_v2'
ROOT=RECOVERY/'hybrid_top50_fulltext'
REWRITES=WORKSPACE/'runs/llm_experiments/qwen35_v1/rewrite'
VARIANTS=('pretrained','finetuned')


@contextmanager
def model_loading_guard():
    """Serialize large loads across our workers without touching other users' jobs."""
    import fcntl
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "model_loading.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluate_stage(path, task):
    from recare_baselines.ranked_eval import evaluate_ranked_files
    result = evaluate_ranked_files(
        path, WORKSPACE / "ReCaRe" / f"qrels-{task}/test.jsonl", filter_self=task == "rev2rev-en")
    expected = 113 if task == "rat2rev-en" else 503
    if result.n_queries != expected or result.missing_qids or result.extra_qids:
        raise ValueError("Hybrid stage does not cover the complete official test split")
    write_json(path.with_suffix(".official_metrics.json"), asdict(result))
    return {key: result.metrics[key] for key in ("ndcg_10", "recall_100")}


def retrieve(task):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    corpus_path = WORKSPACE / "ReCaRe/corpus-en/corpus.jsonl"
    manifests = [ROOT / v / "retrieval" / task / "manifest.json" for v in VARIANTS]
    if all(p.exists() and json.loads(p.read_text()).get("complete") for p in manifests):
        selection = json.loads((RECOVERY / "final_selection.json").read_text())
        checkpoints = (MODEL, Path(selection["selected_checkpoint"]))
        index_paths = (
            WORKSPACE / "runs/official_recare/native-jina-v3-rat2rev-en-test/native_indexes/jina-v3/en/embeddings.npy",
            checkpoints[1] / "evaluation/native_index/embeddings.npy")
        for path, checkpoint, index_path in zip(manifests, checkpoints, index_paths):
            old = json.loads(path.read_text())
            expected = {
                "corpus_sha256": file_hash(corpus_path),
                "rewrites_sha256": file_hash(REWRITES / task / "rewrites.jsonl"),
                "original_queries_sha256": file_hash(WORKSPACE / "data/queries" / task / "test.jsonl"),
                "embedding_weights_sha256": file_hash(checkpoint / "model.safetensors"),
                "passage_index_sha256": file_hash(index_path),
                "dense_top_k": 200, "bm25_top_k": 200, "fusion_top_k": 200,
                "bm25_k1": 1.2, "bm25_b": .75, "rrf_k": 60, "weights": [1., 1.],
            }
            if (any(old["identity"][k] != value for k, value in expected.items())
                    or fingerprint(old["identity"]) != old["identity_hash"]
                    or any(file_hash(path.parent / name) != sha for name, sha in old["run_sha256"].items())):
                raise ValueError("Completed retrieval provenance changed")
        print("REUSE_COMPLETE_PAIRED_RETRIEVAL " + task, flush=True)
        return
    corpus = list(read_jsonl(corpus_path))
    ids = [r["_id"] for r in corpus]
    positions = {doc: i for i, doc in enumerate(ids)}
    original = unique_ids(WORKSPACE / "data/queries" / task / "test.jsonl")
    rewrite_path = REWRITES / task / "rewrites.jsonl"
    rewritten = list(read_jsonl(rewrite_path))
    rewrite_manifest = json.loads((REWRITES / task / "manifest.json").read_text())
    if (rewrite_manifest["status"] != "complete"
            or rewrite_manifest["result_sha256"] != file_hash(rewrite_path)
            or set(original) != {r["_id"] for r in rewritten}
            or any(r["original_query_sha256"] != fingerprint(original[r["_id"]]["text"]) for r in rewritten)):
        raise ValueError("Saved rewrites no longer match the complete original test queries")
    corpus_sha = file_hash(corpus_path)
    with model_loading_guard():
        lexical = BM25.build(corpus, WORKSPACE / "cache", corpus_sha, k1=1.2, b=.75)
    bm25 = {
        qid: lexical.search(row["text"], 200, excluded=positions.get(qid) if task == "rev2rev-en" else None)
        for qid, row in original.items()
    }
    selection = json.loads((RECOVERY / "final_selection.json").read_text())
    fine_model = Path(selection["selected_checkpoint"])
    base_index = WORKSPACE / "runs/official_recare/native-jina-v3-rat2rev-en-test/native_indexes/jina-v3/en"
    if ids != (base_index / "ids.txt").read_text().splitlines():
        raise ValueError("Original passage row mapping differs")
    for variant, checkpoint, index_path in (
        ("pretrained", MODEL, base_index / "embeddings.npy"),
        ("finetuned", fine_model, fine_model / "evaluation/native_index/embeddings.npy"),
    ):
        folder = ROOT / variant / "retrieval" / task
        identity = {
            "variant": variant, "task": task, "split": "test",
            "corpus_sha256": corpus_sha, "rewrites_sha256": file_hash(rewrite_path),
            "original_queries_sha256": file_hash(WORKSPACE / "data/queries" / task / "test.jsonl"),
            "embedding_weights_sha256": file_hash(checkpoint / "model.safetensors"),
            "passage_index_sha256": file_hash(index_path),
            "query_task": "retrieval.query", "passage_task": "retrieval.passage",
            "max_length": 8192, "dense_top_k": 200, "bm25_top_k": 200, "fusion_top_k": 200,
            "bm25_k1": 1.2, "bm25_b": .75, "rrf_k": 60, "weights": [1., 1.],
            "rewrite_used_for": "dense queries only",
            "reranker_queries": "complete original queries",
        }
        digest = fingerprint(identity)
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["identity_hash"] != digest:
                raise ValueError("Hybrid retrieval checkpoint belongs to different inputs/models")
            if previous.get("complete"):
                if any(file_hash(folder / name) != sha for name, sha in previous["run_sha256"].items()):
                    raise ValueError("Hybrid retrieval output checksum differs")
                print(f"REUSE_RETRIEVAL {variant}/{task}", flush=True)
                continue
        model = load_native(checkpoint)
        documents = np.load(index_path, mmap_mode="r")
        query_vectors = native_vectors(model, rewritten, folder / "query_vectors" / digest)
        dense = rank([r["_id"] for r in rewritten], query_vectors, documents, ids, task, top_k=200)
        dense_by_id = {r["qid"]: r["ranked"] for r in dense}
        lexical_rows, fused = [], []
        for qid in original:
            lexical_rows.append({"qid": qid, "ranked": [ids[i] for i in bm25[qid]]})
            order = reciprocal_rank_fusion(
                [bm25[qid], [positions[d] for d in dense_by_id[qid]]], [1., 1.], 60, 200)
            fused.append({"qid": qid, "ranked": [ids[i] for i in order]})
        runs = {"dense.jsonl": dense, "bm25.jsonl": lexical_rows, "rrf.jsonl": fused}
        hashes, metrics = {}, {}
        for name, rows in runs.items():
            path = folder / name
            write_jsonl(path, rows)
            hashes[name] = file_hash(path)
            metrics[name] = evaluate_stage(path, task)
        write_json(manifest_path, {
            "complete": True, "identity": identity, "identity_hash": digest,
            "queries": len(original), "run_sha256": hashes, "metrics": metrics,
            "qrels_used_for_inference": False, "completed_at": timestamp(),
        })
        print("HYBRID_RETRIEVAL " + json.dumps({"variant": variant, "task": task, "metrics": metrics}), flush=True)
        del model, documents
        gc.collect()
        torch.cuda.empty_cache()

