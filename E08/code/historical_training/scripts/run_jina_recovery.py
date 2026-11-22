"""Train and validate corpus-aware LoRA recovery variants from original Jina-v3."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "runs/finetune/jina_v3_recovery_v2"
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / "hf_modules"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "official_baselines/ReCaRe/src"))

from recare_retrieval.common import TASKS, file_hash, fingerprint, read_jsonl, timestamp, write_json, write_jsonl
from recare_retrieval.jina_recovery import (
    corpus_contrastive_loss, encoding_chunks, exclusions, masked_batch_loss, positive_pool_loss, text_key,
)
from recare_retrieval.jina_training import retrieval_forward, gradient_cached_group
from scripts.train_joint_jina import load_trainable, save_model

BASE_INDEX = WORKSPACE / "runs/official_recare/native-jina-v3-rat2rev-en-test/native_indexes/jina-v3/en"
BASE_MODEL = Path("/root/user/cs_tcci_zhouyulin/LLM_list/jina-embeddings-v3")


def evaluate(model, tokenizer, validation, documents, ids, folder, step,
             split_name="validation", query_field="text"):
    """Always evaluate current query/passage weights against the complete corpus."""
    import numpy as np
    import torch
    from recare_baselines.ranked_eval import evaluate_ranked_files
    model.eval()
    results, query_geometry = {}, {}
    with torch.no_grad():
        for task in TASKS:
            rows = [r for r in validation if r["task"] == task]
            predictions, queries = [], []
            for row in rows:
                query = retrieval_forward(model, tokenizer, row[query_field], is_query=True)
                queries.append(query[0].cpu().numpy())
                scores = (query @ documents.T)[0]
                if task == "rev2rev-en":
                    scores[ids.index(row["qid"])] = -torch.inf
                top = torch.topk(scores, 101).indices.cpu().tolist()
                ranked = [ids[i] for i in top if task != "rev2rev-en" or ids[i] != row["qid"]][:100]
                predictions.append({"qid": row["qid"], "ranked": ranked})
            destination = folder / f"{split_name}/step_{step:05d}/{task}.jsonl"
            write_jsonl(destination, predictions)
            labels = folder / f"{split_name}/{task}.qrels.jsonl"
            if not labels.exists():
                write_jsonl(labels, [
                    {"query-id": r["qid"], "corpus-id": doc, "score": 1}
                    for r in rows for doc in r["positive_ids"]
                ])
            metrics = evaluate_ranked_files(destination, labels, filter_self=task == "rev2rev-en")
            if metrics.missing_qids or metrics.extra_qids or metrics.n_queries != len(rows):
                raise ValueError("Validation query coverage differs")
            write_json(destination.with_suffix(".metrics.json"), asdict(metrics))
            results[task] = {key: metrics.metrics[key] for key in ("ndcg_10", "recall_100")}
            values = np.stack(queries)
            n = len(values)
            query_geometry[task] = {
                "mean_pairwise_cosine": float(((values @ values.T).sum() - n) / (n * (n - 1))),
                "dimension_variance": float(values.var(axis=0).sum()),
            }
    result = {
        "step": step, "split": split_name, "query_field": query_field,
        "tasks": results, "query_geometry": query_geometry,
        "macro_ndcg_10": sum(r["ndcg_10"] for r in results.values()) / len(results),
        "macro_recall_100": sum(r["recall_100"] for r in results.values()) / len(results),
    }
    sample = random.Random(20261027).sample(range(len(ids)), 500)
    values = documents[sample].cpu().numpy()
    similarities = values @ values.T
    result["document_geometry"] = {
        "sample_documents": 500,
        "mean_pairwise_cosine": float((similarities.sum() - np.trace(similarities)) / (500 * 499)),
        "dimension_variance": float(values.var(axis=0).sum()),
    }
    model.train()
    print("EVALUATION " + json.dumps(result), flush=True)
    return result


def refresh_documents(model, tokenizer, corpus, folder, step, batch_size=4):
    """Re-encode every document after updating the passage adapter."""
    import numpy as np
    import torch
    path = folder / f"indexes/step_{step:05d}/embeddings.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Intermediate full indexes are replaced only inside this variant's folder.
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(corpus), 1024))
    order = sorted(range(len(corpus)), key=lambda i: len(corpus[i]["text"]))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            positions = order[start:start+batch_size]
            encoded = retrieval_forward(
                model, tokenizer, [corpus[i]["text"] for i in positions], is_query=False,
            ).cpu().numpy()
            values[positions] = encoded
            if start % 4096 == 0 or start + batch_size >= len(order):
                write_json(folder / "status.json", {
                    "status": "encoding_current_passages", "step": step,
                    "documents_done": min(start+batch_size, len(order)), "documents_total": len(order),
                    "updated_at": timestamp(),
                })
                print(f"INDEX step={step} {min(start+batch_size,len(order))}/{len(order)}", flush=True)
    values.flush()
    if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-3):
        raise ValueError("Invalid refreshed corpus")
    write_json(path.with_suffix(".manifest.json"), {
        "step": step, "documents": len(corpus), "max_length": 8192,
        "task": "retrieval.passage", "index_sha256": file_hash(path),
        "corpus_sha256": file_hash(WORKSPACE / "ReCaRe/corpus-en/corpus.jsonl"),
        "complete": True,
    })
    model.train()
    return torch.from_numpy(np.asarray(values).copy()).cuda()


def restore_document_bank(model, tokenizer, corpus, folder, step, history, batch_size):
    """Reuse a completed, validated same-step bank after checking fresh encodings."""
    import numpy as np
    import torch
    path = folder / f"indexes/step_{step:05d}/embeddings.npy"
    manifest_path = path.with_suffix(".manifest.json")
    if history[-1]["step"] == step and path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["complete"] and manifest["step"] == step and manifest["index_sha256"] == file_hash(path):
            values = np.load(path, mmap_mode="r")
            checks = []
            model.eval()
            with torch.no_grad():
                for index in random.Random(20261027).sample(range(len(corpus)), 16):
                    fresh = retrieval_forward(
                        model, tokenizer, corpus[index]["text"], is_query=False)[0].cpu().numpy()
                    checks.append(float(fresh @ values[index]))
            model.train()
            write_json(path.parent / "resume_consistency.json", {
                "step": step, "sampled_documents": 16, "minimum_cosine": min(checks),
                "verified_index_sha256": manifest["index_sha256"],
                "same_step_already_validated": True,
            })
            if min(checks) > .9999:
                print(f"REUSE_INDEX step={step} minimum_fresh_cosine={min(checks)}", flush=True)
                return torch.from_numpy(np.asarray(values).copy()).cuda()
    return refresh_documents(model, tokenizer, corpus, folder, step, batch_size)


def cached_mixed_batch(model, tokenizer, rows, corpus, documents, positions, canonical, weights, seed,
                       microbatch_size=2, reference_queries=None, reference_documents=None,
                       anchor_weight=0.):
    """Replay one genuinely cross-query batch, not independent accumulated losses."""
    import torch
    rng = random.Random(seed)
    query_texts, doc_indices, doc_set = [], [], set()
    per_row = []
    for row in rows:
        excluded = exclusions(row, positions, canonical)
        chosen = [positions[d] for d in row["positive_ids"] + row["negative_ids"]]
        model.eval()
        with torch.no_grad():
            q = retrieval_forward(model, tokenizer, row["text"], is_query=True)
            scores = (q @ documents.T)[0]
            scores[list(excluded)] = -torch.inf
            dynamic = torch.topk(scores, 256).indices.tolist()
        model.train()
        added = 0
        for doc in dynamic:
            if doc not in excluded and doc not in chosen:
                chosen.append(doc)
                added += 1
                if added == 8:
                    break
        random_count = 0
        while random_count < 8:
            doc = rng.randrange(len(corpus))
            if doc not in excluded and doc not in chosen:
                chosen.append(doc)
                random_count += 1
        for doc in chosen:
            if doc not in doc_set:
                doc_indices.append(doc)
                doc_set.add(doc)
        offset = len(query_texts)
        query_texts.extend([row["text"], row["rewrite"]])
        per_row.append((row, offset, excluded))
    local = {doc: i for i, doc in enumerate(doc_indices)}
    specs = [{
        "query_start": offset, "query_stop": offset + 2,
        "positives": [local[positions[d]] for d in row["positive_ids"]],
        "excluded": [local[d] for d in excluded if d in local],
        "weight": weights[row["task"]],
    } for row, offset, excluded in per_row]
    entries = [(text, True) for text in query_texts] + [
        (corpus[i]["text"], False) for i in doc_indices]
    chunks = encoding_chunks(entries, len(query_texts), microbatch_size)
    cached, states = [], []
    for indices in chunks:
        states.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
        with torch.no_grad():
            cached.append(retrieval_forward(
                model, tokenizer, [entries[i][0] for i in indices],
                is_query=entries[indices[0]][1]))
    order = [i for indices in chunks for i in indices]
    restore_order = sorted(range(len(order)), key=lambda i: order[i])
    leaves = torch.cat(cached)[restore_order].detach().requires_grad_(True)
    loss = masked_batch_loss(leaves[:len(query_texts)], leaves[len(query_texts):], specs)
    if anchor_weight:
        reference_q = torch.cat([reference_queries[row["group_id"]] for row in rows]).cuda()
        query_anchor = (1 - (leaves[:len(query_texts)] * reference_q).sum(dim=1)).mean()
        passage_anchor = (1 - (
            leaves[len(query_texts):] * reference_documents[doc_indices]
        ).sum(dim=1)).mean()
        loss = loss + anchor_weight * .5 * (query_anchor + passage_anchor)
    loss.backward()
    gradients = leaves.grad.detach()
    for i, (indices, (cpu_rng, cuda_rng)) in enumerate(zip(chunks, states)):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            encoded = retrieval_forward(
                model, tokenizer, [entries[j][0] for j in indices],
                is_query=entries[indices[0]][1])
            if not torch.allclose(encoded.detach(), cached[i], atol=1e-5, rtol=1e-4):
                raise ValueError("Mixed-batch gradient replay mismatch")
            (encoded * gradients[indices]).sum().backward()
    return float(loss.detach())


def main():
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(
        "query_full", "query_full_anchor", "query_full_all_lora", "query_full_pool", "query_phase2_pool", "query_phase2_uniform",
        "dual_fixed", "dual_mixed", "dual_mixed_anchor"), required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="Pilot cap; can later resume with cap 0")
    parser.add_argument("--eval-every", type=int, default=24)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--anchor-weight", type=float, default=1.)
    parser.add_argument("--run-name", help="Separate output directory for a new planned experiment")
    parser.add_argument("--evaluate-train", action="store_true",
                        help="Full-corpus training evaluation at baseline and every complete epoch")
    parser.add_argument("--initial-checkpoint", type=Path,
                        default=ROOT / "exports/dual_mixed_anchor-step00024")
    parser.add_argument("--initial-index", type=Path,
                        default=ROOT / "dual_mixed_anchor/indexes/step_00024/embeddings.npy")
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed = 20261027
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    run_name = args.run_name or args.variant
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("Run name must be one directory name")
    folder = ROOT / run_name
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "dataset/manifest.json").read_text(encoding="utf-8"))
    if (not manifest["complete"]
            or manifest["groups_sha256"] != file_hash(ROOT / "dataset/groups.jsonl")
            or manifest["corpus_sha256"] != file_hash(WORKSPACE / "ReCaRe/corpus-en/corpus.jsonl")):
        raise ValueError("Recovery data changed")
    corpus = list(read_jsonl(WORKSPACE / "ReCaRe/corpus-en/corpus.jsonl"))
    ids = [r["_id"] for r in corpus]
    if ids != (BASE_INDEX / "ids.txt").read_text(encoding="utf-8").splitlines():
        raise ValueError("Original corpus index order differs")
    positions = {doc: i for i, doc in enumerate(ids)}
    text_positions = defaultdict(list)
    for i, row in enumerate(corpus):
        text_positions[text_key(row["text"])].append(i)
    canonical = {row["_id"]: text_positions[text_key(row["text"])] for row in corpus}
    groups = list(read_jsonl(ROOT / "dataset/groups.jsonl"))
    training = [r for r in groups if r["training_split"] == "train"]
    validation = [r for r in groups if r["training_split"] == "holdout"]
    counts = Counter(r["task"] for r in training)
    task_weights = {task: len(training) / (len(TASKS) * counts[task]) for task in TASKS}
    query_only = args.variant.startswith("query_")
    phase2 = args.variant in {"query_phase2_pool", "query_phase2_uniform"}
    anchor_weight = args.anchor_weight if args.variant in {
        "query_full_anchor", "dual_mixed_anchor", "query_phase2_pool", "query_phase2_uniform"} else 0.
    config = {
        "variant": args.variant, "learning_rate": 1e-5, "temperature": .05,
        "epochs": args.epochs, "batch_groups": args.batch_groups, "anchor_weight": anchor_weight,
        "query_views": ["original", "rewritten"], "max_length": 8192,
        "test_labels_used": False, "dataset_sha256": manifest["groups_sha256"],
        "source_model": str(BASE_MODEL), "train_groups": len(training), "holdout_groups": len(validation),
        "passage_frozen": query_only, "weight_decay": 0., "seed": seed,
        "checkpoint_metric": "event-isolated validation full-corpus macro nDCG@10",
        "warmup_steps": 8,
        "negative_policy": ("all eligible corpus documents" if query_only else (
            "clean fixed10 teacher negatives" if args.variant == "dual_fixed" else
            "fixed10 teacher + dynamic8 + random8 + masked cross-query batch")),
    }
    if args.run_name or args.evaluate_train:
        config.update(run_name=run_name, evaluate_train=args.evaluate_train,
                      scheduler="8-step warmup followed by cosine over the planned run")
    if args.variant == "query_full_all_lora":
        config["train_embedding_adapters"] = True
    if args.variant in {"query_full_pool", "query_phase2_pool"}:
        config["positive_objective"] = "0.9 pooled-positive NCE + 0.1 uniform-positive NCE"
    if args.variant == "query_phase2_uniform":
        config["positive_objective"] = "Each positive independently competes against all eligible negatives; equal positive weights"
    if phase2:
        if not args.initial_checkpoint.is_relative_to(ROOT) or not args.initial_index.is_relative_to(ROOT):
            raise ValueError("Phase-two sources must remain in the recovery workspace")
        config.update(
            initial_checkpoint=str(args.initial_checkpoint),
            initial_weights_sha256=file_hash(args.initial_checkpoint / "model.safetensors"),
            training_passage_index=str(args.initial_index),
            training_passage_index_sha256=file_hash(args.initial_index),
            selection_guard="each task ndcg >= 99% and recall >= 98% of phase-one validation",
        )
    identity = fingerprint(config)
    config_path = folder / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("Recovery config changed; use another variant directory")
    write_json(config_path, {**config, "identity": identity})
    model, tokenizer, parameters = load_trainable()
    if phase2:
        from safetensors import safe_open
        real_parameters = dict(torch.nn.Module.named_parameters(model))
        with safe_open(args.initial_checkpoint / "model.safetensors", framework="pt", device="cpu") as source:
            with torch.no_grad():
                for name, parameter in real_parameters.items():
                    if "lora_" in name:
                        value = source.get_tensor(name)
                        parameter.data = parameter.data.to(value.dtype)
                        parameter.copy_(value.cuda())
    if args.variant == "query_full_all_lora":
        for name, parameter in torch.nn.Module.named_parameters(model):
            if "lora_" in name and "embeddings" in name:
                parameter.requires_grad_(True)
                parameter.data = parameter.data.float()
                parameters[name] = parameter
    gc.collect()
    torch.cuda.empty_cache()
    free_memory = torch.cuda.mem_get_info()[0]
    encoder_microbatch = 4 if free_memory > 16 * 1024**3 else 2
    index_batch_size = 16 if free_memory > 24 * 1024**3 else 4
    write_json(folder / "runtime.json", {
        "encoder_microbatch": encoder_microbatch, "index_batch_size": index_batch_size,
        "mixed_batch_packing": "homogeneous roles; length-sorted; adaptive token budget",
        "free_memory_bytes_after_model_load": free_memory,
        "torch": torch.__version__, "training_source_sha256": file_hash(Path(__file__)),
        "gpu": torch.cuda.get_device_name(0), "created_at": timestamp(),
    })
    frozen_slices = None
    if query_only:
        frozen_slices = {name: p.detach()[1:].cpu().clone() for name, p in parameters.items()}
        for p in parameters.values():
            mask = torch.zeros_like(p)
            mask[0] = 1
            p.register_hook(lambda grad, active=mask: grad * active)
    base_documents = np.load(args.initial_index if phase2 else BASE_INDEX / "embeddings.npy", mmap_mode="r")
    if base_documents.shape != (len(corpus), 1024):
        raise ValueError("Passage bank shape differs")
    documents = torch.from_numpy(np.asarray(base_documents).copy()).cuda()
    reference_documents = documents.detach() if args.variant == "dual_mixed_anchor" else None
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-5, weight_decay=0.)
    steps_per_epoch = math.ceil(len(training) / args.batch_groups)
    total_steps = steps_per_epoch * args.epochs
    warmup = 8
    step, epoch_start, offset_start = 0, 0, 0
    history, best, baseline = [], None, None
    resume_path = folder / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume identity differs")
        with torch.no_grad():
            for name, values in saved["parameters"].items():
                parameters[name].copy_(values.cuda())
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.cuda()
        step, epoch_start, offset_start = saved["step"], saved["epoch"], saved["offset"]
        history, best, baseline = saved["history"], saved["best"], saved["baseline"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        del saved
        if not query_only:
            del documents
            torch.cuda.empty_cache()
            documents = restore_document_bank(
                model, tokenizer, corpus, folder, step, history, index_batch_size)
    else:
        if phase2:
            model.eval()
            alignment = []
            with torch.no_grad():
                for index in random.Random(seed).sample(range(len(corpus)), 16):
                    fresh = retrieval_forward(model, tokenizer, corpus[index]["text"], is_query=False)[0]
                    alignment.append(float(fresh @ documents[index]))
            if min(alignment) < .9999:
                raise ValueError("Phase-two passage model differs from its frozen bank")
            write_json(folder / "initial_passage_alignment.json", {
                "sample_documents": 16, "minimum_cosine": min(alignment),
                "initial_weights_sha256": config["initial_weights_sha256"],
                "passage_index_sha256": config["training_passage_index_sha256"],
            })
        baseline = evaluate(model, tokenizer, validation, documents, ids, folder, 0)
        best = baseline
        history = [baseline]
        save_model(model, tokenizer, folder / "model/best", {
            **baseline, "variant": args.variant, "identity": identity,
            "run_name": run_name,
            "is_pretrained_baseline": not phase2, "is_initial_checkpoint": True,
        })
    fit_history_path = folder / "fit_history.json"
    fit_history = json.loads(fit_history_path.read_text()) if fit_history_path.exists() else []

    def evaluate_fit(epoch, validation_result):
        result = evaluate(model, tokenizer, training, documents, ids, folder, step,
                          split_name="training")
        result.update(epoch=epoch, validation=validation_result,
                      full_corpus_documents=len(ids), diagnostic_only=True)
        fit_history.append(result)
        write_json(fit_history_path, fit_history)
        previous = fit_history[:-1]
        if not previous or result["macro_ndcg_10"] > max(r["macro_ndcg_10"] for r in previous):
            save_model(model, tokenizer, folder / "model/train_best", {
                **result, "variant": args.variant, "run_name": run_name,
                "identity": identity, "checkpoint_metric": "training macro nDCG@10; diagnostic only",
            })
        if epoch in {3, 5, 10, args.epochs}:
            rewritten = evaluate(
                model, tokenizer, training, documents, ids, folder, step,
                split_name="training_rewritten", query_field="rewrite")
            write_json(folder / f"training_rewritten/step_{step:05d}/summary.json", rewritten)

    if args.evaluate_train and not step and not fit_history:
        evaluate_fit(0, baseline)
    references = {}
    if anchor_weight:
        reference_path = folder / "initial_query_references.pt" if phase2 else ROOT / "base_query_references.pt"
        if reference_path.exists():
            reference_data = torch.load(reference_path, map_location="cpu", weights_only=True)
            if reference_data["dataset_sha256"] != manifest["groups_sha256"]:
                raise ValueError("Base query reference dataset differs")
            references = reference_data["vectors"]
        elif step:
            raise ValueError("Cannot create original-model references after updating weights")
        else:
            model.eval()
            with torch.no_grad():
                for i, row in enumerate(training):
                    references[row["group_id"]] = torch.cat([
                        retrieval_forward(model, tokenizer, row[field], is_query=True).cpu()
                        for field in ("text", "rewrite")])
                    if i % 100 == 0:
                        print(f"REFERENCES {i}/{len(training)}", flush=True)
            torch.save({"dataset_sha256": manifest["groups_sha256"], "vectors": references}, reference_path)
            model.train()
        # The fixed references total only about 8 MiB. Keeping them on CUDA avoids
        # a blocking CPU-to-GPU copy between each query forward and backward.
        cpu_references = references
        references = {key: value.cuda() for key, value in cpu_references.items()}
        if set(references) != {row["group_id"] for row in training}:
            raise ValueError("Query reference coverage differs from training groups")
        checked = list(references)[:16]
        if any(not torch.equal(references[key].cpu(), cpu_references[key]) for key in checked):
            raise ValueError("Moving fixed references changed their numerical values")
        write_json(folder / "reference_placement.json", {
            "device": "cuda", "groups": len(references), "unchanged_values_checked": len(checked),
            "bytes": sum(v.numel()*v.element_size() for v in references.values()),
            "dataset_sha256": manifest["groups_sha256"],
        })
        del cpu_references

    def save_resume(epoch, offset):
        temporary = resume_path.with_suffix(".tmp")
        torch.save({
            "identity": identity, "step": step, "epoch": epoch, "offset": offset,
            "parameters": {name: p.detach().cpu() for name, p in parameters.items()},
            "optimizer": optimizer.state_dict(), "history": history,
            "best": best, "baseline": baseline,
            "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        }, temporary)
        temporary.replace(resume_path)

    def validate_now(epoch, offset):
        nonlocal documents, best
        if not query_only:
            del documents
            gc.collect()
            torch.cuda.empty_cache()
            documents = refresh_documents(model, tokenizer, corpus, folder, step, index_batch_size)
        result = evaluate(model, tokenizer, validation, documents, ids, folder, step)
        result["epoch"] = epoch + 1
        history.append(result)
        adapter_path = folder / f"validation_adapters/step_{step:05d}.pt"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = adapter_path.with_suffix(".tmp")
        torch.save({"step": step, "identity": identity,
                    "parameters": {name: p.detach().cpu() for name, p in parameters.items()},
                    "validation": result}, temporary)
        temporary.replace(adapter_path)
        score = (result["macro_ndcg_10"], result["macro_recall_100"])
        eligible = not phase2 or all(
            result["tasks"][task]["ndcg_10"] >= baseline["tasks"][task]["ndcg_10"] * .99
            and result["tasks"][task]["recall_100"] >= baseline["tasks"][task]["recall_100"] * .98
            for task in TASKS)
        result["selection_eligible"] = eligible
        if eligible and score > (best["macro_ndcg_10"], best["macro_recall_100"]):
            best = result
            save_model(model, tokenizer, folder / "model/best", {
                **result, "variant": args.variant, "identity": identity,
                "run_name": run_name,
                "is_pretrained_baseline": False,
            })
        if args.evaluate_train and offset >= len(training):
            evaluate_fit(epoch + 1, result)
        write_json(folder / "history.json", history)
        save_resume(epoch, offset)

    model.train()
    for epoch in range(epoch_start, args.epochs):
        order = list(range(len(training)))
        random.Random(seed + epoch).shuffle(order)
        offset = offset_start if epoch == epoch_start else 0
        while offset < len(order):
            if args.max_steps and step >= args.max_steps:
                break
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            selected = [training[i] for i in order[offset:offset + args.batch_groups]]
            losses = []
            if args.variant in {"dual_mixed", "dual_mixed_anchor"}:
                # A physical cross-query batch; batch-groups should be 2--4 on a small GPU.
                losses.append(cached_mixed_batch(
                    model, tokenizer, selected, corpus, documents, positions, canonical,
                    task_weights, seed + step, encoder_microbatch,
                    references, reference_documents, anchor_weight))
            else:
                for row in selected:
                    weight = task_weights[row["task"]] / len(selected)
                    if query_only:
                        positive = [positions[d] for d in row["positive_ids"]]
                        excluded = exclusions(row, positions, canonical)
                        for view, field in enumerate(("text", "rewrite")):
                            q = retrieval_forward(model, tokenizer, row[field], is_query=True)
                            objective = positive_pool_loss if args.variant in {
                                "query_full_pool", "query_phase2_pool"} else corpus_contrastive_loss
                            loss = objective(q, documents, positive, excluded)
                            if anchor_weight:
                                reference = references[row["group_id"]][view:view+1].cuda()
                                loss = loss + anchor_weight * (1 - (q * reference).sum(dim=1)).mean()
                            (loss * weight / 2).backward()
                            losses.append(float(loss.detach()))
                    else:
                        docids = row["positive_ids"] + row["negative_ids"]
                        losses.append(gradient_cached_group(
                            model, tokenizer, [row["text"], row["rewrite"]],
                            [corpus[positions[d]]["text"] for d in docids], len(row["positive_ids"]),
                            scale=weight, microbatch_size=encoder_microbatch,
                        ))
            norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite recovery gradient")
            step += 1
            multiplier = step / warmup if step <= warmup else .5 * (
                1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = 1e-5 * multiplier
            optimizer.step()
            offset += len(selected)
            status = {
                "status": "training", "variant": args.variant, "step": step,
                "epoch": epoch + 1, "groups_done": offset, "groups_total": len(order),
                "loss": sum(losses) / len(losses), "lr": optimizer.param_groups[0]["lr"],
                "gradient_norm": float(norm), "step_seconds": time.monotonic() - started,
                "best": best, "updated_at": timestamp(),
            }
            write_json(folder / "status.json", status)
            print("TRAIN " + json.dumps(status), flush=True)
            at_cap = args.max_steps and step >= args.max_steps
            if step % args.eval_every == 0 or offset >= len(order) or at_cap:
                validate_now(epoch, offset)
            elif step % 4 == 0:
                save_resume(epoch, offset)
        save_resume(epoch + (offset >= len(order)), 0 if offset >= len(order) else offset)
        if args.max_steps and step >= args.max_steps:
            break
    if frozen_slices is not None:
        if any(not torch.equal(p.detach()[1:].cpu(), frozen_slices[name]) for name, p in parameters.items()):
            raise ValueError("Query-only training modified a frozen passage/task adapter")
    save_model(model, tokenizer, folder / "model/last", {
        "step": step, "variant": args.variant, "identity": identity,
        "run_name": run_name,
    })
    write_json(folder / "history.json", history)
    write_json(folder / "result.json", {
        "complete": True, "pilot": bool(args.max_steps), "step": step, "config": config,
        "epochs_completed": step // steps_per_epoch,
        "training_groups_processed": (step // steps_per_epoch) * len(training)
                                    + (step % steps_per_epoch) * args.batch_groups,
        "baseline": baseline, "best": best,
        "frozen_passage_verified": query_only,
        "test_labels_used": False, "completed_at": timestamp(),
    })
    write_json(folder / "status.json", {
        "status": "pilot_complete" if args.max_steps else "complete",
        "variant": args.variant, "step": step, "best": best, "updated_at": timestamp(),
    })


if __name__ == "__main__":
    main()
