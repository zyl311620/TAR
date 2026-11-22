"""Joint native-adapter fine-tuning with all qrels positives and ten hard negatives."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE/"runs/finetune/jina_v3_joint_v1"
BASE = Path("/root/user/cs_tcci_zhouyulin/LLM_list/jina-embeddings-v3")
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT/"hf_modules"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from recare_retrieval.common import (
    TASKS, file_hash, fingerprint, read_jsonl, timestamp, unique_ids, write_json,
)
from recare_retrieval.jina_training import (
    enable_layer_checkpointing, gradient_cached_group, multi_positive_loss, retrieval_forward,
)
from scripts.reproduce_official_recare import stage_local_model_code


def load_trainable():
    import torch
    from transformers import AutoModel, AutoTokenizer
    stage_local_model_code(BASE)
    model = AutoModel.from_pretrained(
        BASE, trust_remote_code=True, torch_dtype=torch.bfloat16,
        use_flash_attn=True, lora_main_params_trainable=False,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    trainable = {}
    for name, parameter in torch.nn.Module.named_parameters(model):
        enabled = "lora_" in name and "embeddings" not in name
        parameter.requires_grad_(enabled)
        if enabled:
            parameter.data = parameter.data.float()
            trainable[name] = parameter
    if not trainable or model._adaptation_map["retrieval.query"] != 0:
        raise ValueError("Native retrieval adapters are unavailable")
    enable_layer_checkpointing(model)
    return model, tokenizer, trainable


def save_model(model, tokenizer, path, metadata):
    path.mkdir(parents=True, exist_ok=True)
    model.config.lora_main_params_trainable = False
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    # Export self-contained custom code, removing absolute server-only auto_map origins.
    config = json.loads((path/"config.json").read_text(encoding="utf-8"))
    source = Path(json.loads((BASE/"config.json").read_text(encoding="utf-8"))[
        "auto_map"]["AutoModel"].split("--", 1)[0])
    for file in source.glob("*.py"):
        shutil.copyfile(file, path/file.name)
    config["auto_map"] = {key: value.split("--")[-1]
                          for key, value in config["auto_map"].items()}
    write_json(path/"config.json", config)
    write_json(path/"checkpoint_meta.json", {
        **metadata, "model": "jina-v3", "task": "joint_english", "lang": "en",
        "tuning_method": "lora", "base_model": str(BASE),
        "inference": "native model.encode; retrieval.query/retrieval.passage; 8192 tokens",
    })


def validate(model, tokenizer, groups, corpus):
    import torch
    model.eval()
    losses = {task: [] for task in TASKS}
    document_cache = {}
    with torch.no_grad():
        for row in groups:
            query = retrieval_forward(
                model, tokenizer, [row["text"], row["rewrite"]], is_query=True,
            )
            doc_ids = row["positive_ids"] + row["negative_ids"]
            missing = [doc for doc in doc_ids if doc not in document_cache]
            for start in range(0, len(missing), 4):
                batch = missing[start:start+4]
                encoded = retrieval_forward(
                    model, tokenizer, [corpus[doc]["text"] for doc in batch], is_query=False,
                )
                for index, doc in enumerate(batch):
                    document_cache[doc] = encoded[index:index+1]
            docs = torch.cat([document_cache[doc] for doc in doc_ids])
            n_pos = len(row["positive_ids"])
            loss = multi_positive_loss(query, docs[:n_pos], docs[n_pos:])
            losses[row["task"]].append(float(loss))
    per_task = {task: sum(values)/len(values) for task, values in losses.items()}
    return {"macro_loss": sum(per_task.values())/len(per_task), "by_task": per_task}


def gradient_check(model, tokenizer, trainable):
    """Check cached gradients against a single ordinary differentiable graph."""
    import torch
    texts = ["financial reporting obligations", "company reporting requirements",
             "annual financial statements must be filed",
             "quarterly disclosure of financial information",
             "fisheries quotas and environmental inspections"]
    model.eval()
    native_differences = {}
    for is_query, task in ((True, "retrieval.query"), (False, "retrieval.passage")):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            native = model.encode(
                [texts[0]], task=task, max_length=8192, truncate_dim=None,
                show_progress_bar=False, convert_to_tensor=True,
            ).float()
            forward = retrieval_forward(model, tokenizer, texts[0], is_query=is_query)
        if not torch.allclose(native, forward, atol=1e-5, rtol=1e-4):
            raise ValueError("Differentiable pooling/task path differs from native Jina encode")
        native_differences[task] = float((native-forward).abs().max())
    model.train()
    model.zero_grad(set_to_none=True)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    query = retrieval_forward(model, tokenizer, texts[:2], is_query=True)
    docs = retrieval_forward(model, tokenizer, texts[2:], is_query=False)
    loss = multi_positive_loss(query, docs[:2], docs[2:])
    loss.backward()
    reference = {name: p.grad.detach().cpu().clone()
                 for name, p in trainable.items() if p.grad is not None}
    del query, docs, loss
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    gradient_cached_group(model, tokenizer, texts[:2], texts[2:], 2, microbatch_size=4)
    largest = 0.
    for name, parameter in trainable.items():
        if name not in reference:
            continue
        actual, expected = parameter.grad.detach().cpu(), reference[name]
        difference = float((actual-expected).abs().max())
        largest = max(largest, difference)
        if not torch.allclose(actual, expected, atol=2e-5, rtol=2e-3):
            raise ValueError(f"Gradient-cache parity check failed for {name}: {difference}")
        if not torch.isfinite(actual).all():
            raise ValueError("Nonfinite adapter gradient")
        if torch.count_nonzero(actual[2:]):
            raise ValueError("A non-retrieval adapter received a gradient")
    model.zero_grad(set_to_none=True)
    return {"passed": True, "checked_parameter_tensors": len(reference),
            "maximum_absolute_gradient_difference": largest,
            "dropout_active": True, "non_retrieval_gradients_zero": True,
            "encoder_microbatch_size": 4,
            "native_encode_maximum_differences": native_differences}


def train(args):
    import numpy as np
    import torch
    seed = 20261025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = args.root
    model, tokenizer, trainable = load_trainable()
    check = gradient_check(model, tokenizer, trainable)
    write_json(root/"gradient_replay_check.json", check)
    if args.check_only:
        print(json.dumps(check), flush=True)
        return
    manifest = json.loads((root/"dataset/manifest.json").read_text(encoding="utf-8"))
    if (not manifest["complete"] or manifest["test_labels_used"]
            or manifest["groups_sha256"] != file_hash(root/"dataset/groups.jsonl")):
        raise ValueError("Dataset is incomplete or changed")
    groups = list(read_jsonl(root/"dataset/groups.jsonl"))
    training = [r for r in groups if r["training_split"] == "train"]
    holdout = [r for r in groups if r["training_split"] == "holdout"]
    if not training or not holdout:
        raise ValueError("Empty training or internal holdout split")
    train_ids = {r["group_id"] for r in training}
    holdout_ids = {r["group_id"] for r in holdout}
    train_text = {" ".join(r["text"].lower().split()) for r in training}
    holdout_text = {" ".join(r["text"].lower().split()) for r in holdout}
    if train_ids & holdout_ids or train_text & holdout_text:
        raise ValueError("Query group leakage into validation")
    corpus = unique_ids(WORKSPACE/"ReCaRe/corpus-en/corpus.jsonl")
    for row in groups:
        if len(row["negative_ids"]) != 10 or set(row["negative_ids"]) & set(row["positive_ids"]):
            raise ValueError("Invalid hard negative labels")
    max_epochs = 5 if len(training)*2 < 5000 else 3
    accumulation = 16
    steps_per_epoch = math.ceil(len(training)/accumulation)
    total_steps = steps_per_epoch*max_epochs
    warmup_steps = max(1, round(total_steps*0.1))
    config = {
        "dataset_sha256": manifest["groups_sha256"], "seed": seed, "max_epochs": max_epochs,
        "early_stopping_patience": 2, "checkpoint_metric": "internal holdout macro contrastive loss",
        "learning_rate": 1e-4, "weight_decay": 0., "temperature": 0.05,
        "max_length": 8192, "gradient_accumulation_groups": accumulation,
        "query_views_per_group": 2, "hard_negatives_per_group": 10,
        "all_qrels_positives_used_each_epoch": True,
        "tuning": "native transformer retrieval LoRA; frozen backbone and embedding adapters",
        "precision": "BF16 frozen base, FP32 adapter parameters, BF16 autocast",
        "gradient_checkpoint": "bound layer reentrant replay",
        "gradient_cache": True, "task_weighting": "equal expected loss per task",
        "encoder_microbatch_size": 4,
        "validation_document_cache": "within one fixed-weight validation pass only",
        "train_groups": len(training), "holdout_groups": len(holdout),
        "test_labels_used": False, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    identity = fingerprint(config)
    config_path = root/"training/config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Training config changed; use a fresh experiment root")
    write_json(config_path, {**config, "identity": identity})
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=0.)
    start_epoch, start_offset, global_step = 0, 0, 0
    best_loss, bad_epochs = float("inf"), 0
    history = []
    resume = root/"training/resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Resume checkpoint identity mismatch")
        with torch.no_grad():
            for name, value in saved["adapters"].items():
                trainable[name].copy_(value.cuda())
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.cuda()
        start_epoch, start_offset, global_step = saved["epoch"], saved["offset"], saved["global_step"]
        best_loss, bad_epochs, history = saved["best_loss"], saved["bad_epochs"], saved["history"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["gpu_rng"])
        if bad_epochs >= 2:
            start_epoch = max_epochs
    else:
        initial = validate(model, tokenizer, holdout, corpus)
        write_json(root/"training/initial_holdout.json", initial)
    task_counts = Counter(r["task"] for r in training)

    def save_resume(epoch, offset):
        saved = {
            "identity": identity, "epoch": epoch, "offset": offset, "global_step": global_step,
            "best_loss": best_loss, "bad_epochs": bad_epochs, "history": history,
            "adapters": {name: p.detach().cpu() for name, p in trainable.items()},
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(), "gpu_rng": torch.cuda.get_rng_state(),
        }
        temporary = resume.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(resume)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        order = list(range(len(training)))
        random.Random(seed+epoch).shuffle(order)
        offset = start_offset if epoch == start_epoch else 0
        losses = []
        while offset < len(order):
            optimizer.zero_grad(set_to_none=True)
            chunk = order[offset:offset+accumulation]
            started = time.monotonic()
            for index in chunk:
                row = training[index]
                task_weight = len(training)/(len(TASKS)*task_counts[row["task"]])
                docids = row["positive_ids"]+row["negative_ids"]
                value = gradient_cached_group(
                    model, tokenizer, [row["text"], row["rewrite"]],
                    [corpus[doc]["text"] for doc in docids], len(row["positive_ids"]),
                    scale=task_weight/len(chunk),
                    microbatch_size=4,
                )
                losses.append(value)
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite parameter gradient norm")
            global_step += 1
            if global_step <= warmup_steps:
                multiplier = global_step/warmup_steps
            else:
                progress = (global_step-warmup_steps)/max(1, total_steps-warmup_steps)
                multiplier = 0.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group["lr"] = 1e-4*multiplier
            optimizer.step()
            offset += len(chunk)
            status = {
                "status": "training", "epoch": epoch+1, "max_epochs": max_epochs,
                "groups_done": offset, "groups_total": len(order), "global_step": global_step,
                "mean_training_loss": sum(losses)/len(losses), "last_step_seconds": time.monotonic()-started,
                "lr": optimizer.param_groups[0]["lr"], "updated_at": timestamp(),
            }
            write_json(root/"training/status.json", status)
            print(json.dumps(status), flush=True)
            if global_step % 5 == 0:
                save_resume(epoch, offset)
        scores = validate(model, tokenizer, holdout, corpus)
        improved = scores["macro_loss"] < best_loss-1e-5
        if improved:
            best_loss = scores["macro_loss"]
        bad_epochs = 0 if improved else bad_epochs+1
        record = {"epoch": epoch+1, "global_step": global_step, **scores, "improved": improved}
        history.append(record)
        if improved:
            save_model(model, tokenizer, root/"model/best", {**record, "training_identity": identity})
        save_model(model, tokenizer, root/"model/last", {**record, "training_identity": identity})
        write_json(root/"training/history.json", history)
        save_resume(epoch+1, 0)
        print("EPOCH_RESULT " + json.dumps(record), flush=True)
        if bad_epochs >= 2:
            break
    write_json(root/"training/status.json", {
        "status": "complete", "epochs_trained": len(history), "best_holdout_loss": best_loss,
        "best_checkpoint": str(root/"model/best"), "history": history,
        "identity": identity, "completed_at": timestamp(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
