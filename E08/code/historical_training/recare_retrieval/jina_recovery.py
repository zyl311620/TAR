"""Event-isolated data and corpus-aware objectives for the Jina recovery study."""
from __future__ import annotations

from collections import defaultdict
import random


def text_key(text):
    return " ".join(text.lower().split())


def connected_holdout(groups, fraction=0.15, seed=20261027):
    """Keep overlapping target sets and identical queries in one component.

    Adding the Rev2Rev anchor restores its event's target set. Connecting on
    individual targets is more conservative than matching whole target sets.
    """
    parent = list(range(len(groups)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    targets, texts = {}, {}
    for index, row in enumerate(groups):
        documents = set(row["positive_ids"])
        if row["task"] == "rev2rev-en":
            documents.add(row["qid"])
        for doc in documents:
            if doc in targets:
                union(index, targets[doc])
            targets[doc] = index
        key = text_key(row["text"])
        if key in texts:
            union(index, texts[key])
        texts[key] = index
    components = defaultdict(list)
    for index in range(len(groups)):
        components[find(index)].append(index)
    ordered = sorted(components.values(), key=lambda ids: min(groups[i]["group_id"] for i in ids))
    random.Random(seed).shuffle(ordered)
    wanted = round(len(groups) * fraction)
    selected, count = set(), 0
    for component in ordered:
        if count < wanted:
            selected.update(component)
            count += len(component)
    result = []
    for index, row in enumerate(groups):
        result.append({**row, "training_split": "holdout" if index in selected else "train",
                       "event_component": min(groups[i]["group_id"] for i in components[find(index)])})
    if not selected or len(selected) == len(groups):
        raise ValueError("Connected event split did not leave both partitions")
    return result, {"components": len(components),
                    "maximum_component_groups": max(map(len, components.values())),
                    "holdout_groups": count, "train_groups": len(groups) - count}


def exclusions(row, doc_positions, canonical_positions):
    """Exclude every known positive, text-equivalent positive, and anchor."""
    excluded = set()
    for doc in row["positive_ids"]:
        excluded.update(canonical_positions[doc])
    if row["task"] == "rev2rev-en":
        excluded.add(doc_positions[row["qid"]])
    return excluded


def corpus_contrastive_loss(queries, documents, positive_positions, excluded_positions,
                            temperature=0.05):
    """Each positive competes against all eligible negatives in documents.

    Other positives and text duplicates are removed from the shared negative
    mass. The original and rewritten query views have equal weight.
    """
    import torch
    scores = queries.float() @ documents.float().T / temperature
    mask = torch.zeros(documents.shape[0], dtype=torch.bool, device=scores.device)
    mask[list(excluded_positions)] = True
    negative_mass = torch.logsumexp(scores.masked_fill(mask[None], -torch.inf), dim=1, keepdim=True)
    return torch.nn.functional.softplus(
        negative_mass - scores[:, list(positive_positions)]
    ).mean()


def masked_batch_loss(query_vectors, passage_vectors, group_specs, temperature=0.05):
    """Cross-group contrastive loss with per-query qrel/text exclusions."""
    import torch
    losses = []
    for spec in group_specs:
        q = query_vectors[spec["query_start"]:spec["query_stop"]]
        losses.append(corpus_contrastive_loss(
            q, passage_vectors, spec["positives"], spec["excluded"], temperature,
        ) * spec.get("weight", 1.))
    return torch.stack(losses).mean()


def positive_pool_loss(queries, documents, positive_positions, excluded_positions,
                       temperature=0.05, uniform_weight=0.1):
    """All positives contribute to a pooled numerator plus uniform coverage.

    The pooled term rewards retrieving genuine targets without forcing equally
    close scores for every article in a broad amendment event. The uniform
    component retains training pressure on every annotated positive.
    """
    import torch
    scores = queries.float() @ documents.float().T / temperature
    positive = scores[:, list(positive_positions)]
    mask = torch.zeros(documents.shape[0], dtype=torch.bool, device=scores.device)
    mask[list(excluded_positions)] = True
    negative_mass = torch.logsumexp(scores.masked_fill(mask[None], -torch.inf), dim=1)
    pooled = torch.nn.functional.softplus(negative_mass - torch.logsumexp(positive, dim=1))
    uniform = torch.nn.functional.softplus(negative_mass[:, None] - positive).mean(dim=1)
    return ((1 - uniform_weight) * pooled + uniform_weight * uniform).mean()


def encoding_chunks(entries, n_queries, microbatch_size):
    """Pack similar lengths without mixing query and passage adapter roles."""
    chunks = []
    capacity = microbatch_size * 4
    token_budget = microbatch_size * 2048
    for begin, end in ((0, n_queries), (n_queries, len(entries))):
        ordered = sorted(range(begin, end), key=lambda i: len(entries[i][0]))
        current, maximum = [], 0
        for index in ordered:
            tokens = min(8192, max(1, (len(entries[index][0]) + 2) // 3))
            next_max = max(maximum, tokens)
            if current and (len(current) == capacity or next_max * (len(current) + 1) > token_budget):
                chunks.append(current)
                current, maximum = [], 0
            current.append(index)
            maximum = max(maximum, tokens)
        if current:
            chunks.append(current)
    return chunks


def encode_native_index(model, checkpoint, corpus, corpus_path, folder, batch_size=None):
    """Build a resumable native FP16 index, restoring original document order."""
    import gc
    import json
    import numpy as np
    import torch
    from .common import file_hash, fingerprint, write_json
    identity = {
        "weights_sha256": {p.name: file_hash(p) for p in checkpoint.glob("*.safetensors")},
        "config_sha256": file_hash(checkpoint / "config.json"),
        "corpus_sha256": file_hash(corpus_path),
        "ids_sha256": fingerprint([r["_id"] for r in corpus]),
        "task": "retrieval.passage", "max_length": 8192, "dimensions": 1024,
        "backend": "native FP16 global-length-sorted v1",
    }
    digest = fingerprint(identity)
    manifest_path = folder / "manifest.json"
    output_path = folder / "embeddings.npy"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["identity_hash"] != digest:
            raise ValueError("Native corpus cache belongs to different weights/data")
        if saved.get("complete"):
            if file_hash(output_path) != saved["index_sha256"]:
                raise ValueError("Native corpus cache checksum mismatch")
            return np.load(output_path, mmap_mode="r")
    if batch_size is None:
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = 16 if torch.cuda.mem_get_info()[0] > 16 * 1024**3 else 4
    folder.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {
        "identity": identity, "identity_hash": digest, "batch_size": batch_size, "complete": False})
    order = sorted(range(len(corpus)), key=lambda i: len(corpus[i]["text"]))
    chunks = folder / "chunks"
    chunks.mkdir(exist_ok=True)
    for start in range(0, len(order), 2048):
        path = chunks / f"{start:06d}.npy"
        positions = order[start:start+2048]
        if path.exists():
            block = np.load(path)
            if block.shape != (len(positions), 1024) or not np.isfinite(block).all():
                raise ValueError("Native corpus chunk shape/data mismatch")
        else:
            batches = []
            for offset in range(0, len(positions), batch_size):
                batch = positions[offset:offset+batch_size]
                vectors = np.asarray(model.encode(
                    [corpus[i]["text"] for i in batch],
                    task="retrieval.passage", max_length=8192, truncate_dim=None,
                    batch_size=batch_size, show_progress_bar=False,
                ), dtype=np.float32)
                batches.append(vectors)
            block = np.concatenate(batches)
            if block.shape != (len(positions), 1024) or not np.isfinite(block).all():
                raise ValueError("Invalid native passage vectors")
            if not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-3):
                raise ValueError("Native passage vectors are not normalized")
            with path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, block)
            path.with_suffix(".tmp").replace(path)
        if not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-3):
            raise ValueError("Cached native passage vectors are not normalized")
        print(f"NATIVE_INDEX {min(start+len(positions),len(order))}/{len(order)}", flush=True)
    values = np.empty((len(corpus), 1024), dtype=np.float32)
    for start in range(0, len(order), 2048):
        positions = order[start:start+2048]
        values[positions] = np.load(chunks / f"{start:06d}.npy")
    with output_path.with_suffix(".tmp").open("wb") as stream:
        np.save(stream, values)
    output_path.with_suffix(".tmp").replace(output_path)
    write_json(manifest_path, {
        "identity": identity, "identity_hash": digest, "batch_size": batch_size,
        "complete": True, "documents": len(corpus), "index_sha256": file_hash(output_path),
    })
    del values
    return np.load(output_path, mmap_mode="r")
