"""Freeze corrected data for recovery experiments without modifying the failed run."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from recare_retrieval.common import TASKS, file_hash, read_jsonl, write_json, write_jsonl
from recare_retrieval.jina_recovery import connected_holdout, text_key

ROOT = WORKSPACE / "runs/finetune/jina_v3_recovery_v2"
SOURCE = WORKSPACE / "runs/finetune/jina_v3_joint_v1"


def main():
    corpus = list(read_jsonl(WORKSPACE / "ReCaRe/corpus-en/corpus.jsonl"))
    docs = {r["_id"]: r["text"] for r in corpus}
    groups = list(read_jsonl(SOURCE / "dataset/groups.jsonl"))
    groups, split = connected_holdout(groups)
    rankings = {task: {r["qid"]: r["ranked"] for r in read_jsonl(
        SOURCE / "mining" / task / "best.jsonl")} for task in TASKS}
    repaired = []
    for row in groups:
        positive_keys = {text_key(docs[d]) for d in row["positive_ids"]}
        selected, negative_keys = [], set()
        for doc in rankings[row["task"]][row["qid"]]:
            key = text_key(docs[doc])
            if (doc == row["qid"] or doc in row["positive_ids"]
                    or key in positive_keys or key in negative_keys):
                continue
            selected.append(doc)
            negative_keys.add(key)
            if len(selected) == 10:
                break
        if len(selected) != 10:
            raise ValueError(f"Insufficient clean teacher negatives: {row['group_id']}")
        if selected != row["negative_ids"]:
            repaired.append(row["group_id"])
        row["negative_ids"] = selected
    partitions = defaultdict(set)
    for row in groups:
        for doc in set(row["positive_ids"]) | ({row["qid"]} if row["task"] == "rev2rev-en" else set()):
            partitions[doc].add(row["training_split"])
    if any(len(value) != 1 for value in partitions.values()):
        raise ValueError("Target-article leakage across the new split")
    article_events = defaultdict(set)
    rat_ids = {r["qid"] for r in groups if r["task"] == "rat2rev-en"}
    for row in read_jsonl(WORKSPACE / "ReCaRe/metadata-en/dataset.jsonl"):
        event, article = row.get("amendment_law_id"), row.get("article_id_before")
        if isinstance(event, str) and event in rat_ids and isinstance(article, str):
            article_events[article].add(event)
    metadata_partitions = defaultdict(set)
    metadata_groups = 0
    for row in groups:
        events = {row["qid"]} if row["task"] == "rat2rev-en" else article_events[row["qid"]]
        metadata_groups += bool(events)
        for event in events:
            metadata_partitions[event].add(row["training_split"])
    conflicts = [event for event, parts in metadata_partitions.items() if len(parts) > 1]
    if conflicts:
        raise ValueError(f"Metadata event leakage remains: {conflicts}")
    counts = Counter((r["task"], r["training_split"]) for r in groups)
    path = ROOT / "dataset/groups.jsonl"
    if path.exists():
        import json
        previous = list(read_jsonl(path))
        if previous != groups:
            raise ValueError("Frozen recovery dataset differs; use a new experiment root")
    else:
        write_jsonl(path, groups)
    manifest = {
        "complete": True, "source_groups_sha256": file_hash(SOURCE / "dataset/groups.jsonl"),
        "groups_sha256": file_hash(path), "corpus_sha256": file_hash(WORKSPACE / "ReCaRe/corpus-en/corpus.jsonl"),
        "split_policy": "Connected target-article sets plus identical original query texts; seed 20261027; 15% holdout",
        "split": split, "counts": {f"{task}/{part}": count for (task, part), count in counts.items()},
        "repaired_negative_groups": repaired, "test_labels_used": False,
        "target_id_overlap_between_train_holdout": 0,
        "metadata_event_overlap_between_train_holdout": 0,
        "metadata_event_resolved_groups": metadata_groups,
        "source_splits": ["train", "validation"],
    }
    write_json(ROOT / "dataset/manifest.json", manifest)
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
