from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

TASKS = ("rat2rev-en", "rev2rev-en")
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield row


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Team/run names must contain only ASCII letters, digits, _ or -.")
    return value


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"run(?:[1-9]|10)", value):
        raise ValueError("Official run IDs must be run1 through run10 (run1 is the primary run).")
    return value


def unique_ids(path: Path, key: str = "_id") -> dict[str, dict]:
    result = {}
    for row in read_jsonl(path):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path}: missing/non-string {key}")
        if value in result:
            raise ValueError(f"{path}: duplicate {key}={value}")
        result[value] = row
    return result

