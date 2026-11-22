from pathlib import Path
import hashlib,json,shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_local_model_code(model_path: Path) -> dict:
    """Populate transitive local model modules without changing their source.

    Transformers' dynamic loader can omit transitive imports when auto_map
    points at another local directory. This only copies the original .py files
    into this process's dedicated HF module cache.
    """
    from transformers.dynamic_module_utils import HF_MODULES_CACHE, get_cached_module_file

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    reference = config.get("auto_map", {}).get("AutoConfig", "")
    if "--" not in reference:
        return {}
    origin, class_reference = reference.split("--", 1)
    source = Path(origin)
    if not source.is_absolute() or not source.is_dir():
        return {}
    module_name = class_reference.rsplit(".", 1)[0]
    relative = get_cached_module_file(str(source), module_name + ".py", local_files_only=True)
    target = (Path(HF_MODULES_CACHE) / relative).parent
    hashes = {}
    for path in sorted(source.glob("*.py")):
        destination = target / path.name
        digest = sha256(path)
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError(f"Dynamic model cache differs from original source: {path.name}")
        else:
            shutil.copyfile(path, destination)
        hashes[path.name] = digest
    return {"source": str(source), "cache": str(target), "unmodified_source_sha256": hashes}

