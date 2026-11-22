from pathlib import Path
import json,shutil,logging
logger=logging.getLogger(__name__)


def _prime_local_remote_code_cache(checkpoint_dir: Path) -> None:
    """Copy all local remote-code modules into Transformers' dynamic cache.

    For local checkpoints, Transformers keys the dynamic module cache by the
    checkpoint directory basename (``best`` / ``last`` here) and may copy only
    direct imports. Jina's saved modeling code has transitive relative imports,
    so pre-populate the cache with every saved Python module before loading.
    """
    if not checkpoint_dir.is_dir():
        return
    py_files = list(checkpoint_dir.glob("*.py"))
    if not py_files:
        return

    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE,
        TRANSFORMERS_DYNAMIC_MODULE_NAME,
        _sanitize_module_name,
    )

    submodule = _sanitize_module_name(checkpoint_dir.name)
    cache_dir = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / submodule
    cache_dir.mkdir(parents=True, exist_ok=True)
    init_path = cache_dir / "__init__.py"
    if not init_path.exists():
        init_path.touch()
    for source in py_files:
        shutil.copy2(source, cache_dir / source.name)

