from kernel_common import *


SPECS = {
    "bge-m3": {"directory": "bge-m3", "method": "BGE-M3 dense",
               "pooling": "cls", "max_length": 8192, "normalize": True},
    "me5-base": {"directory": "multilingual-e5-base", "method": "multilingual-e5-base",
                "pooling": "mean", "max_length": 512, "normalize": True,
                "query_prefix": "query: ", "passage_prefix": "passage: "},
    "jina-v3": {"directory": "jina-embeddings-v3", "method": "Jina-v3 原始权重",
               "pooling": "jina_encode", "max_length": 8192, "normalize": True},
    "jina-v4": {"directory": "jina-embeddings-v4", "method": "Jina-v4 原始权重",
               "pooling": "jina_v4", "max_length": 32768, "normalize": True},
    "qwen3-embedding-8b": {
        "directory": "Qwen3-Embedding-8B", "method": "Qwen3-Embedding-8B",
        "pooling": "last_token", "max_length": 32768, "normalize": True,
        "dtype": "bfloat16", "attention": "flash_attention_2",
        "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    },
    "jina-v5-small": {
        "directory": "jina-embeddings-v5-text-small", "method": "Jina-v5-text-small",
        "pooling": "jina_v5", "max_length": 32768, "normalize": True,
    },
}


def model_identity(path: Path) -> dict:
    manifest_path = ROOT / f"models/{path.name}.sha256.json"
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and p.suffix in (".json", ".py", ".safetensors", ".bin", ".model"))
    signature = {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
    if manifest_path.exists():
        previous = load(manifest_path)
        if previous["file_stats"] == signature:
            return previous
    hashes = {}
    for p in files:
        hashes[str(p.relative_to(path))] = digest(p)
        print("MODEL_HASH " + str(p.relative_to(path)), flush=True)
    result = {"path": str(path), "file_stats": signature, "files": hashes,
              "sha256": identity(hashes)}
    save(manifest_path, result)
    return result


class Encoder:
    def __init__(self, spec: dict, path: Path):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.spec, self.torch = spec, torch
        trust = spec["pooling"] in ("jina_encode", "jina_v4", "jina_v5")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust,
                                                      local_files_only=True)
        if trust:
            import shutil
            from transformers.dynamic_module_utils import (
                HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME, _sanitize_module_name,
            )
            cache = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / _sanitize_module_name(path.name)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "__init__.py").touch(exist_ok=True)
            for source in path.glob("*.py"):
                shutil.copy2(source, cache / source.name)
        use_bf16 = spec["pooling"] in ("jina_v4", "jina_v5") or spec.get("dtype") == "bfloat16"
        kwargs = dict(trust_remote_code=trust, local_files_only=True,
                      torch_dtype=torch.bfloat16 if use_bf16 else torch.float16)
        if spec.get("attention"):
            kwargs["attn_implementation"] = spec["attention"]
        if spec["pooling"] in ("jina_v4", "jina_v5"):
            kwargs.update(torch_dtype=torch.bfloat16, attn_implementation=(
                "flash_attention_2" if spec["pooling"] == "jina_v4" else "sdpa"))
        self.model = AutoModel.from_pretrained(path, **kwargs).eval().cuda()
        if not use_bf16:
            self.model = self.model.half()
        elif spec["pooling"] == "jina_v4":
            base = self.model.get_base_model()
            if not isinstance(base.lm_head, (torch.nn.Linear, torch.nn.Identity)):
                raise ValueError("Unexpected Jina-v4 output projection")
            base.lm_head = torch.nn.Identity()
            base.verbosity = 0

    def encode(self, texts, query=False):
        import numpy as np
        torch, spec = self.torch, self.spec
        with torch.no_grad():
            if spec["pooling"] == "jina_v4":
                result = self.model.encode_text(
                    texts=list(texts), task="retrieval",
                    prompt_name="query" if query else "passage",
                    max_length=spec["max_length"], batch_size=len(texts),
                    return_multivector=False, return_numpy=True, truncate_dim=None,
                )
                return np.asarray(result, dtype=np.float32)
            if spec["pooling"] == "jina_v5":
                parameters = {
                    "task": "retrieval", "max_length": spec["max_length"],
                    "prompt_name": "query" if query else "document",
                }
                result = self.model.encode(
                    texts=list(texts), **parameters,
                )
                if hasattr(result, "cpu"):
                    result = result.float().cpu().numpy()
                return np.asarray(result, dtype=np.float32)
            if spec["pooling"] == "jina_encode":
                result = self.model.encode(
                    texts, task="retrieval.query" if query else "retrieval.passage",
                    max_length=spec["max_length"], truncate_dim=None,
                )
                if hasattr(result, "cpu"):
                    result = result.cpu().numpy()
                return np.asarray(result, dtype=np.float32)
            prefix = spec.get("query_prefix" if query else "passage_prefix", "")
            batch = self.tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                                   max_length=spec["max_length"], return_tensors="pt")
            batch = {k: v.cuda() for k, v in batch.items()}
            hidden = self.model(**batch).last_hidden_state
            if spec["pooling"] == "last_token":
                if batch["attention_mask"][:, -1].sum() == len(texts):
                    result = hidden[:, -1]
                else:
                    last = batch["attention_mask"].sum(dim=1) - 1
                    result = hidden[torch.arange(len(texts), device=hidden.device), last]
            elif spec["pooling"] == "cls":
                result = hidden[:, 0]
            else:
                mask = batch["attention_mask"].unsqueeze(-1).float()
                result = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if spec["normalize"]:
                result = torch.nn.functional.normalize(result, p=2, dim=1)
            return result.float().cpu().numpy()

