import importlib


class ContextOverflow(ValueError):
    pass


def jina_load(path):
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", local_files_only=True).eval().cuda()
    model._ensure_tokenizer()
    original = model._compute_single_batch
    model.paper_batches = []

    def compute(query, docs, instruction=None):
        module = importlib.import_module(model.__class__.__module__)
        prompt = module.format_docs_prompts_func(
            query, docs, instruction=instruction, special_tokens=model.special_tokens, no_thinking=True)
        length = len(model._tokenizer(prompt)["input_ids"])
        if length > model.config.max_position_embeddings:
            raise ContextOverflow(f"Native Jina batch context {length} exceeds limit")
        model.paper_batches.append({"documents": len(docs), "tokens": length})
        return original(query, docs, instruction=instruction)

    model._compute_single_batch = compute
    return model


def jina_rank(model, query, documents):
    original = model._tokenizer.model_max_length
    try:
        for reserve in (0, 4096, 8192, 16384):
            model.paper_batches = []
            model._tokenizer.model_max_length = original-reserve
            try:
                result = model.rerank(query, documents, top_n=None)
                indexes = [int(r["index"]) for r in result]
                if sorted(indexes) != list(range(len(documents))):
                    raise ValueError("Jina did not return each document exactly once")
                return indexes, {"context_reserve": reserve, "batches": model.paper_batches,
                                 "raw_results": [
                                     {"index": int(r["index"]), "relevance_score": float(r["relevance_score"])}
                                     for r in result]}
            except ContextOverflow:
                if reserve == 16384:
                    raise
    finally:
        model._tokenizer.model_max_length = original

