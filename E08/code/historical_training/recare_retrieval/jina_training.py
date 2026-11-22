"""Differentiable native Jina-v3 paths and small-memory contrastive training."""
from __future__ import annotations

from functools import partial


def enable_layer_checkpointing(model):
    """Checkpoint bound layer forwards, keeping FlashAttention kwargs in closure.

    The native encoder passes keyword arguments to reentrant checkpoint(), which
    PyTorch rejects. Wrapping layers leaves the downloaded model source intact.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    encoder = model.roberta.encoder
    encoder.gradient_checkpointing = False
    model.roberta.embeddings.register_forward_hook(
        lambda module, args, output: output.requires_grad_(True)
    )
    for layer in encoder.layers:
        original = layer.forward

        def forward(*args, _original=original, _layer=layer, **kwargs):
            if _layer.training and torch.is_grad_enabled():
                return checkpoint(partial(_original, **kwargs), *args, use_reentrant=True)
            return _original(*args, **kwargs)

        layer.forward = forward


def retrieval_forward(model, tokenizer, text, *, is_query, max_length=8192):
    import torch
    import torch.nn.functional as F
    task = "retrieval.query" if is_query else "retrieval.passage"
    texts = [text] if isinstance(text, str) else list(text)
    inputs = tokenizer(
        [model._task_instructions[task] + value for value in texts],
        truncation=True, padding=True, max_length=max_length, return_tensors="pt",
    ).to("cuda")
    task_id = model._adaptation_map[task]
    adapter_mask = torch.full((len(texts),), task_id, dtype=torch.int32, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(**inputs, adapter_mask=adapter_mask)
    hidden = result.last_hidden_state.float()
    mask = inputs.attention_mask[..., None]
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
    return F.normalize(pooled, dim=1)


def multi_positive_loss(query, positive, negative, temperature=0.05):
    """Each qrel positive competes with the ten genuine hard negatives.

    Other positives never enter the negative denominator. Query groups, not the
    number of annotated articles, determine each group's loss weight.
    """
    import torch
    positive_scores = query @ positive.T / temperature
    negative_mass = torch.logsumexp(query @ negative.T / temperature, dim=1, keepdim=True)
    return torch.nn.functional.softplus(negative_mass - positive_scores).mean()


def gradient_cached_group(model, tokenizer, queries, passages, n_positive, *, scale=1.0,
                          microbatch_size=1):
    """Cache normalized embeddings and replay homogeneous microbatches with exact RNG."""
    import torch
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")
    entries = [(text, True) for text in queries] + [(text, False) for text in passages]
    chunks = []
    for begin, end in ((0, len(queries)), (len(queries), len(entries))):
        chunks.extend((offset, min(offset+microbatch_size, end))
                      for offset in range(begin, end, microbatch_size))
    cached, rng_states = [], []
    for start, stop in chunks:
        texts = [text for text, _ in entries[start:stop]]
        is_query = entries[start][1]
        rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
        with torch.no_grad():
            cached.append(retrieval_forward(model, tokenizer, texts, is_query=is_query))
    leaves = torch.cat(cached).detach().requires_grad_(True)
    n_queries = len(queries)
    loss = multi_positive_loss(
        leaves[:n_queries], leaves[n_queries:n_queries+n_positive],
        leaves[n_queries+n_positive:],
    )
    (loss * scale).backward()
    gradients = leaves.grad.detach()
    for index, ((start, stop), (cpu_rng, gpu_rng)) in enumerate(zip(chunks, rng_states)):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(gpu_rng)
            embedding = retrieval_forward(
                model, tokenizer, [text for text, _ in entries[start:stop]],
                is_query=entries[start][1],
            )
            if not torch.allclose(embedding.detach(), cached[index], atol=1e-5, rtol=1e-4):
                raise RuntimeError("Gradient-cache replay did not reproduce its cached embedding")
            (embedding * gradients[start:stop]).sum().backward()
    return float(loss.detach())
