import numpy as np


def _rank_inner_product(
    qids: list[str],
    q_embs: np.ndarray,
    embeddings: np.ndarray,
    ids: list[str],
    *,
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    """Rank documents for each query by inner product using the baseline path.

    Fine-tuned retrieval reuses this helper so the score computation and top-k
    tie handling stay aligned with ``run_search``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if embeddings.shape[0] != len(ids):
        raise ValueError(
            f"index row/doc-id mismatch: {embeddings.shape[0]} embeddings vs {len(ids)} ids"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("cannot rank against an empty embedding index")

    out: dict[str, list[tuple[str, float]]] = {}
    scores = q_embs @ embeddings.T  # (n_q, n_d)
    eff_k = min(top_k, scores.shape[1])
    for i, qid in enumerate(qids):
        s = scores[i]
        idx = np.argpartition(-s, eff_k - 1)[:eff_k]
        idx = idx[np.argsort(-s[idx])]
        out[qid] = [(ids[j], float(s[j])) for j in idx]
    return out

