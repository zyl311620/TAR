from urllib.error import HTTPError
from llm_ranking import ranking_messages,parse_permutation,apply_permutation,window_schedule


class WindowScopeFailure(ValueError):
    """A fixed protocol cannot fit a query; retain this explicit failure."""


def clipped(text, cap, tokenizer):
    if not cap:
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(tokens[:cap], skip_special_tokens=True) if len(tokens) > cap else text


def rerank_query(client, task, query, ids, corpus, *, criterion="T1", depth=100,
                 fixed_window=False, passage_cap=0, query_cap=0, tokenizer=None, namespace="main",
                 passage_word_cap=0, native_singleton_identity=False, deterministic_windows=None):
    if native_singleton_identity and fixed_window:
        raise ValueError("Singleton identity is limited to adaptive native windows")
    query = clipped(query, query_cap, tokenizer)
    texts = {
        doc: clipped(" ".join(corpus[doc].split()[:passage_word_cap]) if passage_word_cap else corpus[doc],
                     passage_cap, tokenizer)
        for doc in ids[:depth]
    }
    calls = []

    def rank_window(pool, allow_split):
        # A singleton has exactly one permutation. This explicit native-only
        # operation is not a parsed/repaired model response or an HTTP attempt.
        if native_singleton_identity and allow_split and len(pool) == 1:
            if deterministic_windows is not None:
                deterministic_windows.append({
                    "operation": "singleton_identity", "candidate_ids": list(pool),
                    "model_request": False, "possible_permutations": 1,
                })
            return list(pool)
        prompt = ranking_messages(task, query, [texts[d] for d in pool], criterion)
        if sum(len(m["content"].encode("utf-8")) for m in prompt) > 1400000:
            if not allow_split or len(pool) <= 1:
                raise WindowScopeFailure("Fixed window exceeds byte admission limit")
            return rank_sliding(pool, max(1, len(pool)//2))
        for retry in range(2):
            try:
                raw, key = client.request(
                    prompt, namespace=namespace if retry == 0 else namespace+f":format-retry:{retry}")
            except HTTPError as exc:
                body = getattr(exc, "paper_error_body", "").lower()
                context_failure = exc.code == 413 or (
                    exc.code == 400 and any(marker in body for marker in (
                        "context length", "maximum context", "max_model_len", "too many tokens")))
                if allow_split and len(pool) > 1 and context_failure:
                    return rank_sliding(pool, max(1, len(pool)//2))
                if context_failure:
                    raise WindowScopeFailure("Fixed window exceeds model context") from exc
                raise
            calls.append({"request_identity": key, "documents": len(pool),
                          "usage": raw.get("usage"), "response_id": raw.get("id"),
                          "cache_hit": client.last_was_cached})
            try:
                permutation = parse_permutation(raw["choices"][0]["message"]["content"], len(pool))
                return apply_permutation(pool, permutation)
            except (ValueError, KeyError, TypeError):
                if retry == 1:
                    if allow_split and len(pool) > 1:
                        return rank_sliding(pool, max(1, len(pool)//2))
                    raise
        raise RuntimeError("Window could not be ranked")

    def rank_sliding(pool, size, allow_nested=True):
        current = list(pool)
        for start, end in window_schedule(len(current), size, max(1, size//2)):
            current[start:end] = rank_window(current[start:end], allow_nested)
        return current

    head = rank_sliding(ids[:depth], 20, False) if fixed_window else rank_window(ids[:depth], True)
    if set(head) != set(ids[:depth]) or len(head) != depth:
        raise ValueError("LLM candidate permutation invariant failed")
    output = head + ids[depth:200]
    if set(output[:100]) != set(ids[:100]) or output[100:] != ids[100:200]:
        raise ValueError("Top100 membership or tail changed")
    return output, calls

