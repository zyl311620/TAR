import inspect
from kernel_common import ROOT as STUDY,digest,now,save


def optimize(scorer, alias):
    if getattr(scorer.spec, "family", None) != "qwen3-lm":
        return
    original = scorer.model.forward
    if "logits_to_keep" not in inspect.signature(original).parameters:
        return
    probes = [
        [("legal obligation and scope", "An authority shall publish the conditions and exceptions.")],
        [("関連条文の改正と適用範囲", "この法律は、行政機関の権限及び義務に関する事項を定める。")],
        [(" ".join(["actor obligation condition exception"]*180),
          " ".join(["The authority shall apply the rule subject to the exception."]*500))],
    ]
    before = [scorer.score_pairs(p) for p in probes]

    def final_position_only(*args, **kwargs):
        kwargs.setdefault("logits_to_keep", 1)
        kwargs.setdefault("use_cache", False)
        return original(*args, **kwargs)

    scorer.model.forward = final_position_only
    try:
        after = [scorer.score_pairs(p) for p in probes]
        exact = before == after
    except Exception:
        scorer.model.forward = original
        raise
    if not exact:
        scorer.model.forward = original
    save(STUDY/f"audit/reranker_forward_equivalence/{alias}.json", {
        "at": now(), "before": before, "after": after, "bitwise_equal": exact,
        "optimization_enabled": exact, "unchanged": "weights, dtype, tokenizer, prompt, truncation, scoring",
        "optimization": "compute only final-position logits; do not return unused KV cache",
        "code_sha256": digest(__import__("pathlib").Path(__file__)),
        "does_not_recompute_or_replace_existing_scores": True,
    })
    print("RERANK_FINAL_POSITION_OPTIMIZATION " + ("VERIFIED" if exact else "DISABLED"), flush=True)

