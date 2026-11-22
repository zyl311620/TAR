import math


def rrf(lists: list[list[str]], weights: list[float], k=100, count=200):
    if len(lists) != len(weights) or any(weight < 0 for weight in weights):
        raise ValueError("Invalid RRF branches")
    terms = {}
    for ranking, weight in zip(lists, weights):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate IDs inside an RRF source")
        for rank, docid in enumerate(ranking, 1):
            terms.setdefault(docid, []).append(weight / (k + rank))
    scores = {d: math.fsum(v) for d, v in terms.items()}
    return sorted(scores, key=lambda d: (scores[d], d), reverse=True)[:count]

