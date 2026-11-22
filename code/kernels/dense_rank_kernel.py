

def rank_vectors(vectors, bank, ids, qids, task):
    import numpy as np
    scores = vectors @ bank.T
    positions = {d: i for i, d in enumerate(ids)}
    output = []
    for i, qid in enumerate(qids):
        values = scores[i]
        if task.startswith("rev") and qid in positions:
            values[positions[qid]] = -np.inf
        selected = np.argpartition(-values, 999)[:1000]
        boundary = values[selected].min()
        above = np.flatnonzero(values > boundary).tolist()
        equal = sorted(np.flatnonzero(values == boundary).tolist(), key=lambda j: ids[j],
                       reverse=True)[:1000-len(above)]
        chosen = sorted(above+equal, key=lambda j: (float(values[j]), ids[j]), reverse=True)
        output.append([ids[j] for j in chosen])
    return output

