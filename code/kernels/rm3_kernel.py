from kernel_common import *
from collections import Counter,defaultdict
import re
from bm25_kernel import *


def feedback_weights(query_counts, vectors, scores, fb_terms, original_weight):
    model = defaultdict(float)
    for vector, score in zip(vectors, scores):
        kept = dict(sorted(vector.items(), key=lambda p: (-p[1], p[0]))[:fb_terms])
        norm = sum(kept.values())
        if norm:
            for term, count in kept.items():
                model[term] += count/norm*score
    model = dict(sorted(model.items(), key=lambda p: (-p[1], p[0]))[:fb_terms])
    norm = sum(model.values())
    query_norm = sum(query_counts.values())
    output = defaultdict(float)
    if norm:
        for term, value in model.items():
            output[term] += (1-original_weight)*value/norm
    for term, count in query_counts.items():
        output[term] += original_weight*count/query_norm
    return dict(sorted(output.items()))


def run_setting(cls, analyzer, searcher, reader, index_signature, task, split, config, cache):
    folder = ROOT / f"experiments/E01/native/rm3/{task}/{split}/{identity(config)[:16]}"
    prepared = ROOT / f"inputs/queries/{task}/{split}.jsonl"
    signature = identity({"index": index_signature, "queries": digest(prepared),
                          "config": config, "code": digest(Path(__file__))})
    if (folder/"manifest.json").exists():
        manifest = load(folder/"manifest.json")
        if manifest["signature"] != signature:
            raise ValueError("RM3 protocol changed")
        return folder, load(folder/"metrics.json")
    rows = []
    stored = searcher.storedFields()
    for query in records(prepared):
        counts = Counter(analyze(cls, analyzer, query["text"]))
        if query["_id"] not in cache:
            builder = cls("org.apache.lucene.search.BooleanQuery$Builder")()
            for term, count in sorted(counts.items()):
                builder.add(cls("org.apache.lucene.search.BoostQuery")(
                    cls("org.apache.lucene.search.TermQuery")(
                        cls("org.apache.lucene.index.Term")("contents", term)), float(count)),
                    cls("org.apache.lucene.search.BooleanClause$Occur").SHOULD)
            hits = searcher.search(builder.build(), 21).scoreDocs
            selected = [(str(stored.document(h.doc).get("id")), int(h.doc), float(h.score))
                        for h in hits]
            selected = sorted([v for v in selected if not (
                task.startswith("rev") and v[0] == query["_id"])],
                key=lambda p: (p[2], p[0]), reverse=True)[:10]
            vectors = []
            for _, doc, score in selected:
                terms = reader.termVectors().get(doc, "contents")
                vector = {}
                if terms is not None:
                    enum = terms.iterator()
                    while True:
                        raw = enum.next()
                        if raw is None:
                            break
                        term = str(raw.utf8ToString())
                        if not 2 <= len(term) <= 20:
                            continue
                        if task.endswith("-en") and not re.fullmatch("[a-z0-9]+", term):
                            continue
                        df = reader.docFreq(cls("org.apache.lucene.index.Term")("contents", term))
                        if df/reader.numDocs() > .1:
                            continue
                        vector[term] = int(enum.totalTermFreq())
                vectors.append(vector)
            cache[query["_id"]] = (selected, vectors)
        selected, vectors = cache[query["_id"]]
        n = config["fb_docs"]
        weights = feedback_weights(
            counts, vectors[:n], [v[2] for v in selected[:n]],
            config["fb_terms"], config["original_query_weight"])
        builder = cls("org.apache.lucene.search.BooleanQuery$Builder")()
        for term, weight in weights.items():
            if weight <= 0:
                continue
            builder.add(cls("org.apache.lucene.search.BoostQuery")(
                cls("org.apache.lucene.search.TermQuery")(
                    cls("org.apache.lucene.index.Term")("contents", term)), float(weight)),
                cls("org.apache.lucene.search.BooleanClause$Occur").SHOULD)
        hits = searcher.search(builder.build(), 1001).scoreDocs
        ranked = [(str(stored.document(h.doc).get("id")), float(h.score)) for h in hits]
        ranked = [v for v in ranked if not (task.startswith("rev") and v[0] == query["_id"])]
        ranked.sort(key=lambda p: (p[1], p[0]), reverse=True)
        rows.append({"qid": query["_id"], "ranked": [d for d, _ in ranked[:200]]})
        save(folder / f"feedback/{identity(query['_id'])}.json", {
            "qid": query["_id"], "query_term_counts": dict(counts), "expanded_weights": weights,
            "feedback_documents": [{"id": v[0], "bm25_score": v[2]} for v in selected[:n]],
            "filtered_term_vectors": vectors[:n],
        })
        if len(rows) % 50 == 0:
            print(f"RM3 {task} {split} {config} {len(rows)}", flush=True)
    save_records(folder/"ranked.jsonl", rows)
    result = evaluate(folder/"ranked.jsonl", task, split=split)
    persist_evaluation(folder, result, task, split=split)
    save(folder/"manifest.json", {"complete": True, "signature": signature, "at": now(),
                                 "config": config, "split": split, "task": task,
                                 "index_signature": index_signature, "output_length": 200,
                                 "implementation": "direct Lucene RM3; not bitwise Anserini/Pyserini",
                                 "term_filter": "length 2..20; df<=10%; ascii only for English",
                                 "tie_break": "feedback term frequency desc then term asc"})
    return folder, result.metrics

