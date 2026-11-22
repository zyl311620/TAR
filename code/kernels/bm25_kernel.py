from kernel_common import *
from collections import Counter
import sys


def lucene():
    dependencies = ROOT / "framework/vendor_ja"
    sys.path.insert(0, str(dependencies))
    import jpype
    jars = sorted((dependencies / "lucene").glob("*.jar"))
    if len(jars) != 3:
        raise ValueError("Missing pinned Lucene jars")
    if not jpype.isJVMStarted():
        jpype.startJVM("-Xmx4g", classpath=[str(p) for p in jars])
    cls = jpype.JClass
    cls("org.apache.lucene.search.IndexSearcher").setMaxClauseCount(65536)
    return cls, {p.name: digest(p) for p in jars}


def build_index(cls, lang: str, jars: dict):
    corpus = DATA / f"corpus-{lang}/corpus.jsonl"
    analyzer = cls("org.apache.lucene.analysis.en.EnglishAnalyzer")() if lang == "en" else (
        cls("org.apache.lucene.analysis.ja.JapaneseAnalyzer")())
    folder = ROOT / f"indexes/lucene_bm25/{lang}"
    signature = identity({"corpus": digest(corpus), "analyzer": str(analyzer.getClass()),
                          "jars": jars, "field": "contents", "docvectors": True})
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        if load(manifest_path)["signature"] != signature:
            raise ValueError("Lucene index identity differs")
        return analyzer, folder, signature
    folder.mkdir(parents=True, exist_ok=True)
    directory = cls("org.apache.lucene.store.FSDirectory").open(
        cls("java.nio.file.Paths").get(str(folder / "index")))
    config = cls("org.apache.lucene.index.IndexWriterConfig")(analyzer)
    config.setOpenMode(cls("org.apache.lucene.index.IndexWriterConfig$OpenMode").CREATE)
    config.setRAMBufferSizeMB(256.)
    writer = cls("org.apache.lucene.index.IndexWriter")(directory, config)
    field_type = cls("org.apache.lucene.document.FieldType")(
        cls("org.apache.lucene.document.TextField").TYPE_NOT_STORED)
    field_type.setStoreTermVectors(True)
    field_type.freeze()
    count = 0
    try:
        for row in records(corpus):
            document = cls("org.apache.lucene.document.Document")()
            document.add(cls("org.apache.lucene.document.StringField")(
                "id", row["_id"], cls("org.apache.lucene.document.Field$Store").YES))
            document.add(cls("org.apache.lucene.document.Field")("contents", row["text"], field_type))
            writer.addDocument(document)
            count += 1
            if count % 5000 == 0:
                print(f"LUCENE_INDEX {lang} {count}", flush=True)
        writer.commit()
    finally:
        writer.close()
        directory.close()
    save(manifest_path, {"complete": True, "signature": signature, "articles": count,
                         "analyzer": str(analyzer.getClass()), "jars": jars, "at": now()})
    return analyzer, folder, signature


def analyze(cls, analyzer, text: str):
    stream = analyzer.tokenStream("contents", text)
    term = stream.addAttribute(cls("org.apache.lucene.analysis.tokenattributes.CharTermAttribute").class_)
    tokens = []
    try:
        stream.reset()
        while stream.incrementToken():
            tokens.append(str(term.toString()))
        stream.end()
    finally:
        stream.close()
    return tokens


def search(cls, analyzer, folder: Path, task: str, split: str,
           k1: float, b: float, index_signature: str):
    output = ROOT / f"experiments/E01/native/bm25/{task}/{split}/k{k1}_b{b}"
    output.mkdir(parents=True, exist_ok=True)
    prepared = ROOT / f"inputs/queries/{task}/{split}.jsonl"
    signature = identity({"index": index_signature, "query": digest(prepared),
                          "k1": k1, "b": b, "query_tf_boost": True})
    if (output / "manifest.json").exists():
        previous = load(output / "manifest.json")
        if previous["signature"] == signature and previous["complete"]:
            return output, load(output / "official_metrics.json")["metrics"]
        raise ValueError("BM25 cached query protocol differs")
    directory = cls("org.apache.lucene.store.FSDirectory").open(
        cls("java.nio.file.Paths").get(str(folder / "index")))
    reader = cls("org.apache.lucene.index.DirectoryReader").open(directory)
    searcher = cls("org.apache.lucene.search.IndexSearcher")(reader)
    searcher.setSimilarity(cls("org.apache.lucene.search.similarities.BM25Similarity")(k1, b))
    stored = searcher.storedFields()
    rows = []
    try:
        for item in records(prepared):
            builder = cls("org.apache.lucene.search.BooleanQuery$Builder")()
            counts = Counter(analyze(cls, analyzer, item["text"]))
            for term, count in sorted(counts.items()):
                clause = cls("org.apache.lucene.search.TermQuery")(
                    cls("org.apache.lucene.index.Term")("contents", term))
                boosted = cls("org.apache.lucene.search.BoostQuery")(clause, float(count))
                builder.add(boosted, cls("org.apache.lucene.search.BooleanClause$Occur").SHOULD)
            hits = searcher.search(builder.build(), 1001).scoreDocs
            scored = []
            for hit in hits:
                docid = str(stored.document(hit.doc).get("id"))
                if task.startswith("rev") and docid == item["_id"]:
                    continue
                scored.append((docid, float(hit.score)))
            scored.sort(key=lambda pair: (pair[1], pair[0]), reverse=True)
            scored = scored[:1000]
            rows.append({"qid": item["_id"], "ranked": [d for d, _ in scored],
                         "scores": [s for _, s in scored]})
            if len(rows) % 25 == 0:
                print(f"BM25 {task} {split} k={k1} b={b} queries={len(rows)}", flush=True)
    finally:
        reader.close()
        directory.close()
    save_records(output / "top1000.scored.jsonl", rows)
    save_records(output / "ranked.jsonl", (
        {"qid": row["qid"], "ranked": row["ranked"][:200]} for row in rows))
    evaluated = evaluate(output / "ranked.jsonl", task, split=split)
    if split == "test":
        persist_evaluation(output, evaluated, task)
    else:
        save(output / "official_metrics.json", asdict(evaluated))
    save(output / "manifest.json", {
        "complete": True, "signature": signature, "k1": k1, "b": b, "at": now(),
        "task": task, "split": split, "output_length": 200, "model_api_calls": 0,
        "query_builder": "query-TF weighted OR, Anserini BagOfWords construction",
        "implementation": "direct Lucene 9.12.1, not a claim of bitwise Pyserini equivalence",
    })
    return output, evaluated.metrics

