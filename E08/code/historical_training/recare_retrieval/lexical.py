from __future__ import annotations

import json
import logging
import re
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .common import fingerprint, write_json

LOGGER = logging.getLogger(__name__)
TOKENIZER_VERSION = "en-alnum-stop-v1"
STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has have "
    "he her hers him his how i if in into is it its may might more most must no nor not "
    "of on or our out shall she should so than that the their them there these they this "
    "those through to under up us was we were what when where which while who will with "
    "would you your".split()
)


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())
            if token not in STOPWORDS]


def top_indices(scores: np.ndarray, count: int, excluded: int | None = None) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32).copy()
    if excluded is not None:
        scores[excluded] = -np.inf
    count = min(count, int(np.isfinite(scores).sum()))
    if count <= 0:
        return []
    # Deterministic ties, including the top-k boundary.
    if count < len(scores):
        threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
        better = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)[:count - len(better)]
        selected = np.concatenate((better, tied))
    else:
        selected = np.arange(len(scores))
    return selected[np.lexsort((selected, -scores[selected]))].tolist()


class BM25:
    def __init__(self, matrix: sparse.csc_matrix, vocabulary: dict[str, int], k1: float, b: float,
                 tokenizer_fn=None):
        self.matrix = matrix
        self.vocabulary = vocabulary
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer_fn or tokenize

    @classmethod
    def build(cls, documents: list[dict], directory: Path, corpus_hash: str,
              k1: float = 1.2, b: float = 0.75, *, tokenizer_fn=None,
              tokenizer_version=TOKENIZER_VERSION) -> "BM25":
        tokenize_document = tokenizer_fn or tokenize
        identity = {"corpus": corpus_hash, "k1": k1, "b": b, "tokenizer": tokenizer_version}
        root = directory / "bm25" / fingerprint(identity)[:20]
        manifest = root / "index.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError("BM25 cache identity mismatch")
            LOGGER.info("Loading cached BM25 index")
            matrix = sparse.load_npz(root / "weights.npz").tocsc()
            if matrix.shape[0] != len(documents):
                raise ValueError("BM25 cache document count mismatch")
            return cls(matrix, saved["vocabulary"], k1, b, tokenizer_fn)
        LOGGER.info("Building BM25 index for %d articles", len(documents))
        vocabulary: dict[str, int] = {}
        rows, columns, frequencies = array("i"), array("i"), array("f")
        lengths = np.zeros(len(documents), dtype=np.float32)
        for index, document in enumerate(documents):
            counts = Counter(tokenize_document(document["text"]))
            lengths[index] = sum(counts.values())
            for term, frequency in counts.items():
                term_id = vocabulary.setdefault(term, len(vocabulary))
                rows.append(index)
                columns.append(term_id)
                frequencies.append(frequency)
        if not vocabulary:
            raise ValueError("Corpus has no BM25 terms")
        matrix = sparse.csr_matrix(
            (np.frombuffer(frequencies, dtype=np.float32),
             (np.frombuffer(rows, dtype=np.int32), np.frombuffer(columns, dtype=np.int32))),
            shape=(len(documents), len(vocabulary)),
        )
        df = np.bincount(matrix.indices, minlength=len(vocabulary))
        idf = np.log1p((len(documents) - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1.0))
        repeated_norm = np.repeat(norm, np.diff(matrix.indptr))
        matrix.data = (idf[matrix.indices] * matrix.data * (k1 + 1)
                       / (matrix.data + repeated_norm)).astype(np.float32)
        matrix = matrix.tocsc()
        root.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(root / "weights.npz", matrix)
        write_json(manifest, {"identity": identity, "vocabulary": vocabulary})
        return cls(matrix, vocabulary, k1, b, tokenizer_fn)

    def search(self, query: str, count: int, excluded: int | None = None) -> list[int]:
        counts = Counter(self.tokenizer(query))
        terms = [(self.vocabulary[word], 1 + np.log(frequency))
                 for word, frequency in counts.items() if word in self.vocabulary]
        scores = np.zeros(self.matrix.shape[0], dtype=np.float32)
        for column, weight in terms:
            start, stop = self.matrix.indptr[column:column + 2]
            scores[self.matrix.indices[start:stop]] += self.matrix.data[start:stop] * weight
        return top_indices(scores, count, excluded)


def reciprocal_rank_fusion(rankings: list[list[int]], weights: list[float],
                           k: int, count: int) -> list[int]:
    if len(rankings) != len(weights):
        raise ValueError("One weight is required per ranking")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, docid in enumerate(ranking, 1):
            if docid not in seen:
                scores[docid] = scores.get(docid, 0.0) + weight / (k + rank)
                seen.add(docid)
    return sorted(scores, key=lambda docid: (-scores[docid], docid))[:count]
