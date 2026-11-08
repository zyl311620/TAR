# Models, Data, and Runtime Environments

## Data

Dataset: [kasys/ReCaRe](https://huggingface.co/datasets/kasys/ReCaRe). The package includes the manuscript's qrels, fixed queries, validation25 scope, and event-cluster assignments. Offline evaluation does not require the full corpora. Prepare the full dataset on a server when needed:

```bash
hf download kasys/ReCaRe --repo-type dataset --local-dir /srv/data/ReCaRe
```

Use the query text in `data/queries/`. Preserve the recorded procedure used to recover text from the source data. Keep the original train/validation/test splits and do not select parameters using test results.

## Base Model Repositories

The repositories below come from the model configurations and official IDs used in the study. Downloading the latest default revision does not guarantee bitwise reproduction. Check the weight-file hashes in `configs/original/` and `_shared/model_identities/`, and record the downloaded commit ID. The moving `main` branch is not a frozen revision.

| Model | Repository / download page | Experiments |
|---|---|---|
| multilingual-e5-base | [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | E01 |
| BGE-M3 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | E01 |
| Jina-v3 | [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) | E01, E08 |
| Jina-v4 | [jinaai/jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | E01–E03, E05–E07, historical H |
| Jina-v5-small | [jinaai/jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) | E01, E07 |
| Qwen3-Embedding-8B | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | E01, E07 |
| BGE-Reranker-v2-M3 | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | E01 |
| Jina Reranker v2 | [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | E01 |
| Jina Reranker v3.5 | [jinaai/jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) | E01 |
| Qwen3-Reranker-0.6B/4B/8B | [0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), [4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B), [8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | E01, E02, E05, E07, historical H |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Saved views in E03; E07, E10 |
| Qwen3.5-9B | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | E07, including the constrained R10 Rev settings |

Download examples, to be run on a server only:

```bash
# Replace COMMIT_ID with the actual revision verified against the recorded hashes.
hf download Qwen/Qwen3-Reranker-8B --revision COMMIT_ID --local-dir /srv/models/Qwen3-Reranker-8B
hf download jinaai/jina-embeddings-v4 --revision COMMIT_ID --local-dir /srv/models/jina-embeddings-v4
hf download Qwen/Qwen3.5-9B --revision COMMIT_ID --local-dir /srv/models/Qwen3.5-9B
```

Follow the repository instructions if access requires a license agreement or login. No access tokens are included. In the original study, weights were stored at `/root/user/cs_tcci_zhouyulin/LLM_list/` on the server. The standalone adapter accepts a different local path through `--model`.

## Historical Conditions Without Substitute Downloads

The official mDPR, mContriever, and RankGPT rows retain their reported sources. The cached GPT responses identify the model as `gpt-6.1-sol`, which is not a downloadable checkpoint. Historical DeepSeek outputs are also retained only as existing evidence. Do not use a new service to fill or replace these rows. This package excludes 397B models and their unfinished experiments.

The historical fine-tuned Jina-v3 weights in E08 refer to the original study checkpoint, not the public Jina-v3 base model. Complete rankings and paired base/fine-tuned results are included. Training and retrieval scripts are in `E08/code/historical_training/`. For model execution, obtain the original checkpoint or retrain with the recorded training data and configuration. A new run selected for its best test score cannot replace the original checkpoint. This package does not distribute those weights or claim a public download link for them.

## Environments

The minimal offline environment is specified in `requirements.txt`. Use separate GPU environments where required:

- Retrieval and cross-encoder stages use the original study's PyTorch/Transformers environments. Read each model's configuration and source requirements first. The Qwen3/Jina-v4 code uses the Transformers 4.57 series as a reference; Jina-v4 requires its native remote code and compatible FlashAttention 2. A single dependency version may not support every model.
- Qwen3.5-9B requires a Transformers 5.2 or later 5.x environment supporting `AutoModelForImageTextToText`. Consult `_shared/runtime_identities/` for the actual library versions, PyTorch version, and accelerator provenance. Preserve BF16, SDPA, `thinking=false`, greedy decoding, the 2048-token output budget, and the 32768-token context gate.
- Qwen3-8B generation uses the recorded vLLM format protocol. The exact-permutation token mask applies to the Table 4 R10 Rev conditions. Default reproduction evaluates saved responses without restarting native generation services.
- BM25/RM3 use three Lucene 9.12.1 JARs: `lucene-core`, `lucene-analysis-common`, and `lucene-analysis-kuromoji`, together with a compatible JDK and JPype1. Obtain the JARs from [Maven Central](https://repo.maven.apache.org/maven2/org/apache/lucene/), place them in one directory, and pass it through `--jars`. A full Pyserini installation is not required.
- GPU stages also require `torch`, `transformers`, `tqdm`, `Pillow`, and model-specific dependencies. These are separate from the minimal offline environment. The latest dependency versions have not been validated across all models.

The accelerated R10 execution used versioned FLA and fixednorm implementations; their identities are retained with the results. The immutable executed scripts, constraint modules, and fixednorm source are in `E07/code/executed_R10/`, verified file by file against the original source manifest. This directory supports auditing rather than standalone server deployment. The manual adapter provides a rerun entry point, but no GPU models were loaded during packaging. Archived manifests define the identities of the reported results.
