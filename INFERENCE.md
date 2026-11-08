# Experiment Commands and Native Inference Stages

Each experiment has a standalone offline command, from `python E01/run.py --plot` through `python E10/run.py --plot`. These commands evaluate saved complete rankings and regenerate the applicable paired statistics and composite figures without model inference. The sections below describe entry points and scope for rerunning model stages.

## Shared Native Stages

On a Linux server, run `python code/native_stage.py --help`. Set `--work` to a new directory outside the package. Existing directories are rejected to protect the manuscript evidence. This entry point blocks network connections; model files must already be available on the server.

```bash
# Original Jina-v4 retrieval: preserve its encode interface, maximum length, and single-vector output
python code/native_stage.py dense --task rat2rev-en --split test \
  --data /srv/data/ReCaRe --model /srv/models/jina-embeddings-v4 \
  --encoder jina-v4 --batch-size 4 --work /srv/rerun/E01_dense_rat_en

# Official BM25 parameters; validation selections are in E01/configs/original/bm25_selection.*.json
python code/native_stage.py bm25 --task rat2rev-en --split test \
  --data /srv/data/ReCaRe --jars /srv/lucene9.12.1 \
  --k1 0.9 --b 0.4 --work /srv/rerun/E01_bm25_rat_en

# Weighted RRF: top 500 per branch, k=100; dense/BM25 weight ratio is 2 for Rat and 1 for Rev
python code/native_stage.py rrf --task rat2rev-en --split test \
  --input /srv/rerun/E01_bm25_rat_en/experiments/E01/native/bm25/rat2rev-en/test/k0.9_b0.4/top1000.scored.jsonl \
          /srv/rerun/E01_dense_rat_en/ranked.jsonl \
  --weights 1 2 --k 100 --source-depth 500 --candidate-depth 200 \
  --work /srv/rerun/E01_fusion_rat_en

# Cross-encoder scoring over C200; break score ties by descending document ID
python code/native_stage.py rerank --task rat2rev-en --split test \
  --data /srv/data/ReCaRe --model /srv/models/Qwen3-Reranker-8B \
  --reranker qwen3-reranker-8b --input /srv/rerun/E01_fusion_rat_en/ranked.jsonl \
  --candidate-depth 200 --work /srv/rerun/E01_qwen_rat_en
```

The BM25 command above illustrates the official parameters. For a validation-selected row, use that row's recorded parameters. `--encoder` supports the six bundled dense backbones, and `--reranker` supports the original official scorers and Jina-v3.5. `source_provenance.json` records the extracted source definitions and source SHA256 hashes. The adapters retain those definitions rather than substituting an approximate implementation from another library.

The original BM25+RM3 functions, `feedback_weights` and `run_setting`, are in `code/kernels/rm3_kernel.py`. Frozen configurations are in `E01/configs/original/rm3_selected.*.json`. This stage uses Lucene term vectors; bitwise equivalence to Pyserini is not claimed.

```bash
python code/native_stage.py rm3 --task rev2rev-en --split test \
  --data /srv/data/ReCaRe --jars /srv/lucene9.12.1 --k1 0.9 --b 0.4 \
  --rm3-config E01/configs/original/rm3_selected.rev2rev-en.json \
  --work /srv/rerun/E01_rm3_rev_en
```

RM3 fixes the BM25 parameters at 0.9/0.4. See `E01/configs/original/rm3_protocol.json` for the full protocol. This condition is distinct from the BM25 validation-selected row.

## E01: Baselines and System References

Offline: `python E01/run.py --plot`. Native stages proceed through dense/BM25 retrieval, RRF, and the respective cross-encoders. See `E01/configs/rows.json` and `E01/results.json` for each row's source, actual model, and candidate protocol.

The mDPR, mContriever, and official RankGPT values are checked against their reported sources. GPT and historical H are evaluated only from existing complete rankings. No commands for new API calls are provided, and reported values remain distinct from local inference.

## E02: Module Controls

Offline: `python E02/run.py --plot`. A0 reuses original dense retrieval; A1 adds BM25/RRF; A2 uses anchor+5; A3/A4 additionally use Qwen200. Identical predictions are reused by hash. The final two cached GPT rows require no new inference.

For anchor+5, the RRF input order is BM25, original dense ranking, and five generated views. The total dense weight w is divided into w/2 for the original anchor and w/10 for each generated view; BM25 has weight 1. Use the shared RRF entry point with `rrf --weights 1 1 .2 .2 .2 .2 .2` for Rat with w=2, or `1 .5 .1 .1 .1 .1 .1` for Rev with w=1. All inputs must belong to the same task and split. Follow each row's configuration for unanchored conditions, repeated views, and the single long reformulation.

## E03: Query Strategies

Offline: `python E03/run.py --plot`. Saved generated text is in `_shared/generated_queries/`; the original prompt and parsing definitions for the five generation strategies are in `E03/code/query_generation_definitions.py`. Retrieve separately for each saved view, then fuse the rankings with the shared RRF stage. `dense --queries` accepts `_id`/`text` JSONL. Convert the source `query` or each `queries[i]` into a separate view file while preserving query IDs. Do not concatenate five views into one query. The single long reformulation is a separate saved condition.

Packaging and evaluation commands do not restart generation services. The original generation protocol and complete response evidence are retained. Different outputs from a rerun must not replace an existing result row.

## E04: Task-Criterion Controls

Offline: `python E04/run.py --plot`. Recompute differences from complete T0, T1, and T2 rankings produced by the same backend, using 10000 event-cluster bootstrap samples with seed 20261030. Identical T0-budget predictions remain aliases. Criterion definitions are in `code/kernels/llm_ranking.py`; the window algorithm is in `adaptive_kernel.py`. Only existing GPT caches are used, with no entry point for new API inference.

## E05: Retrieval and Reranking Depth

Offline: `python E05/run.py --plot`. Use validation data only. For source depth S=100/300/500/1000, retain the same two retrieval branches and vary `rrf --source-depth`. For cross-encoder depth C=100/200/300/500, vary `rrf --candidate-depth` and `rerank --candidate-depth`, then evaluate at most the first 200 documents. C=100 returns only 100 documents without padding. K=0 is an alias of the C200 baseline, not an additional generation experiment.

## E06: RRF Sensitivity

Offline: `python E06/run.py --plot`. Use the same validation retrieval branches and evaluate `--k 20/60/100` with dense weights 0.5/1/2 and BM25 weight 1. Evaluate and plot only the frozen grid; do not select parameters on test data.

## E07: Retrieval Backbones and Backends (Original E08; Table 4)

Offline: `python E07/run.py --plot`. The first six rows compare original queries and anchor+5 for Jina-v4, Qwen3-Embedding-8B, and Jina-v5-small. Other rows contain only completed conditions under their actual backends.

The two R10 Rev settings use Qwen3.5-9B with exact-permutation constraints. To run the same algorithm manually from a C200 pool:

```bash
python code/native_stage.py native9b --task rev2rev-en --split test \
  --data /srv/data/ReCaRe --model /srv/models/Qwen3.5-9B \
  --input /srv/rerun/C200_rev_en/ranked.jsonl --work /srv/rerun/R10_rev_en_constrained
# For Japanese, use rev2rev-ja and the corresponding Japanese C200 input
```

The protocol is `native-exact-permutation-json-v1-20261106`: T2, full text, fixed top-100 membership, an unchanged tail at positions 101–200, temperature 0/greedy decoding, `thinking=false`, a 2048-token output budget, and a 32768-token context gate. Format failures stop execution without completion or repair. Context overflow triggers window splitting under the recorded rules. Use separate directories for separate runs; do not splice checkpoints from different decoding protocols.

Each setting retains its actual decoding protocol. Differences across decoding settings cannot be attributed entirely to the T2 criterion. Default commands only evaluate existing rankings; they do not start additional local generation or 397B/GPT/DeepSeek requests.

## E08: Historical Fine-Tuning Pairs

Offline: `python E08/run.py --plot`. The original English Jina-v3 base/fine-tuned pairs cover four stages: dense, hybrid, Qwen200, and historical LLM50. Original training code and its required local dependencies are in `E08/code/historical_training/`. Retrieval source retains only native stages, with legacy API and deployment entry points removed. Grouped training data and selection records are in `E08/configs/historical_training/`. The selected checkpoint is `dual_mixed_anchor-step00024`; see `final_selection.json` for its weight hashes.

From the historical source root, the training entry point is `python scripts/run_jina_recovery.py --variant dual_mixed_anchor --max-steps 24 --epochs 3`. Before running it, restore the expected `ReCaRe/`, `runs/finetune/jina_v3_recovery_v2/dataset/`, and original index paths in a separate server workspace, and verify model paths. This is not a one-command cross-machine recovery tool. Base weights are publicly available, but the fine-tuned checkpoint requires identity verification. No public checkpoint link is assumed, and newly trained weights do not become the original checkpoint. Numerical and hardware differences can affect training. The default offline entry point already reproduces all paired differences from archived complete rankings without training or historical LLM50 inference.

## E09: Coverage and the Ranking Gap

Offline: `python E09/run.py --plot`. Independently recompute observed metrics and candidate-pool oracle values for the RRF200, Qwen200, and final LLM100 stages of the same historical H system. The oracle uses qrels only for offline diagnosis, not as retrieval or ranking input. It represents an upper bound rather than achieved model performance.

## E10: Repeatability and Input Order

Offline: `python E10/run.py --plot`. Use the fixed validation25 subset with original order, reversed order, fixed shuffled order, whitespace normalization, and a dropped retrieval view. All observed repeat rankings are retained. Compute means and sample standard deviations from the actual repeat counts; do not fabricate intervals for single observations. Dropping a view changes the candidate pool and must be distinguished from permutations of the same candidate set. The original Qwen3-8B generation service is not automatically restarted.
