# TAR Experiment Reproduction Package: E01–E10

## 1. Quick Start: Offline Evaluation and Plotting

Open a terminal in this `experiments` directory and use Python 3.11. Offline reproduction requires no GPU, model weights, API keys, or original working directory.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py --experiment all --check --plot
python code/audit_evidence.py
```

These commands verify SHA256 checksums, recompute metrics for all available rankings with the bundled official evaluator, and produce booktabs-style table fragments or composite figures. IP network connections are blocked, and no models are called. Outputs are written to `outputs/` without replacing the packaged rankings or manuscript files.

To run a single experiment:

```bash
python E07/run.py --plot
# Equivalent to python reproduce.py --experiment E07 --plot
```

| Manuscript ID | Experiment | Original study ID | Command | Main outputs |
|---|---|---|---|---|
| E01 | Baselines and complete-system references | E01 | `python E01/run.py --plot` | Table 2 and metrics for each setting |
| E02 | Module controls and interactions | E02 | `python E02/run.py --plot` | Table 3 and metrics for each setting |
| E03 | Query strategies and anchored views | E03 | `python E03/run.py --plot` | Composite figure covering three metrics |
| E04 | Task criteria under a fixed cached backend | E04 | `python E04/run.py --plot` | Paired differences and event-cluster 95% confidence intervals |
| E05 | Retrieval-source and cross-encoder depth | E05 | `python E05/run.py --plot` | Validation curves for both depth factors |
| E06 | RRF parameter sensitivity | E07 | `python E06/run.py --plot` | Validation heatmaps |
| E07 | Retrieval backbones and ranking backends | E08 | `python E07/run.py --plot` | Table 4, including R10 |
| E08 | Historical fine-tuning pairs | E10 | `python E08/run.py --plot` | Paired differences across four stages in English |
| E09 | Candidate coverage and the ranking gap | E11 | `python E09/run.py --plot` | Three stages of historical H and the oracle gap |
| E10 | Repeatability and input-order sensitivity | E12 | `python E10/run.py --plot` | Means and sample standard deviations on validation25 |

`experiment_map.csv` provides a machine-readable mapping between manuscript and original study IDs. Original filenames, run IDs, and internal LaTeX labels may retain the original numbering; the metadata records these identities. In particular, the original E09 transfer experiment is distinct from manuscript E09, which studies coverage.

## 2. Directory Structure

```text
README.md / MODELS.md / INFERENCE.md  Usage, model downloads, and inference instructions
reproduce.py                         Unified offline entry point
experiment_map.csv                   Manuscript IDs, original IDs, and table/figure mapping
release_manifest.json                Distribution scope and provenance
file_manifest.json                   SHA256 checksums and sizes; outputs are regenerable
source_provenance.json               Original paths, hashes, and code extraction records
E01/ ... E10/
  run.py                             Standalone entry point for the experiment
  configs/rows.json                   Only configurations used in the manuscript
  results.json                       Full-precision metrics, tasks, splits, rankings, and sources
  evidence/                          Run manifests, per-query results, request indexes, checkpoints
  paper/                             Manuscript table fragments or PDF/PNG/SVG figures
_shared/rankings/                    Complete rankings deduplicated by SHA256
_shared/requests/                    Referenced raw requests/responses, losslessly gzipped
_shared/generated_queries/           Completed query expansions and generation manifests
_shared/model_identities/            Weight-file hashes; no model weights
data/                               Qrels, fixed queries, and event-cluster assignments
code/                               Official evaluator, plotting, native adapters, algorithms
paper/                              Unmodified snapshot of the main TeX and referenced figures
audit/                              Manuscript numeric checks and original audit records
outputs/                            Regenerated results from the reproduction commands
```

Identical predictions are stored once by hash and explicitly referenced in each experiment's `results.json`. Aliases are not independent runs. `source_row_id` preserves the original study identity, while `row_id` follows the manuscript numbering.

## 3. Metrics and Result Interpretation

- JSON/CSV metrics use a 0–1 scale; manuscript values are multiplied by 100. The metrics are nDCG@10, R@100, and MAP.
- Complete rankings are evaluated with the official implementation, with at most 200 documents and full query coverage. Self-matches are removed for Rev tasks. Missing queries, duplicate candidates, and split mismatches are rejected.
- E05/E06 use validation data. E10 uses the fixed validation25 subset, not the full test set. E08 includes only the two English settings completed in the original study.
- The mDPR, mContriever, and two official RankGPT conditions in E01 are marked `official_reported`. Their official metric sources are included; local complete rankings are not claimed. Reproduction checks the source values and labels these rows separately.
- GPT and historical DeepSeek outputs retain their actual backend labels. Only existing results and evidence are read. This package contains no external model clients, keys, probes, or automatic cache completion. The manuscript does not report 397B conditions, so they are excluded.
- Table 4 consistently uses the model name Qwen3.5-9B. The R10 Rev settings use exact-permutation constraints and cover 503 English and 535 Japanese queries. Comparisons across decoding settings do not isolate task-criterion effects; see the corresponding `results.json` for the settings and sources.
- Equal R@100 values can arise from fixed top-100 membership: a permutation changes order without changing the retrieved set. Equal recall alone does not establish that two ranking runs are identical.

Each experiment produces either a table or a composite figure, following the manuscript. `metrics.csv` provides auditable numeric data rather than an additional manuscript table. Table fragments come from the current author-formatted TeX and preserve its boldface, footnotes, and layout. Independently recomputed full-precision metrics are checked against these fragments.

## 4. Model Inference and Downloads

See [MODELS.md](MODELS.md) for model repositories, download commands, weight identities, and environment requirements. See [INFERENCE.md](INFERENCE.md) for the stages and commands used by each experiment. GPU inference adapters run only on Linux servers and require a new output directory outside this package. They do not overwrite manuscript results. No inference or model downloads were performed during packaging or acceptance checks.

Evaluating historical caches does not call the original APIs, and checking officially reported values is not a local reproduction. Public base weights cannot substitute for the historical fine-tuned checkpoint. Its complete rankings support offline evaluation, and E08 includes the relevant training source and provenance. Results from a different checkpoint must not replace the original row.

Dataset: [kasys/ReCaRe](https://huggingface.co/datasets/kasys/ReCaRe). Official code: [kasys-lab/ReCaRe](https://github.com/kasys-lab/ReCaRe). Full corpora and model weights are not bundled; follow their respective licenses. Dataset cards and official code licenses are retained where supplied by the source directories.
