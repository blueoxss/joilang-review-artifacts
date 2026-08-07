# Service Pre-mapping evaluation

`gpt_mg/premap.py` is the runtime retriever. This directory contains its training inputs,
evaluation notebook, coverage scripts, and stored measurements.

## Two different evaluations

The notebook evaluation ranks device or device-role documents over 1,200 held-out queries.
The runtime-coverage script instead calls the shipped pre-mapper and asks whether the
selected services still contain every service used by an accepted JOILang scenario.

Do not interpret notebook Recall@10 as runtime service coverage. The two paths use different
document units, budgets, text fields, query handling, and merge rules.

## Inputs

The reviewed package includes:

```text
data/service_list_ver1.1.7_eng.json
data/path_controlled/query_roles.json
data/integrated/finetune_pairs_split.jsonl
data/integrated/finetune_pairs_nosplit.jsonl
../gpt_mg/version0_6/service_list_ver1.5.4_value.json
../gpt_mg/version0_6/service_list_ver1.5.4_function.json
../JOICommands-170.csv
```

The two pair files each contain 2,305 rows. The natural-command catalog has 53 devices, and
the role audit covers every non-empty catalog example with `condition`, `action`, or
`fragment`.

## Stored retrieval results

The default stored run uses 2,305 training queries, seed 2025, dense weight 0.6, and a
top-10 evaluation budget.

| Configuration | Encoder state | nDCG@10 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| `e5-small` + BM25, split | fine-tuned | 0.9830 | 0.9953 | 1.0000 |
| `e5-small` + BM25, no split | fine-tuned | 0.9809 | 0.9878 | 0.9981 |
| `mxbai-embed-xsmall-v1` + BM25, split | released | 0.9798 | 0.9938 | 1.0000 |
| `multilingual-e5-small`, split | released | 0.9783 | 0.9919 | 0.9996 |
| `bge-m3`, split | released | 0.9807 | 0.9925 | 0.9996 |

`results/premap_eval_split_fixes.json` records 0.9941 nDCG@10 for the corrected
fine-tuned split `e5-small` arm. Use the JSON files for full precision, per-pool values,
latency, and provenance.

## Runtime coverage

```bash
python premapping/runtime_coverage.py
```

| Arm | Pair recall | Full GT1 coverage | Full coverage of any GT |
|---|---:|---:|---:|
| released `e5-small` hybrid | 313/500 | 84/169 | 89/169 |
| fine-tuned `e5-split-v7` | 332/500 | 94/169 | 100/169 |
| sparse only | 234/500 | 50/169 | 56/169 |

The script also reports a catalog-and-budget ceiling. That ceiling is arithmetic over the
accepted scenarios, not a retrieval run.

During this review, the sparse path was rerun from the supplied catalogs and reproduced the
stored counts: 234/500 pairs, 50/169 complete GT1 rows, and 56/169 rows complete under at
least one accepted scenario.

## Generation A/B comparison

```bash
python premapping/premap_ab.py --rows 20
```

The script launches separate `run.py` processes with and without `--premap` and compares the
generated service set with `gt1`-`gt3`. It requires an OpenAI API key. A stored
`results/premap_ab.json` was not supplied, so no A/B result is claimed here.

## Notebook

The notebook builds retrieval documents and query pools, trains selected encoders, evaluates
split and no-split layouts, and calls the runtime pre-mapper. Its code-cell outputs were
preserved; Markdown and code comments were shortened without changing code-cell ASTs.
Execution counters were cleared because the notebook was not rerun during review.

Options used by the notebook include:

| Variable | Effect |
|---|---|
| `JOI_FT_ALL=1` | Fine-tune all candidate encoders and evaluate extra search modes. |
| `JOI_FIX_SPLIT=1` | Build role-consistent split pairs. |
| `JOI_RETRAIN=0` | Reuse an existing checkpoint. |

Local checkpoints are written under `premapping/models/` and are not included in the
review package.

## Role labels

`label_query_roles.py` is a one-off data-preparation tool. It calls GPT-4.1-mini at
temperature 0 and writes `data/path_controlled/query_roles.json`. The committed audit is
already present; rerunning the tool requires an API key.

## Integrity check

```bash
python premapping/check_stored_outputs.py
```

The checker verifies the included notebook outputs, normalized execution counters, and
stored JSON/Markdown result files. It does not create or infer missing measurements.
