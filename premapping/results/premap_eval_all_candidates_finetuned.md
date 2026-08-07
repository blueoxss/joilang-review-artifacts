# Retrieval evaluation - all_candidates_finetuned

- training queries: 2305 (1137 template + 845 natural + 323 clause), seed 2025
- built from: gpt_mg/version0_6/service_list_ver1.5.4_value.json, gpt_mg/version0_6/service_list_ver1.5.4_function.json, premapping/data/service_list_ver1.1.7_eng.json
- fine-tuned: e5/split, e5/nosplit, mxbai/split, multilingual/split, bge-m3/split
- evaluation: 1200 held-out queries {'template singles': 142, 'natural singles': 106, 'paraphrase singles': 352, 'compound-template': 300, 'compound-paraphrase': 300}, w = 0.6, top-10, cuda

| Configuration | Encoder | nDCG@10 | Recall@5 | Recall@10 | Latency (ms) | Search |
|---|---|---|---|---|---|---|
| intfloat/e5-small + BM25 (split) | fine-tuned | 0.9830 | 0.9953 | 1.0000 | 10.3 | hybrid |
| intfloat/e5-small + BM25 (no split) | fine-tuned | 0.9809 | 0.9878 | 0.9981 | 7.8 | hybrid |
| mxbai-embed-xsmall-v1 + BM25 (split) | fine-tuned | 0.9828 | 0.9947 | 1.0000 | 6.4 | hybrid |
| multilingual-e5-small (split) | fine-tuned | 0.9845 | 0.9944 | 1.0000 | 10.2 | dense-only |
| BAAI/bge-m3 (split) | fine-tuned | 0.9841 | 0.9942 | 0.9992 | 17.8 | dense-only |
| intfloat/e5-small (split, dense-only) | fine-tuned | 0.9847 | 0.9953 | 1.0000 | 9.7 | dense-only |
| mxbai-embed-xsmall-v1 (split, dense-only) | fine-tuned | 0.9838 | 0.9944 | 1.0000 | 6.7 | dense-only |
| multilingual-e5-small + BM25 (split) | fine-tuned | 0.9826 | 0.9947 | 1.0000 | 9.1 | hybrid |
| BAAI/bge-m3 + BM25 (split) | fine-tuned | 0.9838 | 0.9947 | 1.0000 | 15.8 | hybrid |
| intfloat/e5-small (no split, dense-only) | fine-tuned | 0.9783 | 0.9771 | 0.9925 | 7.6 | dense-only |

Per-pool nDCG@10 / Recall@5:

| Configuration | template singles | natural singles | paraphrase singles | compound-template | compound-paraphrase |
|---|---|---|---|---|---|
| intfloat/e5-small + BM25 (split) | 1.0000 / 1.0000 | 0.9668 / 1.0000 | 1.0000 / 1.0000 | 0.9932 / 1.0000 | 0.9505 / 0.9811 |
| intfloat/e5-small + BM25 (no split) | 1.0000 / 1.0000 | 0.9668 / 1.0000 | 1.0000 / 1.0000 | 0.9728 / 0.9828 | 0.9626 / 0.9683 |
| mxbai-embed-xsmall-v1 + BM25 (split) | 1.0000 / 1.0000 | 0.9668 / 1.0000 | 1.0000 / 1.0000 | 0.9947 / 1.0000 | 0.9484 / 0.9789 |
| multilingual-e5-small (split) | 1.0000 / 1.0000 | 0.9807 / 1.0000 | 1.0000 / 1.0000 | 0.9949 / 1.0000 | 0.9499 / 0.9778 |
| BAAI/bge-m3 (split) | 1.0000 / 1.0000 | 0.9754 / 0.9906 | 1.0000 / 1.0000 | 0.9947 / 1.0000 | 0.9504 / 0.9800 |
| intfloat/e5-small (split, dense-only) | 1.0000 / 1.0000 | 0.9814 / 1.0000 | 1.0000 / 1.0000 | 0.9948 / 1.0000 | 0.9506 / 0.9811 |
| mxbai-embed-xsmall-v1 (split, dense-only) | 1.0000 / 1.0000 | 0.9779 / 1.0000 | 1.0000 / 1.0000 | 0.9948 / 1.0000 | 0.9482 / 0.9778 |
| multilingual-e5-small + BM25 (split) | 1.0000 / 1.0000 | 0.9633 / 1.0000 | 1.0000 / 1.0000 | 0.9929 / 1.0000 | 0.9504 / 0.9789 |
| BAAI/bge-m3 + BM25 (split) | 1.0000 / 1.0000 | 0.9709 / 1.0000 | 1.0000 / 1.0000 | 0.9949 / 1.0000 | 0.9507 / 0.9789 |
| intfloat/e5-small (no split, dense-only) | 1.0000 / 1.0000 | 0.9779 / 1.0000 | 1.0000 / 1.0000 | 0.9641 / 0.9606 | 0.9571 / 0.9478 |
