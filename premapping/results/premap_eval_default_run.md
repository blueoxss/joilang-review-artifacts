# Retrieval evaluation - default_run

- training queries: 2305 (1137 template + 845 natural + 323 clause), seed 2025
- built from: gpt_mg/version0_6/service_list_ver1.5.4_value.json, gpt_mg/version0_6/service_list_ver1.5.4_function.json, premapping/data/service_list_ver1.1.7_eng.json
- fine-tuned: e5/split, e5/nosplit
- evaluation: 1200 held-out queries, dense weight 0.6, top-10, cuda

| Configuration | Encoder | nDCG@10 | Recall@5 | Recall@10 | Latency (ms) | Search |
|---|---|---|---|---|---|---|
| intfloat/e5-small + BM25 (split) | fine-tuned | 0.9834 | 0.9950 | 1.0000 | 8.7 | hybrid |
| intfloat/e5-small + BM25 (no split) | fine-tuned | 0.9806 | 0.9881 | 0.9975 | 7.7 | hybrid |
| mxbai-embed-xsmall-v1 + BM25 (split) | as released | 0.9798 | 0.9938 | 1.0000 | 5.6 | hybrid |
| multilingual-e5-small (split) | as released | 0.9783 | 0.9919 | 0.9996 | 9.2 | dense-only |
| BAAI/bge-m3 (split) | as released | 0.9807 | 0.9925 | 0.9996 | 16.3 | dense-only |

Per-pool nDCG@10 / Recall@5:

| Configuration | template singles | natural singles | paraphrase singles | compound-template | compound-paraphrase |
|---|---|---|---|---|---|
| intfloat/e5-small + BM25 (split) | 1.0000 / 1.0000 | 0.9668 / 1.0000 | 1.0000 / 1.0000 | 0.9950 / 1.0000 | 0.9504 / 0.9800 |
| intfloat/e5-small + BM25 (no split) | 1.0000 / 1.0000 | 0.9703 / 1.0000 | 1.0000 / 1.0000 | 0.9710 / 0.9828 | 0.9619 / 0.9694 |
| mxbai-embed-xsmall-v1 + BM25 (split) | 0.9974 / 1.0000 | 0.9668 / 1.0000 | 0.9979 / 1.0000 | 0.9915 / 0.9972 | 0.9433 / 0.9778 |
| multilingual-e5-small (split) | 0.9974 / 1.0000 | 0.9601 / 0.9906 | 0.9979 / 1.0000 | 0.9858 / 0.9944 | 0.9453 / 0.9767 |
| BAAI/bge-m3 (split) | 1.0000 / 1.0000 | 0.9601 / 0.9906 | 0.9990 / 1.0000 | 0.9902 / 0.9961 | 0.9479 / 0.9772 |
