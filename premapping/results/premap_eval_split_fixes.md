# Retrieval evaluation - split_fixes

- training queries: 2305 (1137 template + 845 natural + 323 clause), seed 2025
- built from: gpt_mg/version0_6/service_list_ver1.5.4_value.json, gpt_mg/version0_6/service_list_ver1.5.4_function.json, premapping/data/service_list_ver1.1.7_eng.json
- fine-tuned: e5/split, e5/nosplit
- evaluation: 1200 held-out queries {'template singles': 142, 'natural singles': 106, 'paraphrase singles': 352, 'compound-template': 300, 'compound-paraphrase': 300}, w = 0.6, top-10, cuda

| Configuration | Encoder | nDCG@10 | Recall@5 | Recall@10 | Latency (ms) | Search |
|---|---|---|---|---|---|---|
| intfloat/e5-small + BM25 (split) | fine-tuned | 0.9941 | 1.0000 | 1.0000 | 9.6 | hybrid |
| intfloat/e5-small + BM25 (no split) | fine-tuned | 0.9809 | 0.9878 | 0.9981 | 7.7 | hybrid |
| mxbai-embed-xsmall-v1 + BM25 (split) | as released | 0.9914 | 0.9993 | 1.0000 | 5.8 | hybrid |
| multilingual-e5-small (split) | as released | 0.9900 | 0.9975 | 0.9996 | 8.7 | dense-only |
| BAAI/bge-m3 (split) | as released | 0.9923 | 0.9982 | 0.9996 | 15.1 | dense-only |

Per-pool nDCG@10 / Recall@5:

| Configuration | template singles | natural singles | paraphrase singles | compound-template | compound-paraphrase |
|---|---|---|---|---|---|
| intfloat/e5-small + BM25 (split) | 1.0000 / 1.0000 | 0.9668 / 1.0000 | 1.0000 / 1.0000 | 0.9932 / 1.0000 | 0.9948 / 1.0000 |
| intfloat/e5-small + BM25 (no split) | 1.0000 / 1.0000 | 0.9668 / 1.0000 | 1.0000 / 1.0000 | 0.9728 / 0.9828 | 0.9626 / 0.9683 |
| mxbai-embed-xsmall-v1 + BM25 (split) | 0.9974 / 1.0000 | 0.9668 / 1.0000 | 0.9979 / 1.0000 | 0.9915 / 0.9972 | 0.9897 / 1.0000 |
| multilingual-e5-small (split) | 0.9974 / 1.0000 | 0.9601 / 0.9906 | 0.9979 / 1.0000 | 0.9858 / 0.9944 | 0.9919 / 0.9989 |
| BAAI/bge-m3 (split) | 1.0000 / 1.0000 | 0.9601 / 0.9906 | 0.9990 / 1.0000 | 0.9902 / 0.9961 | 0.9944 / 1.0000 |
