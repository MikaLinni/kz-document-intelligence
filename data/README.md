# Data

## Public sources

- `kurumikz/kazakhstan_legal_corpus` contains document-aligned Kazakh and Russian legal texts. The local experiment used 20,000 bilingual records from `kz_legal_parallel.jsonl.gz`. The dataset card states ODC-BY 1.0. [Corpus card](https://huggingface.co/datasets/kurumikz/kazakhstan_legal_corpus)
- `olzhasAl/adilet-legal-qa-kz` provides the legal QA source. Project-specific flattening yielded 13,159 QA rows before filtering. The dataset card states Apache-2.0. [QA dataset card](https://huggingface.co/datasets/olzhasAl/adilet-legal-qa-kz)

This repository contains aggregate results and code. Source texts, QA records, derived Parquet files, training JSONL and the frozen snapshot are intentionally kept local.

## Benchmark preparation

The project split aligned QA by `context_id` so questions sharing the same context remained together. Initial gold IDs sometimes pointed at the wrong article because the QA contexts could span several articles. The repair script used verified document IDs, language, article matching and text similarity to select a new gold chunk. The clean selection kept exact article matches that passed similarity thresholds.

| Local artifact | Size | RU | KK |
|---|---:|---:|---:|
| Structure-aware corpus | 305,375 | 156,760 | 148,615 |
| Frozen TRAIN | 1,087 | 920 | 167 |
| Frozen DEV | 122 | 108 | 14 |
| Frozen TEST | 268 | 244 | 24 |
| TRAIN records after BM25 negative mining | 1,086 | 919 | 167 |
| Candidate chunks across seven verified documents | 3,009 | 1,692 | 1,317 |

See the [methodology](../results/baseline/methodology.md) and the [recorded checksums](../results/baseline/reproducibility.md).

## Local layout

The demo reads the exact corpus snapshot at `data/legal_chunks_20k.parquet`. Training and evaluation also require a private copy of the original frozen bundle at the following project-relative paths.

```text
data/
├── legal_chunks_20k.parquet
└── frozen/baseline_v1/
    ├── retrieval_train_clean.parquet
    ├── retrieval_dev_clean.parquet
    ├── retrieval_test_clean.parquet
    ├── bge_m3_training_records_clean.jsonl
    ├── repair_all_splits.py
    ├── build_clean_splits.py
    ├── prepare_training_data.py
    └── SHA256SUMS.txt
```

From the repository root, run `shasum -a 256 -c data/frozen/baseline_v1/SHA256SUMS.txt` after placing the original bundle locally. The code also checks the corpus hash.

The data-preparation entry points start with the preprocessed chunk corpus and the original aligned splits `data/retrieval_train.parquet`, `data/retrieval_dev.parquet` and `data/retrieval_test.parquet`. Run repair, clean selection and TRAIN negative mining in that order. Their outputs go under `data/`. The original corpus chunker and initial QA-to-corpus alignment are outside this release, so downloading the two public sources alone does not recreate the frozen snapshot. Use a separately prepared snapshot for new experiments and keep the historical hashes for comparison.
