# Dataset Cleaning — Issue #2

## Source

- Dataset: `ronantakizawa/github-codereview`
- Revision: `c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca`
- Language: Python
- Base model tokenizer: `Qwen/Qwen2.5-7B-Instruct`
- Tokenizer revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Maximum full sequence length: 2,048 tokens

## Cleaning

The preprocessing pipeline:

- preserves the original train, validation, and test splits;
- keeps Python examples only;
- removes empty or malformed diffs;
- removes empty review comments;
- removes clearly external-context-dependent comments;
- removes exact duplicate `(diff, review comment)` examples within each split
  (case-sensitive, after normalizing line endings and surrounding whitespace);
- removes cross-split duplicate examples to prevent evaluation leakage;
- removes examples whose full chat-formatted sequence exceeds 2,048 tokens;
- preserves both actionable reviews and `No issues found.` examples;
- does not apply a quality-score threshold.

## Final counts

| Split | Python before cleaning | Final |
|---|---:|---:|
| Train | 82,288 | 52,441 |
| Validation | 2,639 | 1,938 |
| Test | 2,609 | 1,606 |

## Cross-split deduplication

| Split | Removed |
|---|---:|
| Validation vs Train | 688 |
| Test vs Train/Validation | 972 |

After filtering:

- Train ↔ Validation: 0 exact overlaps
- Train ↔ Test: 0 exact overlaps
- Validation ↔ Test: 0 exact overlaps

## Reproducibility

Preprocessing is deterministic and uses pinned dataset and tokenizer
revisions. Both revisions are recorded in `data/processed/manifest.json`.

Generated dataset files are excluded from Git and can be reproduced with:

```bash
python scripts/prepare_dataset.py \
  --dataset-revision c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28
```
