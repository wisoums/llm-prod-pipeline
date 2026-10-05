# Dataset Selection — Issue #1

## Selected dataset

- Dataset: `ronantakizawa/github-codereview`
- Revision: `c3e3c6e7e9f61e3e7a5b52894bcd440d586ae6ca`
- Domain: Python code review
- Task: given a Python diff, generate a concise actionable review comment or return `No issues found.`

## Python subset

- Train: 82,288
- Validation: 2,639
- Test: 2,609
- Total: 87,536
- Positive: 70,903
- Negative: 16,633 (19%)

## Why it fits

The dataset contains real code-review interactions with fields such as:

- `diff_context`
- `reviewer_comment`
- `quality_score`
- `comment_type`
- `is_negative`

It includes both actionable review comments and negative examples, which allows the model to learn when not to comment.

## Audit findings

- 341 examples with empty `diff_context`; no missing `reviewer_comment`
- No obvious vague comments detected by simple pattern checks
- 25,104 repeated comment texts
- 1,331 potentially external-context-dependent comments
- `diff_context` length (characters):
  - Median: 800
  - P90: 2,852
  - P95: 4,483
  - P99: 8,871
  - Maximum: 35,054

Manual inspection of 100 examples showed generally useful, human-like review comments, but also some context-dependent, weak, duplicated, and excessively long examples.

## Cleaning risks

Issue #2 should address:

- Python-only filtering
- malformed or empty diffs
- duplicate / near-duplicate examples
- external-context-dependent comments
- weak review comments
- excessive input length
- quality-score filtering only if supported by further analysis

## Licensing

The Hugging Face dataset currently declares its license as `other`.

The dataset card states that source repositories were selected from permissively licensed projects such as MIT, Apache-2.0, and BSD. However, the compiled dataset does not currently expose one uniform permissive license covering every row.

For this project:

- the raw dataset will not be redistributed;
- dataset provenance and licensing will remain documented;
- the dataset may be used for local experimentation and model development;
- redistribution/publication of the final adapter or model will require a separate licensing review before release.

## Decision

**Accepted for experimentation and development, with publication conditional on a final licensing review.**
