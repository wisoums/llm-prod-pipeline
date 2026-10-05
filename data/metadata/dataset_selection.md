# Dataset Selection — Issue #1

## Selected dataset

- Dataset: `ronantakizawa/github-codereview`
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

- No missing `before_code` or `reviewer_comment`
- No obvious vague comments detected by simple pattern checks
- 25,104 repeated comment texts
- 1,331 potentially external-context-dependent comments
- Median code length: 1,876 characters
- P95: 2,851
- P99: 18,104
- Maximum: 389,803

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

The Hugging Face dataset currently shows license `other`.

The dataset card states that source repositories were selected from permissively licensed projects such as MIT, Apache-2.0, and BSD.

The raw dataset will not be redistributed through this repository. Licensing and provenance will be documented again before publishing the final model to Hugging Face.

## Decision

**Accepted for the project, pending cleaning in Issue #2.**