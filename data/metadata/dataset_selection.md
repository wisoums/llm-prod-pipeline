# Dataset Selection — Issue #1

## Selected dataset

- Hugging Face dataset: `benjaminmacklin/IT_Support_V2`
- Domain: English IT helpdesk / technical support
- Size: 103,226 conversations
- Format: conversational JSON with a `messages` field containing user/assistant turns
- License: MIT (per the Hugging Face dataset card)
- Intended task: supervised fine-tuning for technical-support response generation

## Why this dataset

This dataset matches the project domain closely: IT support and administration rather than generic customer-service chat. It covers technical troubleshooting topics such as Windows issues, networking, drivers, SQL Server administration, and hardware debugging.

It is also large enough for meaningful QLoRA experiments while remaining manageable for a focused project.

## Target model behavior

The fine-tuned model should:

- answer technical-support questions clearly and directly;
- provide useful step-by-step troubleshooting when appropriate;
- remain within the IT/helpdesk domain;
- avoid fabricating certainty when the information is insufficient;
- refuse clearly unsafe requests in the separate safety evaluation.

## Preliminary risks to audit before training

The dataset must be sampled and checked for:

- factual correctness and unsafe technical advice;
- duplicated or near-duplicated conversations;
- inconsistent response style or quality;
- overly long or malformed conversations;
- unsupported chain-of-thought or hidden-reasoning style content;
- poor coverage balance across common technical-support topics;
- data leakage risks when train/validation/test splits are created;
- any content or metadata that should not be redistributed.

## Readiness decision

Status: **selected, pending quality audit**.

The dataset should not be treated as training-ready until the local audit confirms schema validity, quality, and acceptable duplication/coverage.
