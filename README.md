<div align="center">

# ReviewPy

**Human-style Python code review from your terminal.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--7B-7C3AED)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
[![Fine-tuning](https://img.shields.io/badge/Fine--tuning-QLoRA-F59E0B)](#)
[![Serving](https://img.shields.io/badge/Serving-vLLM-2563EB)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-Apache%202.0-0EA5E9)](LICENSE)

</div>

---

ReviewPy is a CLI for reviewing GitHub pull requests with a model fine-tuned on real human code-review feedback.

Paste a PR URL and ReviewPy analyzes the changed Python lines, identifies meaningful issues, and returns concise reviewer-style comments tied to the relevant diff.

```bash
reviewpy https://github.com/org/repo/pull/123
```

```text
src/config.py:48

Using `x or default` here will replace valid falsy values such as `0`.
Consider checking `x is None` explicitly.
```

The goal is simple: useful comments when something matters, and `No issues found.` when it does not.

## How it works

```text
GitHub PR URL
     │
     ▼
GitHub API
     │
     ▼
Python diff hunks
     │
     ▼
Input preparation
     │
     ▼
Qwen2.5-7B-Instruct
+ QLoRA PEFT adapter
     │
     ▼
Review comments
(file + line + text)
     │
     ▼
CLI output
```

## Model pipeline

- **Dataset:** `ronantakizawa/github-codereview`
- **Domain:** Python code review
- **Fine-tuning:** QLoRA-based parameter-efficient fine-tuning (PEFT)
- **Experiment tracking:** MLflow
- **Evaluation:** held-out review benchmark + basic safety checks
- **Serving:** vLLM + FastAPI
- **Packaging:** Docker
- **Metrics:** review quality, training cost, p50/p95 latency, throughput

## Planned CLI

```bash
reviewpy <PR_URL>
reviewpy <PR_URL> --json
```

Publishing comments directly to GitHub will be considered later and will require explicit confirmation.

## Project status

- [x] Dataset selected and audited
- [ ] Reproducible cleaning and preprocessing
- [ ] Fixed evaluation split
- [ ] Baseline evaluation
- [ ] QLoRA PEFT training
- [ ] Base vs adapted model comparison
- [ ] CLI PR ingestion and line mapping
- [ ] vLLM + FastAPI serving
- [ ] Docker + CI evaluation
- [ ] Hugging Face release

## Repository layout

```text
configs/          Training, evaluation, and serving configuration
scripts/          Dataset inspection and preprocessing utilities
src/
  data/           Dataset loading and validation
  training/       QLoRA PEFT training
  evaluation/     Review and safety evaluation
  serving/        FastAPI + vLLM integration
  benchmarking/   Latency and throughput measurement
tests/            Unit and smoke tests
docs/             Results and lessons learned
.github/workflows/ CI
```

## Target results

| Metric | Result |
|---|---:|
| Held-out review score | TBD |
| No-comment decision quality | TBD |
| Training cost | TBD |
| p50 latency | TBD |
| p95 latency | TBD |
| Throughput | TBD |
| Output tokens/sec | TBD |

Results will only be added after measured runs.

## Local setup

```bash
git clone https://github.com/wisoums/reviewpy.git
cd reviewpy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

GPU-specific training and serving dependencies are installed separately through the optional training and serving dependency groups.

## Scope

ReviewPy v1 is intentionally focused on **Python pull-request review**.

Out of scope for v1: RAG, agents, frontend, Kubernetes, distributed training, custom model architectures, and multi-language review.

## License

Project source code is licensed under the [Apache License 2.0](LICENSE). Model and dataset licenses remain governed by their respective upstream licenses.
