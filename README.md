# LLM Production Fine-Tuning & Serving Pipeline

End-to-end pipeline for fine-tuning an open LLM with QLoRA, evaluating it, and serving it with production-style latency and throughput measurement.

> **Status:** project scaffolded; implementation is tracked in GitHub Issues.

## Scope

This project intentionally stays narrow:

- Fine-tune **Qwen2.5-7B-Instruct** with **QLoRA** on a focused **technical customer-support** dataset.
- Track training runs with **MLflow**.
- Evaluate task quality plus basic safety/refusal behavior.
- Serve the fine-tuned model with **vLLM** behind a **FastAPI** endpoint.
- Package the service with **Docker**.
- Use CI to re-run evaluation checks when model/data/evaluation code changes.
- Report training cost, p50/p95 latency, throughput, token metrics, and held-out evaluation scores.

No RAG, agents, frontend, Kubernetes, distributed training, or custom model architecture.

## Pipeline

```text
Dataset -> validation/splits -> QLoRA fine-tuning -> MLflow
                                      |
                                      v
                         held-out + safety evaluation
                                      |
                                      v
                           vLLM -> FastAPI -> Docker
                                      |
                                      v
                    latency / throughput / token metrics
```

## Repository layout

```text
configs/          Training, evaluation, and serving configuration
src/
  data/           Dataset loading and validation
  training/       QLoRA training
  evaluation/     Task and safety evaluation
  serving/        FastAPI + vLLM integration
  benchmarking/   Latency and throughput measurement
tests/            Unit and smoke tests
docs/             Final results and "what broke" write-up
.github/workflows/ CI
```

## Reproducibility

The final training run will be reproducible from a versioned config and fixed data split. MLflow will record hyperparameters, metrics, run metadata, and artifacts needed to compare experiments.

## Target results

| Metric | Result |
|---|---:|
| Held-out task score | TBD |
| Safety/refusal score | TBD |
| Training cost | TBD |
| p50 latency | TBD |
| p95 latency | TBD |
| Throughput | TBD |
| Output tokens/sec | TBD |

Numbers will only be added after measured runs.

## Local setup

```bash
git clone https://github.com/wisoums/llm-prod-pipeline.git
cd llm-prod-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

GPU-specific training and serving dependencies are installed separately through the optional `training` and `serving` dependency groups.

## Public-repository safety

Never commit API keys, Hugging Face tokens, MLflow credentials, model weights, generated checkpoints, or restricted/raw datasets. See [SECURITY.md](SECURITY.md).

## License

Project source code is licensed under the [Apache License 2.0](LICENSE). Model and dataset licenses remain governed by their respective upstream licenses.
