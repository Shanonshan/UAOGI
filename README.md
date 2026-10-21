# UAOGI Medical LLM Inference

This repository contains inference and evaluation code for the released UAOGI
LoRA adapters on three medical question-answering backbones:

- Qwen3.5-4B
- Phi-4-mini-instruct
- MedGemma-1.5-4B

Training code, base-model checkpoints, and restricted benchmark data are not
included. Obtain each base model under its original license and place the
released LoRA adapter and prediction head under `weights/`.

## Released weights

Download the three-model UAOGI LoRA package from the
[v1.0.0 GitHub Release](https://github.com/Shanonshan/UAOGI/releases/tag/v1.0.0).

```text
File: uaogi_three_model_weights.tar.gz
SHA-256: fe3489d7fbd62b8be1a54c2bfb4904dd16b4c131b33201c16564f28467eb19d8
```

Extract it directly into the repository:

```bash
tar -xzf uaogi_three_model_weights.tar.gz -C weights
```

## Installation

```bash
conda create -n uaogi-inference python=3.10 -y
conda activate uaogi-inference
pip install -r requirements.txt
```

MedGemma is a gated model and requires acceptance of Google's terms and an
authenticated Hugging Face account. Never commit access tokens to this repo.

## Data format

The evaluator accepts JSONL with a question, four options, and a gold answer:

```json
{"id":"example-1","question":"...","options":["...","...","...","..."],"answer":"A"}
```

## Evaluation

```bash
bash run_qwen35.sh data/medqa_test.jsonl
bash run_phi4.sh data/medqa_test.jsonl
bash run_medgemma.sh data/medqa_test.jsonl
```

Set `MODEL=/local/checkpoint/path` to use an existing local base checkpoint.
Each run writes `metrics.json` and `predictions.jsonl` under `outputs/`.

Reported metrics are accuracy, ECE, NLL, Brier score, and confidence-ranked
AURC. The released head produces non-negative evidence using Softplus and maps
it to Dirichlet parameters as `alpha = evidence + 1`.

## Maintainer: package released weights

On the experiment server, upload this repository and run:

```bash
bash package_weights_on_server.sh /data0/sh_shan/MedReID \
  /data0/sh_shan/MedReID/uaogi_three_model_weights.tar.gz
```

Extract the archive into this repository's `weights/` directory. Base-model
weights are intentionally excluded.
