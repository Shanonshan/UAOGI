#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --model "${MODEL:-Qwen/Qwen3.5-4B}" --adapter weights/qwen3.5-4b/lora_adapter --head weights/qwen3.5-4b/evidence_head.pt --prompt-style qwen --data "${1:?test JSONL required}" --output "${2:-outputs/qwen3.5-4b}"
