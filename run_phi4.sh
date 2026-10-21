#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --model "${MODEL:-microsoft/Phi-4-mini-instruct}" --adapter weights/phi-4-mini/lora_adapter --head weights/phi-4-mini/prediction_head.pt --prompt-style matched --data "${1:?test JSONL required}" --output "${2:-outputs/phi-4-mini}"
