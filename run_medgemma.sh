#!/usr/bin/env bash
set -euo pipefail
python evaluate.py --model "${MODEL:-google/medgemma-1.5-4b-it}" --adapter weights/medgemma-1.5-4b/lora_adapter --head weights/medgemma-1.5-4b/prediction_head.pt --prompt-style matched --data "${1:?test JSONL required}" --output "${2:-outputs/medgemma-1.5-4b}"
