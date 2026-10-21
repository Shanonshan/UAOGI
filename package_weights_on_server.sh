#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/data0/sh_shan/MedReID}"
output="${2:-${repo_root}/uaogi_three_model_weights.tar.gz}"
staging="$(mktemp -d)"
trap 'rm -rf "${staging}"' EXIT

copy_weight() {
  local source="$1"
  local target="$2"
  local head="$3"
  test -f "${source}/lora_adapter/adapter_config.json"
  test -f "${source}/lora_adapter/adapter_model.safetensors"
  test -f "${source}/${head}"
  mkdir -p "${staging}/${target}"
  cp -R "${source}/lora_adapter" "${staging}/${target}/"
  cp "${source}/${head}" "${staging}/${target}/${head}"
}

copy_weight "${repo_root}/results/medqa_full/lora_dirichlet_question" "qwen3.5-4b" "evidence_head.pt"
copy_weight "${repo_root}/phi-4/results/qwen_protocol/dirichlet_aligned" "phi-4-mini" "prediction_head.pt"
copy_weight "${repo_root}/MedGemma/results/medgemma_four_methods/dirichlet" "medgemma-1.5-4b" "prediction_head.pt"

tar -C "${staging}" -czf "${output}" .
sha256sum "${output}" > "${output}.sha256"
printf 'Created %s\n' "${output}"
printf 'Created %s.sha256\n' "${output}"
