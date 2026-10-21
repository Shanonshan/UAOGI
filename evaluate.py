#!/usr/bin/env python3
"""Evaluate released UAOGI LoRA adapters on four-choice medical QA JSONL."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ("A", "B", "C", "D")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(row: dict) -> dict:
    options = row.get("options", row.get("choices"))
    if options is None and all(key in row for key in ("opa", "opb", "opc", "opd")):
        options = [row[key] for key in ("opa", "opb", "opc", "opd")]
    if isinstance(options, dict):
        options = [options.get(letter, "") for letter in LETTERS]
    options = list(options or [])[:4]
    if len(options) != 4:
        raise ValueError("Each example must contain exactly four answer options.")
    answer = row.get("answer_idx", row.get("answer", row.get("label", row.get("cop"))))
    if isinstance(answer, int):
        index = answer
    elif str(answer).upper() in LETTERS:
        index = LETTERS.index(str(answer).upper())
    else:
        index = next((i for i, option in enumerate(options) if str(option).strip() == str(answer).strip()), -1)
    if index not in range(4):
        raise ValueError(f"Could not normalize answer: {answer!r}")
    return {"id": row.get("id", row.get("question_id", "")), "question": row.get("question", row.get("stem", "")), "options": options, "answer": index}


def format_prompt(row: dict, style: str) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, row["options"]))
    body = f"{row['question']}\n{options}\nAnswer:"
    if style == "matched":
        return "Answer the medical multiple-choice question. Reply with only one letter (A, B, C, or D).\n\n" + body
    return body


class QADataset(Dataset):
    def __init__(self, rows: list[dict], prompt_style: str):
        self.rows = [normalize(row) for row in rows]
        self.prompt_style = prompt_style

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return format_prompt(row, self.prompt_style), row["answer"], row["id"]


def hidden_size(model) -> int:
    value = getattr(model.config, "hidden_size", None)
    text_config = getattr(model.config, "text_config", None)
    return int(value or getattr(text_config, "hidden_size", 0) or 4096)


def pooled_hidden(model, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
    states = output.hidden_states[-1]
    indices = encoded["attention_mask"].sum(dim=1) - 1
    return states[torch.arange(states.shape[0], device=states.device), indices].float()


def load_head(path: Path, dimension: int, device: torch.device) -> nn.Linear:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    head = nn.Linear(dimension, 4)
    head.load_state_dict(state)
    return head.to(device).eval()


def metrics(rows: list[dict]) -> dict:
    count = len(rows)
    accuracy = sum(row["pred"] == row["gold"] for row in rows) / count
    nll = sum(-math.log(max(row["probs"][row["gold"]], 1e-12)) for row in rows) / count
    brier = sum(sum((prob - float(i == row["gold"])) ** 2 for i, prob in enumerate(row["probs"])) for row in rows) / count
    ece = 0.0
    for index in range(10):
        bucket = [row for row in rows if index / 10 <= row["conf"] < (index + 1) / 10 or (index == 9 and row["conf"] <= 1)]
        if bucket:
            ece += len(bucket) / count * abs(sum(row["pred"] == row["gold"] for row in bucket) / len(bucket) - sum(row["conf"] for row in bucket) / len(bucket))
    ranked = sorted(rows, key=lambda row: row["conf"], reverse=True)
    errors, aurc = 0, 0.0
    for index, row in enumerate(ranked, 1):
        errors += row["pred"] != row["gold"]
        aurc += errors / index / count
    return {"n": count, "accuracy": accuracy, "ece": ece, "nll": nll, "brier": brier, "aurc": aurc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base-model ID or local checkpoint")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-style", choices=("qwen", "matched"), default="matched")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    backbone = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype="auto")
    backbone = PeftModel.from_pretrained(backbone, args.adapter).to(device).eval()
    backbone.config.use_cache = False
    head = load_head(args.head, hidden_size(backbone), device)

    source_rows = load_jsonl(args.data)
    if args.limit:
        source_rows = source_rows[:args.limit]
    dataset = QADataset(source_rows, args.prompt_style)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: batch)
    predictions = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            prompts, labels, identifiers = zip(*batch)
            encoded = tokenizer(list(prompts), padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            evidence = torch.nn.functional.softplus(head(pooled_hidden(backbone, encoded)))
            alpha = evidence + 1.0
            probabilities = alpha / alpha.sum(dim=1, keepdim=True)
            uncertainties = 4.0 / alpha.sum(dim=1)
            for identifier, gold, probability, uncertainty_value in zip(identifiers, labels, probabilities.cpu().tolist(), uncertainties.cpu().tolist()):
                prediction = max(range(4), key=lambda index: probability[index])
                predictions.append({"id": identifier, "gold": gold, "pred": prediction, "probs": probability, "conf": probability[prediction], "uncertainty": uncertainty_value})
            print(f"[eval] {min(batch_index * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    result = metrics(predictions)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
