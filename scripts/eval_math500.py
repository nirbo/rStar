"""
eval_math500.py

Context and purpose:
- Offline exact-match evaluation on the MATH-500 dataset for a causal LM.
- Supports two modes:
  1) Merged model directory (standalone HF model), or
  2) Base model with a PEFT LoRA adapter.

Functions:
- load_model_tokenizer: Loads model/tokenizer with optional PEFT adapter.
- normalize_math_and_score: Uses rStar/VERL math normalization to compute EM.
- run_eval: Main evaluation loop over datasets/MATH-500/test.jsonl, writes CSV.
- main: CLI to configure model path, adapter path, batch size, and limits.

Related files and dependencies:
- Depends on transformers and (optionally) peft. Imports VERL math normalizer.
- Dataset: datasets/MATH-500/test.jsonl (JSONL with keys: problem, answer, subject, level, unique_id).

Integration points:
- Can be used after SFT to quickly quantify accuracy on MATH-500.
- Complements unsloth_sft/merge_lora.py: evaluate either merged weights or adapter.
"""

from __future__ import annotations  # future annotations

import argparse  # CLI parsing
import csv  # write per-sample results
import json  # read JSONL
from pathlib import Path  # path utilities
from typing import Iterable, List, Optional, Tuple  # typing

import torch  # dtype and devices
from transformers import AutoModelForCausalLM, AutoTokenizer  # HF model/tokenizer


def _read_jsonl(path: Path) -> List[dict]:  # read JSONL file to list of dicts
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[eval] Skipping malformed JSONL line: {e}")
    return rows


def load_model_tokenizer(  # helper to load model/tokenizer
    model_path: str,  # path or repo id to model (merged or base)
    adapter_dir: Optional[str] = None,  # optional PEFT adapter dir
    dtype: str = "bfloat16",  # torch dtype name
    device: str = "auto",  # device map strategy
):
    # Map dtype string to torch dtype safely
    torch_dtype = getattr(torch, dtype, torch.bfloat16)
    device_map = "auto" if device == "auto" else ("auto" if device == "cuda" and torch.cuda.is_available() else None)

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)  # load tokenizer
    if tok.pad_token is None:  # ensure pad defined
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(  # load base or merged model
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if adapter_dir:  # if adapter path provided, wrap with PEFT
        try:
            from peft import PeftModel  # lazy import

            model = PeftModel.from_pretrained(model, adapter_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load PEFT adapter from {adapter_dir}: {e}")

    model.eval()  # set eval mode
    return model, tok


def normalize_math_and_score(  # compute EM score via VERL normalization
    pred: str,  # model output string
    gt: str,  # ground-truth answer string
) -> int:
    try:
        from verl.utils.reward_score.math import compute_score  # import normalizer

        return int(compute_score(pred, gt))  # returns 0.0 or 1.0
    except Exception as e:
        print(f"[eval] Normalization failed: {e}")
        return 0


def _build_prompt(problem: str) -> str:  # build a concise instruction prompt
    return (
        "Solve the problem. Provide only the final answer in \\boxed{...}.\n\n"
        + problem.strip()
        + "\n\nAnswer:"
    )


@torch.no_grad()  # disable gradients for inference
def run_eval(  # main evaluation loop
    model_path: str,  # merged model path (or base if adapter_dir provided)
    adapter_dir: Optional[str],  # optional adapter directory
    limit: Optional[int],  # optionally limit number of samples
    batch_size: int,  # batch size for tokenization/decoding
    max_new_tokens: int,  # maximum generation tokens
    temperature: float = 0.0,  # greedy by default
    top_p: float = 1.0,  # sampling params (unused when temperature=0)
    dtype: str = "bfloat16",  # dtype for loading
    device: str = "auto",  # device mapping strategy
    out_csv: Optional[str] = None,  # optional CSV output path
) -> Tuple[float, int]:
    # Load model/tokenizer once
    model, tok = load_model_tokenizer(model_path, adapter_dir=adapter_dir, dtype=dtype, device=device)

    # Read dataset
    ds_path = Path("datasets/MATH-500/test.jsonl")  # static path in repo
    rows = _read_jsonl(ds_path)
    if limit is not None:
        rows = rows[: int(limit)]

    # Prepare output writer if requested
    writer = None
    if out_csv:
        outp = Path(out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(outp.open("w", newline="", encoding="utf-8"))
        writer.writerow(["idx", "unique_id", "subject", "level", "score", "gt", "response"])  # header

    correct = 0
    total = len(rows)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    # Evaluate sequentially in micro-batches (safe and simple)
    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        prompts = [_build_prompt(r.get("problem", "")) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        out = model.generate(**enc, **gen_kwargs)

        for j, r in enumerate(batch):
            prompt_len = enc["input_ids"][j].shape[0]
            resp_ids = out[j][prompt_len:]
            text = tok.decode(resp_ids, skip_special_tokens=True)
            score = normalize_math_and_score(text, r.get("answer", ""))
            correct += int(score)
            if writer:
                writer.writerow([
                    i + j,
                    r.get("unique_id", ""),
                    r.get("subject", ""),
                    r.get("level", ""),
                    score,
                    r.get("answer", ""),
                    text,
                ])

    acc = correct / total if total else 0.0
    print({"samples": total, "correct": correct, "accuracy": acc})
    return acc, total


def main() -> None:  # CLI entry
    p = argparse.ArgumentParser(description="Evaluate a (merged or adapter) model on MATH-500")
    p.add_argument("--model", required=True, help="Merged HF model path (or base when --adapter is set)")
    p.add_argument("--adapter", default=None, help="Optional PEFT adapter dir (if evaluating un-merged)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples (default: all)")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (tokenization/decoding)")
    p.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0, help="Temperature (0=greedy)")
    p.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling (when temperature>0)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Load dtype")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device mapping strategy")
    p.add_argument("--out_csv", default="outputs/eval/math500_eval.csv", help="Where to write per-sample results")
    args = p.parse_args()

    run_eval(
        model_path=args.model,
        adapter_dir=args.adapter,
        limit=args.limit,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        device=args.device,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":  # CLI guard
    main()  # execute

