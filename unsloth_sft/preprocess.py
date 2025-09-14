"""
Pretokenization and caching for Unsloth SFT datasets (JSONL → tokenized HF dataset)

Purpose
- Read JSONL with user-configurable fields (input_key/target_key)
- Build concise instruction-output pairs suitable for non-reasoning SFT (per rStar guidance)
- Apply tokenizer and DROP samples whose tokenized length exceeds max_seq_len
- Save tokenized train/eval datasets to disk to avoid repeated preprocessing on every run

Key features
- Multi-process tokenization (config.num_proc)
- Configurable eval split ratio if no explicit val file provided
- Optional wrapping of targets in <answer> tags
- Strict length filtering (no truncation) to avoid training on truncated samples

Inputs
- YAML config: configs/sft_unsloth.yaml

Outputs
- Saved dataset to `data.dataset_cache_dir` with 'train' and 'validation' splits

Notes
- This script only prepares data; training happens in train_sft.py using the cached dataset
"""

from __future__ import annotations  # enable forward annotations

import os  # standard lib for path and env
import json  # for JSONL reading
from typing import Dict, Any, Tuple, List  # typing hints

import yaml  # to parse YAML configs
from datasets import Dataset, DatasetDict, load_dataset  # HF datasets

# Try to import Unsloth tokenizer route first; fallback to transformers if necessary
try:
    from unsloth import FastLanguageModel  # Unsloth provides tokenizer via from_pretrained
    _HAS_UNSLOTH = True  # flag indicating Unsloth is available
except Exception:
    _HAS_UNSLOTH = False  # Unsloth import failed; we will fallback

from transformers import AutoTokenizer  # fallback tokenizer


def _read_yaml(path: str) -> Dict[str, Any]:
    """Load YAML file into a dictionary with friendly errors."""
    try:  # attempt to open and parse the YAML
        with open(path, "r", encoding="utf-8") as f:  # open file in utf-8
            return yaml.safe_load(f)  # parse YAML safely
    except Exception as e:  # catch and re-raise with context
        raise RuntimeError(f"Failed to read YAML config at {path}: {e}")  # meaningful error


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    try:  # attempt to make dirs
        os.makedirs(path, exist_ok=True)  # create directories if needed
    except Exception as e:  # catch errors
        raise RuntimeError(f"Failed to create directory {path}: {e}")  # detailed error


def _load_tokenizer(cfg: Dict[str, Any]):
    """Load tokenizer via Unsloth if available, otherwise via Transformers."""
    model_name = cfg["model"]["model_name_or_path"]  # get model id/path from config
    try:  # try Unsloth first for tokenizer consistency
        if _HAS_UNSLOTH:  # check if Unsloth import worked
            # Unsloth uses FastLanguageModel to load model and tokenizer
            # but tokenizer can also be created from AutoTokenizer with chat template support
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # load tokenizer
        else:  # fallback path
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # transformers tokenizer
        # Ensure eos/pad tokens are set sanely
        if tokenizer.pad_token is None:  # if no pad token
            tokenizer.pad_token = tokenizer.eos_token  # set pad token to eos for causal LM
        return tokenizer  # return configured tokenizer
    except Exception as e:  # catch load errors
        raise RuntimeError(f"Failed to load tokenizer for {model_name}: {e}")  # surface reason


def _build_text(sample: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """Format a single training text from input/target using a minimal prompt style.

    We keep SFT concise (non-reasoning). If wrap_answer_in_tags is enabled, we wrap the
    target in <answer> tags to align with later RL extraction formats.
    """
    input_key = cfg["data"]["input_key"]  # get input key name
    target_key = cfg["data"]["target_key"]  # get target key name
    sys_prompt = cfg["data"].get("system_prompt", "")  # optional system prompt text
    wrap_ans = bool(cfg["data"].get("wrap_answer_in_tags", False))  # whether to wrap answer
    non_reasoning = bool(cfg["data"].get("non_reasoning", True))  # non-reasoning mode

    user_text = str(sample.get(input_key, "")).strip()  # fetch input text
    target_text_raw = str(sample.get(target_key, "")).strip()  # fetch target text

    # If non_reasoning, heuristically prune chain-of-thought style solutions by taking final line
    # This is a light heuristic; users can provide already concise answers.
    if non_reasoning and "\n" in target_text_raw:  # if multi-line answer and non-reasoning
        last_line = target_text_raw.strip().splitlines()[-1]  # take last line often containing final answer
    else:
        last_line = target_text_raw  # otherwise keep whole target

    if wrap_ans:  # optionally wrap with tags
        final_ans = f"<answer> {last_line} </answer>"  # produce tagged answer
    else:
        final_ans = last_line  # keep as-is

    # Minimal prompt: System (optional), then User question, then Assistant answer
    # We avoid complex chat templates to keep preprocessing generic across tokenizers
    if sys_prompt:  # if system prompt exists
        text = f"<|system|>\n{sys_prompt}\n<|end|>\n<|user|>\n{user_text}\n<|end|>\n<|assistant|>\n{final_ans}\n<|end|>"  # compose text
    else:
        text = f"<|user|>\n{user_text}\n<|end|>\n<|assistant|>\n{final_ans}\n<|end|>"  # compose without system
    return text  # return composed training example


def _tokenize_and_filter(examples: Dict[str, List[Any]], tokenizer, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Tokenize and drop sequences exceeding max_seq_len.

    The input `examples` contains a batch of 'text' strings. We tokenize without truncation,
    check lengths, and keep only those within the limit. We output lists for HF Datasets mapping.
    """
    max_len = int(cfg["data"]["max_seq_len"])  # allowed maximum length
    remove_long = bool(cfg["data"].get("remove_long_samples", True))  # drop long samples

    texts: List[str] = examples["text"]  # extract batch texts
    # Perform tokenization with no truncation to measure exact length
    enc = tokenizer(texts, add_special_tokens=True, truncation=False)  # tokenize batch

    new_input_ids: List[List[int]] = []  # collected input ids
    new_attention_masks: List[List[int]] = []  # collected attention masks

    for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):  # iterate batch
        if len(ids) <= max_len:  # length check
            new_input_ids.append(ids)  # accept
            new_attention_masks.append(attn)  # accept
        else:
            if remove_long:  # if configured to drop
                # drop the sample silently from output
                continue  # skip long sample
            else:  # if not removing long, we will truncate to max_len
                new_input_ids.append(ids[:max_len])  # truncate
                new_attention_masks.append(attn[:max_len])  # truncate

    # Labels identical to input_ids for causal LM SFT
    return {  # return tokenized batch
        "input_ids": new_input_ids,  # token ids
        "attention_mask": new_attention_masks,  # attention mask
        "labels": new_input_ids,  # labels equal input for next-token prediction
    }


def main(config_path: str = "configs/sft_unsloth.yaml") -> None:
    """Entry point: build and cache tokenized dataset according to YAML config."""
    cfg = _read_yaml(config_path)  # load config dict

    # Export optional environment variables from YAML (e.g., LLM_AS_A_JUDGE_*)
    for k, v in (cfg.get("env", {}) or {}).items():  # iterate env entries
        if v is None:
            continue
        os.environ[str(k)] = str(v)  # set into process env

    # Resolve dataset parameters
    train_path = cfg["data"]["train_jsonl_path"]  # training JSONL path
    val_path = cfg["data"].get("val_jsonl_path", None)  # optional validation JSONL
    cache_dir = cfg["data"]["dataset_cache_dir"]  # where to save processed dataset
    eval_split_ratio = float(cfg["data"].get("eval_split_ratio", 0.02))  # split ratio
    num_proc = int(cfg["data"].get("num_proc", os.cpu_count() or 1))  # parallel workers

    _ensure_dir(cache_dir)  # make sure cache dir exists

    tokenizer = _load_tokenizer(cfg)  # load tokenizer

    # Load raw data from JSONL files using HF datasets for parallel mapping ease
    if val_path and os.path.exists(val_path):  # if explicit validation path given
        raw_train = load_dataset("json", data_files=train_path, split="train")  # load train
        raw_val = load_dataset("json", data_files=val_path, split="train")  # load val as train split
    else:  # otherwise split train file into train/validation by ratio
        raw_all = load_dataset("json", data_files=train_path, split="train")  # load entire dataset
        raw_train, raw_val = raw_all.train_test_split(test_size=eval_split_ratio, seed=cfg["training"]["seed"]).values()  # split

    # Map raw fields to formatted text
    def format_fn(example):  # single-sample formatting function
        return {"text": _build_text(example, cfg)}  # produce 'text' field

    train_text = raw_train.map(format_fn, remove_columns=raw_train.column_names, desc="Formatting train")  # format train
    val_text = raw_val.map(format_fn, remove_columns=raw_val.column_names, desc="Formatting validation")  # format val

    # Tokenize + filter using multi-processing
    train_tok = train_text.map(  # map over formatted train set
        lambda batch: _tokenize_and_filter(batch, tokenizer, cfg),  # tokenize and filter
        batched=True,  # operate on batches
        num_proc=num_proc,  # parallel workers
        remove_columns=["text"],  # drop text after tokenization
        desc="Tokenizing+filtering train",  # progress label
    )
    val_tok = val_text.map(  # map over formatted val set
        lambda batch: _tokenize_and_filter(batch, tokenizer, cfg),  # tokenize and filter
        batched=True,  # operate on batches
        num_proc=num_proc,  # parallel workers
        remove_columns=["text"],  # drop text after tokenization
        desc="Tokenizing+filtering validation",  # progress label
    )

    # Ensure datasets are not empty after filtering
    if len(train_tok) == 0:  # check size
        raise RuntimeError("All training samples were filtered out by max_seq_len. Consider increasing max_seq_len or adjusting data.")  # error
    if len(val_tok) == 0:  # check size
        raise RuntimeError("All validation samples were filtered out by max_seq_len. Consider increasing max_seq_len or adjusting data.")  # error

    # Save to disk for reuse
    ds = DatasetDict({"train": train_tok, "validation": val_tok})  # combine splits
    ds.save_to_disk(cache_dir)  # persist dataset for future runs
    print(f"Saved tokenized dataset to {cache_dir} | train={len(train_tok)} validation={len(val_tok)}")  # user feedback


if __name__ == "__main__":  # CLI entry
    import argparse  # parse CLI args

    parser = argparse.ArgumentParser(description="Pretokenize and cache dataset for Unsloth SFT")  # define parser
    parser.add_argument("--config", type=str, default="configs/sft_unsloth.yaml", help="Path to YAML config")  # config path arg
    args = parser.parse_args()  # parse args
    main(args.config)  # run main with provided config
