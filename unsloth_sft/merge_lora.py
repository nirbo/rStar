"""
merge_lora.py

Context and purpose:
- Provides a utility to merge a trained LoRA/QLoRA adapter (PEFT) back into
  the base model weights, producing a standalone Hugging Face model directory
  suitable for serving in vLLM or HF transformers without adapter loading.

Functions:
- merge_lora_adapter: Programmatic API to perform the merge and save.
- main: CLI entry to run the merge from the command line.

Related files and dependencies:
- Depends on transformers (AutoModelForCausalLM, AutoTokenizer, GenerationConfig)
  and peft (PeftModel). Uses torch for dtypes and device control.
- Invoked optionally by unsloth_sft/train_sft.py after SFT when
  training.merge_lora_adapter=true in configs/sft_unsloth.yaml.

Integration points:
- Training pipeline: post-training, when the best adapter is already saved
  to the training.output_dir, the trainer may call merge_lora_adapter() to
  write merged weights to training.merge_output_dir.
"""

from __future__ import annotations  # enable forward type hints  

import argparse  # CLI parsing
import os  # filesystem operations
from pathlib import Path  # path utils
from typing import Optional  # type hints

import torch  # tensor and dtypes
from transformers import (  # HF model/tokenizer I/O
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel  # PEFT wrapper to load LoRA adapters


def merge_lora_adapter(  # define the core merge function
    base_model_path: str,  # HF model dir or repo id for the base
    adapter_dir: str,  # directory containing adapter_model.safetensors and adapter_config.json
    output_dir: str,  # destination directory for merged weights
    dtype: str = "bfloat16",  # desired torch dtype for model load/merge
    device: str = "auto",  # device hint: "auto", "cuda", or "cpu"
    safe_serialization: bool = True,  # save safetensors if possible
) -> str:
    """Merge a PEFT LoRA adapter into base weights and save a standalone model.

    Inputs:
    - base_model_path: The base model directory or Hub id.
    - adapter_dir: The directory containing the trained adapter files.
    - output_dir: Where to save the merged model (created if absent).
    - dtype: One of {"bfloat16", "float16", "float32"} typically.
    - device: "auto" (prefer GPU if available), "cuda", or "cpu".
    - safe_serialization: If True, save safetensors instead of bin.

    Output:
    - Returns the output_dir path for convenience.
    """
    # Resolve dtype string to torch dtype safely
    try:  # attempt to get torch dtype attribute by name
        torch_dtype = getattr(torch, dtype)
    except AttributeError:  # fallback to float16 if provided dtype unknown
        torch_dtype = torch.float16

    # Decide device_map according to requested device hint
    if device == "auto":  # let transformers choose available device(s)
        device_map = "auto"
    elif device == "cuda":  # force CUDA if available
        device_map = "auto" if torch.cuda.is_available() else None
    else:  # default to CPU
        device_map = None

    # Ensure output directory exists
    out = Path(output_dir)  # create Path object for output_dir
    out.mkdir(parents=True, exist_ok=True)  # create parents as needed

    # Load base model in selected dtype; avoid 4/8-bit quantization for accurate merge
    base_kwargs = {  # kwargs for HF load
        "dtype": torch_dtype,  # use new `dtype` per transformers deprecation notice
        "low_cpu_mem_usage": True,
        "device_map": device_map,
    }
    # Load base model weights
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **base_kwargs)  # type: ignore

    # Bind adapter on top of the base model
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)  # wrap with trained adapter

    # Merge adapter weights into the base model and free adapter modules
    merged = peft_model.merge_and_unload()  # in-place merge, returns underlying base

    # Save tokenizer alongside the model for completeness
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)  # load tokenizer
    tok.save_pretrained(out.as_posix())  # save tokenizer files

    # Save generation config if present in base
    try:  # attempt to load generation config from base
        gen_cfg = GenerationConfig.from_pretrained(base_model_path)  # may raise if missing
        gen_cfg.save_pretrained(out.as_posix())  # write generation_config.json
    except Exception:  # silently continue if not available
        pass  # no generation config to save

    # Persist merged model weights to output_dir
    merged.save_pretrained(out.as_posix(), safe_serialization=safe_serialization)  # write model weights

    return out.as_posix()  # return the saved directory path


def main() -> None:  # CLI entry point
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into base model weights")  # build parser
    parser.add_argument("--base_model_path", required=True, help="Path or repo id of the base model")  # base path arg
    parser.add_argument("--adapter_dir", required=True, help="Directory containing the trained adapter")  # adapter arg
    parser.add_argument(
        "--output_dir", required=True, help="Where to save the merged model (new or empty directory)"
    )  # output dir arg
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype to use when loading for merge",
    )  # dtype choice
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device mapping hint for model load",
    )  # device choice
    args = parser.parse_args()  # parse arguments

    # Execute merge with provided args
    merge_lora_adapter(
        base_model_path=args.base_model_path,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        device=args.device,
    )  # perform the merge


if __name__ == "__main__":  # CLI guard
    main()  # run CLI
