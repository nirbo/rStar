"""
Unsloth SFT Trainer for Gemma 3 models (single GPU, 32GB-friendly)

Features implemented per user request:
- Full Unsloth VRAM/performance optimizations (4-bit QLoRA; gradient checkpointing; FlashAttention if available)
- Checkpoint save every N steps, limit to N saved checkpoints, save best model by lowest eval_loss
- Evaluate every N steps
- Dataset pretokenization + caching on disk (see preprocess.py)
- Multi-worker preprocessing controlled via YAML
- Strict filtering: exclude samples exceeding max_seq_len to avoid truncation
- Config-driven via YAML with user-exposed max_seq_len and other knobs
- Optional RSLoRA if supported by Unsloth (falls back gracefully if unavailable)

Usage:
  python unsloth_sft/train_sft.py --config configs/sft_unsloth.yaml
"""

from __future__ import annotations  # forward annotations

import unsloth # must be the first import always
import os  # stdlib for env and paths
from typing import Dict, Any  # typing hints

import yaml  # read config yaml
import torch  # torch core
from datasets import load_from_disk  # load cached dataset

# Prefer Unsloth; fallback to transformers+peft if unavailable
try:  # try to import Unsloth utilities
    from unsloth import FastLanguageModel  # for efficient model loading
    _HAS_UNSLOTH = True  # mark success
except Exception:
    _HAS_UNSLOTH = False  # mark fallback

from transformers import (
    AutoTokenizer,  # tokenizer
    TrainingArguments,  # HF trainer args
    Trainer,  # trainer
    DataCollatorForLanguageModeling,  # data collator for causal LM
    AutoModelForCausalLM,  # fallback model loader
)
from transformers.trainer_utils import get_last_checkpoint
from typing import Optional
from transformers.utils import logging as hf_logging

_HAS_RICH = False  # disable custom rich logging; use default Trainer/Unsloth logs

try:  # LoRA via PEFT
    from peft import LoraConfig, get_peft_model  # peft helpers
    _HAS_PEFT = True  # flag for availability
except Exception:
    _HAS_PEFT = False  # no peft available


def _read_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config with robust error messaging."""
    try:  # try reading file
        with open(path, "r", encoding="utf-8") as f:  # open file
            return yaml.safe_load(f)  # parse YAML
    except Exception as e:  # catch failures
        raise RuntimeError(f"Failed to read YAML config at {path}: {e}")  # report issue


def _as_int(val: Any, key_path: str) -> int:
    """Parse an integer from config with helpful errors.

    Accepts int, numeric str (e.g., "8" or "8.0"), or float that is integral.
    Raises with clear guidance on failure.
    """
    if isinstance(val, bool):
        raise RuntimeError(f"Config {key_path} must be an integer, not boolean: {val}")
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if float(val).is_integer():
            return int(val)
        raise RuntimeError(f"Config {key_path} must be an integer, got non-integer float: {val}")
    if isinstance(val, str):
        s = val.strip()
        try:
            f = float(s)
            if f.is_integer():
                return int(f)
        except Exception:
            pass
        raise RuntimeError(
            f"Config {key_path} must be an integer. Got '{val}'. Please set an integer value (e.g., 6)."
        )
    raise RuntimeError(f"Config {key_path} must be an integer. Got unsupported type: {type(val)}")


def _as_float(val: Any, key_path: str) -> float:
    """Parse a float from config with helpful errors (accepts int/float/str)."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        try:
            return float(s)
        except Exception:
            raise RuntimeError(
                f"Config {key_path} must be a number. Got '{val}'. Please set a decimal like 0.0002."
            )
    raise RuntimeError(f"Config {key_path} must be numeric. Got unsupported type: {type(val)}")


def _set_torch_env(cfg: Dict[str, Any]) -> None:
    """Configure torch backend per config for best perf on modern GPUs."""
    training = cfg["training"]  # training section
    # Enable TF32 if requested (improves throughput on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("tf32", True))  # enable/disable TF32
    torch.backends.cudnn.allow_tf32 = bool(training.get("tf32", True))  # enable/disable TF32 in cuDNN


def _load_model_and_tokenizer(cfg: Dict[str, Any]):
    """Load model+tokenizer with Unsloth if available, else fallback to Transformers+PEFT."""
    model_cfg = cfg["model"]  # model section
    name = model_cfg["model_name_or_path"]  # model id/name
    dtype = getattr(torch, str(model_cfg.get("dtype", "bfloat16")))  # torch dtype
    # Normalize device_map to Accelerate-supported types
    dm_cfg = model_cfg.get("device_map", 0)
    if isinstance(dm_cfg, str):
        if dm_cfg.lower() in ("cuda:0", "cuda", "gpu", "0", "single", "auto"):
            device_map = 0
        else:
            device_map = dm_cfg
    else:
        device_map = dm_cfg

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)  # load tokenizer
    if tokenizer.pad_token is None:  # ensure pad token exists
        tokenizer.pad_token = tokenizer.eos_token  # set pad to eos

    if _HAS_UNSLOTH:  # preferred path
        # Unsloth: efficient 4-bit loading and adapter application
        # Build a conservative max_memory map to avoid unintended CPU/disk offload by HF quantizer
        max_memory = None
        try:
            if torch.cuda.is_available():
                total_gb = int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
                allow_gb = max(total_gb - 1, 1)
                max_memory = {0: f"{allow_gb}GiB"}  # Accelerate expects int device ids
        except Exception:
            max_memory = None
        model, tokenizer = FastLanguageModel.from_pretrained(  # type: ignore
            model_name=name,  # model name or path
            max_seq_length=int(cfg["data"]["max_seq_len"]),  # max seq len
            dtype=dtype,  # compute dtype
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),  # 4-bit quant
            device_map=device_map,  # ensure no CPU/disk offload
            max_memory=max_memory,  # hint available memory to quantizer
            low_cpu_mem_usage=True,
        )
        # Apply LoRA/RSLoRA if requested
        if bool(model_cfg.get("qlora", True)):  # if QLoRA adapters requested
            try:  # try Unsloth-native peft attach
                model = FastLanguageModel.get_peft_model(  # type: ignore
                    model,  # base model
                    r=int(model_cfg.get("lora_r", 64)),  # rank
                    target_modules=list(model_cfg.get("target_modules", [])),  # target modules
                    lora_alpha=int(model_cfg.get("lora_alpha", 16)),  # alpha
                    lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),  # dropout
                    bias="none",  # no bias
                    use_rslora=bool(model_cfg.get("rslora", False)),  # optionally RSLoRA
                )
            except Exception as e:  # if Unsloth peft attach fails
                if not _HAS_PEFT:  # peft not available
                    raise RuntimeError(f"Failed to attach LoRA with Unsloth and PEFT is missing: {e}")  # fatal
                # Fallback to PEFT generic method
                peft_cfg = LoraConfig(  # define LoRA config
                    r=int(model_cfg.get("lora_r", 64)),  # rank
                    lora_alpha=int(model_cfg.get("lora_alpha", 16)),  # alpha
                    lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),  # dropout
                    bias="none",  # no bias
                    task_type="CAUSAL_LM",  # task type
                    target_modules=list(model_cfg.get("target_modules", [])),  # target modules
                )
                model = get_peft_model(model, peft_cfg)  # attach adapters via PEFT
    else:  # fallback: Transformers + PEFT/BNB
        # Configure bitsandbytes 4-bit via kwargs
        bnb_kwargs = dict(  # gather BNB kwargs
            load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),  # 4-bit
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),  # nf4
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),  # double quant
            bnb_4bit_compute_dtype=dtype,  # compute dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **bnb_kwargs,
        )  # load model
        # Apply LoRA if configured and PEFT is available
        if bool(model_cfg.get("qlora", True)) and _HAS_PEFT:  # if LoRA desired
            peft_cfg = LoraConfig(  # LoRA config
                r=int(model_cfg.get("lora_r", 64)),  # rank
                lora_alpha=int(model_cfg.get("lora_alpha", 16)),  # alpha
                lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),  # dropout
                bias="none",  # bias
                task_type="CAUSAL_LM",  # task
                target_modules=list(model_cfg.get("target_modules", [])),  # target modules
            )
            model = get_peft_model(model, peft_cfg)  # attach adapters
        elif bool(model_cfg.get("qlora", True)) and not _HAS_PEFT:  # requested but missing peft
            raise RuntimeError("PEFT is not installed, but qlora=true. Install peft or set qlora=false.")  # error

    # Gradient checkpointing as requested
    if bool(model_cfg.get("gradient_checkpointing", True)):  # if enabled
        model.gradient_checkpointing_enable()  # enable

    return model, tokenizer  # return loaded model and tokenizer


# Custom Rich logger removed; default logs will be used


def main(config_path: str = "configs/sft_unsloth.yaml") -> None:
    """Train SFT with cached dataset and Unsloth optimizations."""
    cfg = _read_yaml(config_path)  # load YAML config
    # Export optional environment variables from YAML (e.g., LLM_AS_A_JUDGE_*)
    for k, v in (cfg.get("env", {}) or {}).items():  # iterate env entries
        if v is None:
            continue
        os.environ[str(k)] = str(v)  # set into process env
    _set_torch_env(cfg)  # configure torch backends
    # Reduce CUDA fragmentation for long runs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Silence HF Trainer default info logs to avoid duplicate dict prints
    try:
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Load pretokenized datasets
    cache_dir = cfg["data"]["dataset_cache_dir"]  # path to cached dataset
    if not os.path.exists(cache_dir):  # ensure dataset is present
        # If configured to pretokenize automatically, run the preprocessor now
        if bool(cfg["data"].get("pretokenize", False)):
            print(f"Dataset cache not found at {cache_dir}; running pretokenization...")
            try:
                # Prefer package import if available
                try:
                    from unsloth_sft.preprocess import main as preprocess_main  # type: ignore
                except Exception:
                    # Fallback: import by file path to be robust to invocation style
                    import importlib.util, pathlib
                    pre_path = pathlib.Path(__file__).with_name("preprocess.py")
                    spec = importlib.util.spec_from_file_location("unsloth_sft.preprocess", str(pre_path))
                    if spec is None or spec.loader is None:
                        raise RuntimeError("Cannot load preprocess.py spec")
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    preprocess_main = getattr(mod, "main")
                preprocess_main(config_path)
            except Exception as e:
                raise RuntimeError(f"Pretokenization failed: {e}")
        # Re-check existence after attempted pretokenization
        if not os.path.exists(cache_dir):
            raise RuntimeError(
                f"Cached dataset not found at {cache_dir}. Run preprocessing first: python unsloth_sft/preprocess.py --config {config_path}"
            )
    ds = load_from_disk(cache_dir)  # load tokenized dataset dict

    # Load model + tokenizer
    model, tokenizer = _load_model_and_tokenizer(cfg)  # returns prepared model/tokenizer

    # Some multimodal models return a Processor without `pad`; use its underlying tokenizer for collation
    collate_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if not hasattr(collate_tokenizer, "pad"):
        # Fallback: reload a text tokenizer for collation
        collate_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name_or_path"], use_fast=True)
        if collate_tokenizer.pad_token is None:
            collate_tokenizer.pad_token = collate_tokenizer.eos_token
    # Build a robust collator that pads batch to max length without relying on processor.pad
    pad_id = collate_tokenizer.pad_token_id if hasattr(collate_tokenizer, "pad_token_id") else None
    eos_id = getattr(collate_tokenizer, "eos_token_id", None)
    default_pad_id = pad_id if pad_id is not None else (eos_id if eos_id is not None else 0)

    def simple_lm_collator(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = f["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [default_pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            # labels equal to input_ids for LM
            lab = f.get("labels", ids)
            labels.append(lab + [ -100 ] * pad_len)
        import torch
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    collator = simple_lm_collator

    # Collect optional dataloader args with version checks
    tcfg = cfg["training"]
    ta_kwargs = {
        "output_dir": tcfg["output_dir"],
        "per_device_train_batch_size": _as_int(tcfg["per_device_train_batch_size"], "training.per_device_train_batch_size"),
        "per_device_eval_batch_size": _as_int(tcfg["per_device_eval_batch_size"], "training.per_device_eval_batch_size"),
        "gradient_accumulation_steps": _as_int(tcfg["gradient_accumulation_steps"], "training.gradient_accumulation_steps"),
        "max_steps": _as_int(tcfg["max_steps"], "training.max_steps"),
        "learning_rate": _as_float(tcfg["learning_rate"], "training.learning_rate"),
        "lr_scheduler_type": str(tcfg["lr_scheduler_type"]),
        "warmup_ratio": _as_float(tcfg["warmup_ratio"], "training.warmup_ratio"),
        "weight_decay": _as_float(tcfg["weight_decay"], "training.weight_decay"),
        "logging_steps": _as_int(tcfg["logging_steps"], "training.logging_steps"),
        # evaluation strategy key changed in some HF versions
        # set eval_strategy below based on config
        "eval_steps": _as_int(tcfg["eval_steps"], "training.eval_steps"),
        "save_strategy": str(tcfg["save_strategy"]),
        "save_steps": _as_int(tcfg["save_steps"], "training.save_steps"),
        "save_total_limit": _as_int(tcfg["save_total_limit"], "training.save_total_limit"),
        "load_best_model_at_end": bool(tcfg["load_best_model_at_end"]),
        "metric_for_best_model": str(tcfg["metric_for_best_model"]),
        "greater_is_better": bool(tcfg["greater_is_better"]),
        "bf16": bool(tcfg.get("bf16", True)),
        "fp16": False,
        "tf32": bool(tcfg.get("tf32", True)),
        "seed": _as_int(tcfg.get("seed", 42), "training.seed"),
        "optim": str(tcfg.get("optim", "adamw_bnb_8bit")),
        "adam_beta1": _as_float(tcfg.get("adam_beta1", 0.9), "training.adam_beta1"),
        "adam_beta2": _as_float(tcfg.get("adam_beta2", 0.999), "training.adam_beta2"),
        "adam_epsilon": _as_float(tcfg.get("adam_epsilon", 1e-8), "training.adam_epsilon"),
        "max_grad_norm": _as_float(tcfg.get("max_grad_norm", 1.0), "training.max_grad_norm"),
        "dataloader_pin_memory": bool(tcfg.get("dataloader_pin_memory", True)),
        "dataloader_num_workers": _as_int(tcfg.get("dataloader_num_workers", 0), "training.dataloader_num_workers"),
        "dataloader_drop_last": bool(tcfg.get("dataloader_drop_last", False)),
        "report_to": ["none"],
    }
    # Handle eval/evaluation strategy compatibility
    eval_strategy = str(cfg["training"].get("evaluation_strategy", cfg["training"].get("eval_strategy", "no")))
    # Transformers >=4.55 uses 'eval_strategy'
    ta_kwargs["eval_strategy"] = eval_strategy
    # Optional fields if supported by current transformers
    if hasattr(TrainingArguments, "dataloader_prefetch_factor"):
        ta_kwargs["dataloader_prefetch_factor"] = _as_int(tcfg.get("dataloader_prefetch_factor", 2), "training.dataloader_prefetch_factor")
    if hasattr(TrainingArguments, "dataloader_persistent_workers"):
        ta_kwargs["dataloader_persistent_workers"] = bool(tcfg.get("dataloader_persistent_workers", True))
    if "warmup_steps" in tcfg:
        ta_kwargs["warmup_steps"] = _as_int(tcfg.get("warmup_steps", 0), "training.warmup_steps")

    # Optional: respect num_train_epochs if provided
    if "num_train_epochs" in cfg["training"]:
        ta_kwargs["num_train_epochs"] = float(cfg["training"]["num_train_epochs"])

    # Training arguments with checkpoint/eval cadence and dataloader controls
    train_args = TrainingArguments(**ta_kwargs)

    # Trainer
    trainer = Trainer(
        model=model,  # model
        args=train_args,  # args
        train_dataset=ds["train"],  # train split
        eval_dataset=ds["validation"],  # val split
        processing_class=collate_tokenizer,  # future-proof vs tokenizer deprecation
        data_collator=collator,  # collator
        callbacks=[RichLogCallback()] if _HAS_RICH else None,
    )

    # Auto-resume from latest checkpoint if available
    last_ckpt = None
    try:
        last_ckpt = get_last_checkpoint(train_args.output_dir)
    except Exception:
        last_ckpt = None
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
    # Train
    trainer.train(resume_from_checkpoint=last_ckpt)  # run training

    # Save final adapter checkpoint (best model is already loaded if configured)
    trainer.save_model(cfg["training"]["output_dir"])  # persist model


if __name__ == "__main__":  # CLI entry
    import argparse  # parse args

    parser = argparse.ArgumentParser(description="Unsloth SFT Trainer")  # build parser
    parser.add_argument("--config", type=str, default="configs/sft_unsloth.yaml", help="Path to YAML config")  # config arg
    args = parser.parse_args()  # parse
    main(args.config)  # run
