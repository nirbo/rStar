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
    TrainerCallback,
)
from typing import Optional

try:
    from rich.console import Console
    from rich.text import Text
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

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


class RichLogCallback(TrainerCallback):
    """Pretty-print training logs with colorblind-friendly value colors."""
    def __init__(self):
        self.console: Optional[Console] = Console(force_jupyter=False) if _HAS_RICH else None
        # value colors (non-bold): loss, grad_norm, learning_rate, epoch
        self.colors = {
            "loss": "cyan",
            "grad_norm": "green3",
            "learning_rate": "orange3",
            "epoch": "magenta",
        }

    @staticmethod
    def _fmt_lr(lr: float) -> str:
        # Force non-scientific decimal, reasonable precision, strip trailing zeros
        s = f"{float(lr):.10f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs or self.console is None:
            return
        parts: list[Text] = []
        for k in ("loss", "grad_norm", "learning_rate", "epoch"):
            if k in logs:
                val = logs[k]
                if k == "learning_rate":
                    try:
                        val = self._fmt_lr(float(val))
                    except Exception:
                        val = str(val)
                else:
                    try:
                        val = f"{float(val):.4f}"
                    except Exception:
                        val = str(val)
                parts.append(Text(f"{k}: "))
                parts.append(Text(str(val), style=self.colors.get(k, "white")))
                parts.append(Text("  "))
        if parts:
            msg = Text().join(parts)
            self.console.print(msg)


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
    ta_kwargs = {
        "output_dir": cfg["training"]["output_dir"],
        "per_device_train_batch_size": int(cfg["training"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["training"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["training"]["gradient_accumulation_steps"]),
        "max_steps": int(cfg["training"]["max_steps"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "lr_scheduler_type": str(cfg["training"]["lr_scheduler_type"]),
        "warmup_ratio": float(cfg["training"]["warmup_ratio"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "logging_steps": int(cfg["training"]["logging_steps"]),
        # evaluation strategy key changed in some HF versions
        # set eval_strategy below based on config
        "eval_steps": int(cfg["training"]["eval_steps"]),
        "save_strategy": str(cfg["training"]["save_strategy"]),
        "save_steps": int(cfg["training"]["save_steps"]),
        "save_total_limit": int(cfg["training"]["save_total_limit"]),
        "load_best_model_at_end": bool(cfg["training"]["load_best_model_at_end"]),
        "metric_for_best_model": str(cfg["training"]["metric_for_best_model"]),
        "greater_is_better": bool(cfg["training"]["greater_is_better"]),
        "bf16": bool(cfg["training"].get("bf16", True)),
        "fp16": False,
        "tf32": bool(cfg["training"].get("tf32", True)),
        "seed": int(cfg["training"].get("seed", 42)),
        "optim": str(cfg["training"].get("optim", "adamw_bnb_8bit")),
        "adam_beta1": float(cfg["training"].get("adam_beta1", 0.9)),
        "adam_beta2": float(cfg["training"].get("adam_beta2", 0.999)),
        "adam_epsilon": float(cfg["training"].get("adam_epsilon", 1e-8)),
        "max_grad_norm": float(cfg["training"].get("max_grad_norm", 1.0)),
        "dataloader_pin_memory": bool(cfg["training"].get("dataloader_pin_memory", True)),
        "dataloader_num_workers": int(cfg["training"].get("dataloader_num_workers", 0)),
        "dataloader_drop_last": bool(cfg["training"].get("dataloader_drop_last", False)),
        "report_to": ["none"],
    }
    # Handle eval/evaluation strategy compatibility
    eval_strategy = str(cfg["training"].get("evaluation_strategy", cfg["training"].get("eval_strategy", "no")))
    # Transformers >=4.55 uses 'eval_strategy'
    ta_kwargs["eval_strategy"] = eval_strategy
    # Optional fields if supported by current transformers
    if hasattr(TrainingArguments, "dataloader_prefetch_factor"):
        ta_kwargs["dataloader_prefetch_factor"] = int(cfg["training"].get("dataloader_prefetch_factor", 2))
    if hasattr(TrainingArguments, "dataloader_persistent_workers"):
        ta_kwargs["dataloader_persistent_workers"] = bool(cfg["training"].get("dataloader_persistent_workers", True))
    if "warmup_steps" in cfg["training"]:
        ta_kwargs["warmup_steps"] = int(cfg["training"].get("warmup_steps", 0))

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

    # Train
    trainer.train()  # run training

    # Save final adapter checkpoint (best model is already loaded if configured)
    trainer.save_model(cfg["training"]["output_dir"])  # persist model


if __name__ == "__main__":  # CLI entry
    import argparse  # parse args

    parser = argparse.ArgumentParser(description="Unsloth SFT Trainer")  # build parser
    parser.add_argument("--config", type=str, default="configs/sft_unsloth.yaml", help="Path to YAML config")  # config arg
    args = parser.parse_args()  # parse
    main(args.config)  # run
