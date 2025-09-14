# Training Guide

This guide explains how to set up single‑GPU SFT training with Unsloth for Gemma 3 models, prepare datasets with pretokenization and caching, and launch the rStar tool services (Redis, Code Judge, vLLM) via Docker Compose. It is optimized for a 32GB GPU and supports local model directories.

## Prerequisites

- GPU + CUDA drivers (bf16/TF32 recommended on recent NVIDIA GPUs)
- Python 3.10+
- Python packages:
  - `pip install unsloth transformers datasets peft bitsandbytes accelerate pyyaml`
  - Optional: `pip install flash-attn --no-build-isolation`
- Docker + Docker Compose (for rStar services)
- NVIDIA Container Toolkit (to pass GPU into vLLM container)

## Local Model

- Place your model in a local directory in standard Hugging Face format: `config.json`, tokenizer files, `generation_config.json`, and `model.safetensors` (sharded is fine).
- In `configs/sft_unsloth.yaml`, set:
  - `model.model_name_or_path: /abs/path/to/your/model`
- The same path can be mounted into Docker for vLLM via `.env.rstar` → `MODEL_PATH`.

## Dataset Format (JSONL)

- Training data is expected as JSONL with keys configured in YAML:
  - `data.input_key` (default `question`)
  - `data.target_key` (default `answer`)
- Example line:
  ```json
  {"question": "What is 12*13?", "answer": "156"}
  ```
- Notes:
  - The preprocessor supports a non‑reasoning SFT style: if answers are multi‑line, it keeps the last line when `non_reasoning: true`.
  - Optionally wrap final answers in `<answer>...</answer>` via `wrap_answer_in_tags: true`.

## Configure SFT (YAML)

Edit `configs/sft_unsloth.yaml`:

- Model (single‑GPU, 32GB friendly):
  - `model_name_or_path`: local path or HF repo id
  - `load_in_4bit: true` (QLoRA)
  - `gradient_checkpointing: true`, `flash_attention_2: true` (if installed)
  - `qlora: true` (optional `rslora: true` if supported)
- Data:
  - `train_jsonl_path`: your JSONL
  - `val_jsonl_path`: optional; if null, eval split is taken from train via `eval_split_ratio`
  - `max_seq_len`: start with 4096 and tune
  - `pretokenize: true`, `dataset_cache_dir`: where tokenized shards are stored
  - `remove_long_samples: true` to drop exceeding samples (avoid training on truncated data)
  - `num_proc`: CPU workers for preprocessing
- Training:
  - `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 16–32`
  - `save_steps`, `save_total_limit`, `eval_steps` control cadence
  - `load_best_model_at_end: true`, `metric_for_best_model: eval_loss`, `greater_is_better: false`
  - Dataloader controls (single-GPU tuning):
    - `dataloader_num_workers`: 0–4 on 32GB boxes (try 4)
    - `dataloader_pin_memory`: true/false (true improves H2D throughput)
    - `dataloader_drop_last`: false (usually fine for SFT)
    - `dataloader_prefetch_factor`: 2 (if supported by your transformers version)
    - `dataloader_persistent_workers`: true (if supported)
  - Optimizer (critical):
    - `optim`: e.g., `adamw_bnb_8bit` (good for QLoRA; requires bitsandbytes) or `adamw_torch_fused` (if available)
    - `adam_beta1`, `adam_beta2`, `adam_epsilon`, `max_grad_norm`
    - `warmup_ratio` (already present) or `warmup_steps` (takes precedence if set)

- Env (optional):
  - You can centralize OpenAI-compatible "LLM-as-a-judge" settings in the same YAML under `env`. They are exported to the process before preprocessing/training runs.
  - Example:
    ```yaml
    env:
      LLM_AS_A_JUDGE_BASE: https://openrouter.ai/api/v1
      LLM_AS_A_JUDGE_API_KEY: ""   # set your key here or via shell env
      LLM_AS_A_JUDGE_MODEL: nvidia/nemotron-nano-9b-v2:free
    ```
  - Note: This is for RL grading components (not the Code Judge container). Do not commit real keys.

## Pretokenize and Cache

- If `data.pretokenize: true` and the cache is missing, `train_sft.py` will automatically run the pretokenizer before training.
- You can also run pretokenization explicitly; subsequent runs reuse the cached dataset.
- Command:
  ```bash
  python unsloth_sft/preprocess.py --config configs/sft_unsloth.yaml
  ```
- Behavior:
  - Formats text from your JSONL
  - Tokenizes without truncation and drops sequences longer than `max_seq_len`
  - Writes `train` and `validation` splits to `dataset_cache_dir`

## Train SFT

- Command:
  ```bash
  python unsloth_sft/train_sft.py --config configs/sft_unsloth.yaml
  ```
- Features:
  - 4‑bit QLoRA adapters on top of the base model
  - Saves every `save_steps`, keeps last `save_total_limit`
  - Evaluates every `eval_steps`; tracks best by lowest `eval_loss`
- Output:
  - Adapters and trainer state saved under `training.output_dir`

## Serving Notes (vLLM)

- vLLM can serve the base model. To serve the fine‑tuned adapter, either:
  - Merge LoRA into base weights offline and point vLLM to the merged directory, or
  - Use vLLM’s LoRA/adapter support (if available in your version) to load adapters at runtime.

## rStar Services via Docker Compose

- One‑shot launcher:
  ```bash
  ./scripts/start_rstar_services.sh
  ```
- First run creates `.env.rstar` (edit `MODEL_PATH` to your local model directory).
- Services:
  - Redis (queue)
  - Code Judge API and workers (executes Python tool calls)
  - vLLM server (OpenAI‑compatible)
- Verify:
  ```bash
  curl http://localhost:8000/v1/models
  # Code Judge at http://localhost:8088
  ```
- Security: Code Judge executes arbitrary code; keep it isolated and never expose publicly.

### .env.rstar Checklist (Required)

Open `.env.rstar` and set these before starting services:

- `MODEL_PATH`: absolute path on host to your model directory (HF format). Must exist.
- `VLLM_HOST`: usually `0.0.0.0`
- `VLLM_PORT`: e.g., `8000`
- `CODE_JUDGE_PORT`: e.g., `8088`
- `REDIS_URI`: `redis://redis:6379`
- `MAX_EXECUTION_TIME`: e.g., `4`
- `MAX_WORKERS`: e.g., `64`

If any are missing/empty, the launcher now fails fast with a clear message. The most common cause of `invalid spec: ::ro` is an empty `MODEL_PATH`, which breaks the volume mapping.

### GPU Troubleshooting

- Verify host driver/toolkit:
  ```bash
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  Both should succeed. If the second fails, (re)install NVIDIA Container Toolkit:
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

- Compose GPU config:
  - The `vllm` service uses `gpus: all` (works with Docker Compose v2). `deploy.resources.devices` is ignored outside Swarm.
  - The launcher validates `MODEL_PATH` exists; an empty path causes volume errors.

- Common errors:
  - `nvidia-container-cli: initialization error: nvml error:: unknown` → usually driver/toolkit mismatch. Reboot after driver updates; ensure `docker run --gpus all ... nvidia-smi` works.
  - `invalid spec: ::ro` → empty `MODEL_PATH` in `.env.rstar`.

If the sanity script shows libnvidia-ml from `/usr/local/cuda/...` (likely a stub):
- Install the driver’s NVML runtime and utils, matching your driver branch (e.g., 560):
  ```bash
  sudo apt-get update
  sudo apt-get install -y libnvidia-ml1 nvidia-utils-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
  sudo ldconfig
  ldconfig -p | grep libnvidia-ml
  ls -l /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
  ```
- Ensure ldconfig prefers `/usr/lib/x86_64-linux-gnu` over CUDA toolkit paths for libnvidia-ml. If needed, edit `/etc/ld.so.conf.d/` entries (e.g., CUDA’s conf) so the driver path takes precedence, then run `sudo ldconfig`.

### Quick Sanity Script

Run our helper to check driver, NVML, Docker runtime, and the CUDA test container:

```bash
chmod +x scripts/gpu_sanity.sh
./scripts/gpu_sanity.sh
```

It reports PASS/FAIL for each step and suggests fixes. Only proceed to start services once the CUDA test container passes.

### Judge Setup Examples

- Example: Local judge + local vLLM
  - Start services:
    ```bash
    ./scripts/start_rstar_services.sh
    ```
  - Verify endpoints:
    ```bash
    curl http://localhost:8000/v1/models
    # Code Judge is served at http://localhost:8088
    ```
  - Quick interactive run with local vLLM:
    ```bash
    python examples/chat_with_tool_call.py \
      --model /abs/path/to/your/model \
      --prompt "Solve: 2x + 3y = 7; x - y = 1" \
      --max_tokens 4096 \
      --base_url http://localhost:8000/v1/completions
    ```

- Example: Point Code Judge to a remote host (use in your own code)
  ```python
  from rstar2_agent.tools.code_judge_utils import run_tool_calls_on_server_async

  # override host/port from defaults (localhost:8088)
  responses = await run_tool_calls_on_server_async(
      tool_calls=tool_calls,
      session=session,
      generate_tool_call_code=generate_tool_call_code,
      generate_tool_call_input=generate_tool_call_input,
      host_addr="judge.example.com",
      host_port="80",
  )
  ```
  Note: the example script `examples/chat_with_tool_call.py` uses localhost; for a remote judge, adapt similarly in your integration.

- Example: Use a remote OpenAI-compatible LLM (self-hosted or provider)
  - CLI using the new flags:
    ```bash
    python examples/chat_with_tool_call.py \
      --model /abs/path/to/local/tokenizer/or/model \
      --remote_model your-remote-model-name \
      --base_url https://api.example.com/v1/completions \
      --api_key YOUR_API_KEY \
      --judge_host localhost \
      --judge_port 8088 \
      --prompt "Compute 123*456" \
      --max_tokens 1024
    ```
    This keeps the local tokenizer for prompt formatting while targeting a remote model name at the provider.
  - curl example:
    ```bash
    BASE_URL="https://api.example.com"
    API_KEY="YOUR_API_KEY"
    curl -X POST "$BASE_URL/v1/completions" \
      -H "Authorization: Bearer $API_KEY" \
      -H "Content-Type: application/json" \
      -d '{"model":"your-remote-model-name","prompt":[...],"max_tokens":4096}'
    ```
    Adjust the path if your provider uses `/v1/chat/completions` and a chat payload.

- Example: OpenRouter (model: `nvidia/nemotron-nano-9b-v2:free`)
  - OpenRouter uses the Chat Completions API:
    - Base URL: `https://openrouter.ai/api/v1/chat/completions`
    - Auth: `Authorization: Bearer $OPENROUTER_API_KEY`
    - Model: `nvidia/nemotron-nano-9b-v2:free`
  - curl example:
    ```bash
    export OPENROUTER_API_KEY=sk-or-...  # your key
    curl -sS https://openrouter.ai/api/v1/chat/completions \
      -H "Authorization: Bearer $OPENROUTER_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
            "model": "nvidia/nemotron-nano-9b-v2:free",
            "messages": [{"role":"user","content":"Compute 123*456 and return only the number."}],
            "max_tokens": 256
          }'
    ```
  - Python example:
    ```python
    import os, requests
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    data = {
      "model": "nvidia/nemotron-nano-9b-v2:free",
      "messages": [{"role": "user", "content": "Compute 123*456 and return only the number."}],
      "max_tokens": 256,
    }
    print(requests.post(url, json=data, headers=headers).json())
    ```
  - Note: `examples/chat_with_tool_call.py` targets an OpenAI-style Completions endpoint with a tokenized prompt. For OpenRouter, prefer the Chat Completions API as shown above, or adapt the script to send a `messages` payload when `--base_url` points to `/v1/chat/completions`.

### LLM-as-a-Judge (for RL grading, not Code Judge)

- Some RL recipes use a separate LLM to grade answers. In this repo, `verl/recipe/deepeyes/deepeyes.py` supports configuring an OpenAI-compatible endpoint via environment variables:
  - `LLM_AS_A_JUDGE_BASE`: base URL, e.g., `https://openrouter.ai/api/v1`
  - `LLM_AS_A_JUDGE_API_KEY`: API key (if required by the provider)
  - `LLM_AS_A_JUDGE_MODEL`: model name, e.g., `nvidia/nemotron-nano-9b-v2:free`

- Example configuration for OpenRouter:
  ```bash
  export LLM_AS_A_JUDGE_BASE="https://openrouter.ai/api/v1"
  export LLM_AS_A_JUDGE_API_KEY="sk-or-..."
  export LLM_AS_A_JUDGE_MODEL="nvidia/nemotron-nano-9b-v2:free"
  # then run your RL script that uses the deepeyes reward component
  ```

- Important: This LLM-as-a-judge is separate from the Code Judge service. Code Judge executes Python code; it does not use or require an LLM.

## VRAM Tips (32GB)

- Prefer `gemma-3-12b-it` with QLoRA 4‑bit; drop to 7B if OOM persists
- Start with `max_seq_len=4096`, `accumulation=16–32`
- Reduce `lora_r` or `max_seq_len` before reducing batch size
- Ensure bf16/TF32 are enabled on supported GPUs
- Disable FlashAttention if not installed

## Evaluation Strategy

- If `val_jsonl_path` is null, the preprocessor splits the train file using `data.eval_split_ratio`.
- Best model is tracked by lowest `eval_loss`.

## Troubleshooting

- OOM: reduce `max_seq_len`, increase `gradient_accumulation_steps`, lower `lora_r`, or switch to 7B
- Flash‑Attn build issues: skip it or use prebuilt wheels matching CUDA/arch
- bitsandbytes not loading: ensure CUDA version matches your PyTorch build
- Empty dataset after preprocessing: increase `max_seq_len` or inspect input formatting

## Next Steps: RL (Preview)

- A minimal single‑GPU GRPO‑RoC loop can be added next:
  - Small group size (e.g., G=4 oversample→8, select 4)
  - Shorter max response and limited turns per rollout
  - Answer‑only reward; simple integer checker; RoC sampling on positives
  - Reuse Unsloth adapters for policy/reference efficiency
- This will integrate with the same config pattern and checkpoints.
