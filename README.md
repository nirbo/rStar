<h1 align="center">
<br>
rStar2-Agent
</h1>

<p align="center">
📃 <a href="https://huggingface.co/papers/2508.20722" target="_blank">[Paper]</a> 
</p>

Repo for "[rStar2-Agent: Agentic Reasoning Technical Report](https://huggingface.co/papers/2508.20722)".

Authors: Ning Shang\*, Yifei Liu\*, Yi Zhu\*, Li Lyna Zhang\*†, Weijiang Xu, Xinyu Guan, Buze Zhang, Bingcheng Dong, Xudong Zhou, Bowen Zhang, Ying Xin, Ziming Miao, Scarlett Li, Fan Yang, Mao Yang†

<p align="center">
    <img src="images/figure-1.png" width="1000">
        <br>
    <em>Figure 1: rStar2-Agent-14B reaches frontier-level math reasoning in just 510 RL training step</em>
</p>

## News 

- **[07/15/2025]** Our rStar-Coder [paper](https://arxiv.org/abs/2505.21297) and [dataset](https://huggingface.co/datasets/microsoft/rStar-Coder) are released. We introduce a large-scale, verified dataset of 418K competition-level code problems with **test cases** of varying difficulty, enabling small LLMs (1.5B-14B) to achieve frontier-level code reasoning performance.
- **[02/10/2025]** We are hiring interns! If you are interested in improving LLM reasoning, please send your CV to lzhani@microsoft.com.
- **[01/21/2025]** rStar-Math code has been open-sourced. 
- **[01/09/2025]** rStar-Math paper is released: https://huggingface.co/papers/2501.04519.

Note: Our prior work [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://huggingface.co/papers/2408.06195) is open-sourced on the [rStar-mutualreasoning b](https://github.com/microsoft/rStar/tree/rStar-mutualreasoning) branch.

Note: Our prior work [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://huggingface.co/papers/2501.04519) is open-sourced on the [rStar-math](https://github.com/microsoft/rStar/tree/rStar-math) branch.

## Contents
- [Introduction](#Introduction)
- [Try rStar2-Agent with Tool Calling](#Try-rStar2-Agent-with-Tool-Calling)
- [Evaluation](#Evaluation)
- [rStar2-Agent RL Training](#rStar2-Agent-RL-Training)
- [Citation](#Citation)

## Introduction
We introduce rStar2-Agent, a 14B math reasoning model that thinks smarter rather than merely longer, achieving performance comparable to 671B DeepSeek-R1 through pure agentic reinforcement learning. The model plans, reasons, and autonomously uses coding tools to efficiently explore, verify, and reflect for more complex problem-solving. This capability relies on three key innovations: (i) GRPO-RoC, an effective agentic reinforcement learning algorithm with a novel Resample-on-Correct rollout strategy that optimizes coding tool usage and enables shorter, smarter reasoning by selectively retaining higher-quality positive trajectories while preserving all failure cases; (ii) a scalable and efficient RL infrastructure that supports high-throughput tool call execution and mitigates the high costs of agentic RL rollout, enabling efficient training on limited GPU resources (64 MI300X GPUs); (iii) an agent training recipe that starts with non-reasoning SFT and proceeds through multi-stage RL with concise maximum response lengths per stage and increasing dataset difficulty. To this end, rStar2-Agent boosts a pre-trained 14B model to state-of-the-art levels in only 510 RL steps within one week, achieving 80.6% and 69.8% average pass@1 on AIME24 and AIME25, surpassing DeepSeek-R1 (671B) with shorter responses. Beyond mathematics, rStar2-Agent-14B also demonstrates strong generalization to alignment, scientific reasoning, and agentic tool-use tasks.

## Try rStar2-Agent with Tool Calling

### Installation

#### Option 1: Manual Installation

```bash
# Initialize and update submodules
git submodule init
git submodule update

# install verl
pip install "torch<2.8"
pip install -r verl/requirements_sglang.txt
pip install -e verl

# install code judge
pip install -r code-judge/requirements.txt
pip install -e code-judge

# install rstar2_agent
pip install -e .
```

#### Option 2: Automated Installation

```bash
bash install.sh
```

### Code Judge Server Setup

> ⚠️ **Security Warning**: Code Judge executes arbitrary code. Always deploy in an isolated environment (preferably Docker) and never expose to external networks.

The rStar2-Agent uses Code Judge as a tool call server to execute model-generated Python code.

#### 1. Start Redis Server

```bash
sudo apt-get update -y && sudo apt-get install redis -y
redis-server --daemonize yes --protected-mode no --bind 0.0.0.0
```

#### 2. Launch Code Judge Server

```bash
# Start the main server (master node only)
# Environment variables can be configured as per: https://github.com/0xWJ/code-judge/blob/main/app/config.py
# Replace $WORKSPACE and $MASTER_ADDR with your actual paths

tmux new-session -d -s server \
  'cd $WORKSPACE/code-judge && \
   MAX_EXECUTION_TIME=4 \
   REDIS_URI="redis://$MASTER_ADDR:6379" \
   RUN_WORKERS=0 \
   uvicorn app.main:app --host 0.0.0.0 --port 8088 --workers 16 \
   2>&1 | tee server.log'
```

#### 3. Start Code Judge Workers

```bash
# Launch workers (can be deployed on multiple nodes for increased parallelism)
# Adjust MAX_WORKERS based on your CPU count per node

tmux new-session -d -s worker \
  'cd $WORKSPACE/code-judge && \
   MAX_EXECUTION_TIME=4 \
   REDIS_URI="redis://$MASTER_ADDR:6379" \
   MAX_WORKERS=64 \
   python run_workers.py \
   2>&1 | tee worker.log'
```

### Launch the VLLM Server

First, start the VLLM server:

```bash
vllm serve /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

Replace `/path/to/your/model` with the actual path to your downloaded model.

### Verify Server Status

Check if the server is running properly:

```bash
curl http://localhost:8000/v1/models
```

### Run Interactive Chat with Tool Calling

Use the provided script to interact with your model:

```bash
python examples/chat_with_tool_call.py \
    --model /path/to/your/model \
    --prompt "Solve the system of equations: 2x + 3y = 7, x - y = 1" \
    --max_tokens 8192
```

### Script Options

The `examples/chat_with_tool_call.py` script supports the following arguments:

- `--model`: Path to your model
- `--prompt`: Input prompt for the model
- `--max_tokens`: Maximum number of tokens to generate

## Evaluation

### Environment Setup

Please view [Installation](#Installation) and [Code Judge Server Setup](#Code-Judge-Server-Setup).

### Run Evaluation Script

We evaluate following mathematical reasoning benchmarks:

- **AIME 2024/2025 (American Invitational Mathematics Examination)**: High-school level competition mathematics
- **MATH500**: A subset of the MATH dataset containing 500 challenging problems

```bash
MODEL_PATH=/path/to/your/model bash examples/aime_eval.sh
MODEL_PATH=/path/to/your/model bash examples/math500_eval.sh
```

## rStar2-Agent RL Training

A comprehensive reinforcement learning training framework for the rStar2-Agent, built on [Verl](https://github.com/volcengine/verl) and [Code Judge](https://github.com/0xWJ/code-judge). This framework enables training models after instruction-following supervised fine-tuning (SFT).

### Environment Setup

Please view [Installation](#Installation) and [Code Judge Server Setup](#Code-Judge-Server-Setup).

## Unsloth SFT (Single‑GPU, 32GB‑friendly)

This repository includes an Unsloth‑based SFT pipeline that is optimized for a single 32 GB GPU (e.g., RTX 5090) using QLoRA and efficient kernels. It supports very large datasets (e.g., MetaMathQA) and exposes controls for caching, JSON→JSONL conversion, batch/accumulation, optimizers, and dataloader tuning.

Key files
- `configs/sft_unsloth.yaml` – all knobs for model, data, training, runtime, and optional env vars
- `unsloth_sft/preprocess.py` – JSONL → tokenized HF `DatasetDict` cache with strict length filtering
- `unsloth_sft/train_sft.py` – Unsloth SFT runner; auto‑pretokenizes; auto‑resumes from latest checkpoint
- `scripts/json_to_jsonl.py` – helper to convert JSON arrays/dicts into JSONL

Highlights
- QLoRA 4‑bit by default; RSLoRA optional
- Gradient checkpointing: `true` (HF), `false`, or `"unsloth"` (Unsloth optimized)
- Pretokenization cache on disk; drops samples longer than `data.max_seq_len` to avoid truncation training
- Optional JSON→JSONL pre‑conversion controlled by YAML (no manual step required)
- Robust collator with dynamic padding per batch; batching can be further stabilized via length bucketing
- Trainer auto‑resumes from the latest checkpoint in `training.output_dir`

### Configure

Edit `configs/sft_unsloth.yaml`:

- Model
  - `model_name_or_path`: local path or HF repo id (local path supported)
  - `load_in_4bit: true` (QLoRA); `rslora: false|true`
  - `gradient_checkpointing: true|false|"unsloth"`
  - `target_modules`: LoRA target projection names

- Data
  - `train_jsonl_path`: path to JSONL or JSON
  - `input_key`, `target_key`: e.g., `query`/`response` for MetaMathQA
  - `max_seq_len`: guardrail; we drop longer samples
  - `dataset_cache_dir`: where tokenized dataset is written
  - `pretokenize: true`: runs tokenizer pipeline once and caches
  - `convert_to_jsonl: false|true`: if true and source is JSON array/dict, convert to JSONL before tokenization
    - `convert_to_jsonl_include_keys`: optional whitelist (e.g., `[query, response]`)
    - `convert_to_jsonl_renames`: optional mapping (e.g., `{query: problem, response: answer}`)

- Training
  - Typical stable setting on a 5090 for math data: `per_device_train_batch_size: 20`, `gradient_accumulation_steps: 3`
  - `dataloader_drop_last: true` recommended for stability
  - `disable_tqdm: false`, `logging_strategy: steps` keep the progress bar and step logs
  - `save_steps`, `eval_steps`, `save_total_limit`: cadence and retention
  - Optimizer: `optim: adamw_bnb_8bit` (QLoRA‑friendly), betas/epsilon, `max_grad_norm`

### Run

If the cache is missing and `pretokenize: true`, the trainer will run the pretokenizer automatically before training.

```bash
python unsloth_sft/train_sft.py --config configs/sft_unsloth.yaml
```

Notes
- The trainer prints the effective total batch = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`.
- Training auto‑resumes from the latest checkpoint in `training.output_dir`.
- For long runs, consider `num_train_epochs: 1.0` (≈ one pass) or ~10k steps for ~385k samples at global batch ~36–60.

### JSON→JSONL Conversion (Optional)

If you point `data.train_jsonl_path` to a JSON file that is a list of objects (or a dict with a single list of objects), set `data.convert_to_jsonl: true` to convert it before tokenization. Optional knobs allow filtering keys and renaming fields to match `input_key`/`target_key`. A helper script is also available:

```bash
python scripts/json_to_jsonl.py \
  --input datasets/metamathqa.json \
  --output datasets/metamathqa.jsonl \
  --include-keys query response \
  --rename query=question --rename response=answer
```

### GPU Sanity and Services

- `scripts/gpu_sanity.sh` checks host driver, NVML library, Docker GPU runtime, and runs a CUDA test container.
- `scripts/start_rstar_services.sh` launches Redis, Code Judge, and vLLM via Compose; validates `.env.rstar` (MODEL_PATH, ports, etc.).

vLLM via Compose runs the OpenAI API server and is invoked with flags:

```yaml
command: >-
  --model ${MODEL_PATH}
  --host ${VLLM_HOST}
  --port ${VLLM_PORT}
  --max-model-len 8192
  --gpu-memory-utilization 0.9
  --enable-auto-tool-choice
  --tool-call-parser hermes
```

The example client `examples/chat_with_tool_call.py` supports `--base_url/--api_key/--remote_model` and `--judge_host/--judge_port` so you can target local vLLM or remote OpenAI‑compatible endpoints.

### Data Preparation

This example uses:
- **Training Dataset**: DAPO-17k (English subset)
- **Test Dataset**: AIME24

```bash
# Process AIME 2024 dataset
python data_preprocess/aime2024_rstar2_agent_loop.py

# Process DAPO dataset
python data_preprocess/dapo_rstar2_agent_loop.py
```

### Model Setup

Download the base model (Qwen3-14B-Base):

```bash
huggingface-cli download Qwen/Qwen3-14B-Base --local-dir $HOME/models/Qwen3-14B-Base
```

> **Note**: The base model requires instruction-following SFT before RL training for optimal performance.

### Training

#### Basic Training

Run the training script (for 8x A100/H100 GPUs):

```bash
bash examples/run_qwen3-14b_rstar2_agent_weave.sh
```

> Adjust configuration parameters based on your hardware environment.

### Configuration

#### Data Augmentation Settings

The framework supports various sampling strategies to improve training efficiency:

```bash
# Global Settings
augmentation.do_down_sampling=True                                   # Enable down sampling
augmentation.down_sampling_config.down_sample_to_n=16                # Target number of traces per data point

# Sampling Strategies
augmentation.down_sampling_config.reject_equal_reward=True           # Enable reject sampling for equal rewards
augmentation.down_sampling_config.roc_error_ratio=True               # Resample correct traces by tool call error ratio
augmentation.down_sampling_config.roc_answer_format=True             # Resample correct traces by answer format

# Minimum Trace Requirements
augmentation.down_sampling_config.min_zero_reward_trace_num=2        # Minimum negative traces to retain
augmentation.down_sampling_config.min_non_zero_reward_trace_num=2    # Minimum positive traces to retain
```

### Important Note

rStar2-Agent was originally training based on VERL v0.2 with our custom multi-turn tool calling training framework. The current training framework released here has been migrated to VERL v0.5 to ensure compatibility with the latest community standards. While this release framework hasn't been used to train a complete model yet, we have verified that the first 50 training steps show minimal differences between our original and migrated frameworks, maintaining the core functionality of our proven training approach.

Although our original framework includes additional advanced features such as rollout request load balance scheduler, we chose to migrate to the latest VERL version to maintain community compatibility and facilitate easier customization by users. This approach ensures you can benefit from ongoing VERL improvements and easily integrate with the latest open-source developments. We also consider migrating all features to the current version in the future.

If you encounter any issues during usage or need assistance with the training framework, please contact us.

### Troubleshooting

#### Common Issues

1. **Redis Connection Errors**: Ensure Redis is running and accessible at the specified address
2. **GPU Memory Issues**: Adjust batch sizes and model parameters for your hardware
3. **Code Judge Timeouts**: Increase `MAX_EXECUTION_TIME` for complex computations
4. **Worker Scaling**: Adjust `MAX_WORKERS` based on available CPU cores

#### Log Locations

- Server logs: `server.log` in the code-judge directory
- Worker logs: `worker.log` in the code-judge directory
- Training logs: Check your training script output directory

---


## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{shang2025rstar2agentagenticreasoningtechnical,
      title={rStar2-Agent: Agentic Reasoning Technical Report}, 
      author={Ning Shang and Yifei Liu and Yi Zhu and Li Lyna Zhang and Weijiang Xu and Xinyu Guan and Buze Zhang and Bingcheng Dong and Xudong Zhou and Bowen Zhang and Ying Xin and Ziming Miao and Scarlett Li and Fan Yang and Mao Yang},
      year={2025},
      eprint={2508.20722},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.20722}, 
}
```
