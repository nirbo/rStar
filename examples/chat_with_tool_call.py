# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
# Launch VLLM server

vllm serve /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Check if the server is running well

curl http://localhost:8000/v1/models
"""
import aiohttp
import argparse
import asyncio
import json
import requests
import yaml

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.tools.schemas import ToolResponse

from rstar2_agent.tools.code_judge_utils import (
    run_tool_calls_on_server_async,
    generate_tool_call_code,
    generate_tool_call_input,
)
from rstar2_agent.tools.tool_parser import (
    RStar2AgentHermesToolParser,
)


async def run_tool_calls(tool_calls, judge_host: str, judge_port: str):
    tool_connector = aiohttp.TCPConnector(limit=32, force_close=True, enable_cleanup_closed=True)
    tool_timeout = aiohttp.ClientTimeout(total=60)
    tool_session = aiohttp.ClientSession(connector=tool_connector, timeout=tool_timeout)
    responses = await run_tool_calls_on_server_async(
        tool_calls=tool_calls,
        session=tool_session,
        generate_tool_call_code=generate_tool_call_code,
        generate_tool_call_input=generate_tool_call_input,
        host_addr=judge_host,
        host_port=judge_port,
    )
    await tool_session.close()
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/path/to/your/model")
    parser.add_argument("--prompt", default="Solve the system of equations: 2x + 3y = 7, x - y = 1")
    parser.add_argument("--max_tokens", default=8192)
    # New flags for OpenAI-compatible LLM endpoint configuration
    parser.add_argument("--base_url", default="http://localhost:8000/v1/completions", help="OpenAI-compatible base URL for completions endpoint")
    parser.add_argument("--api_key", default=None, help="Bearer API key for remote LLM provider (optional)")
    parser.add_argument("--remote_model", default=None, help="Model name at remote provider; defaults to --model if unset")
    # Code Judge host/port
    parser.add_argument("--judge_host", default="localhost", help="Code Judge host address (default: localhost)")
    parser.add_argument("--judge_port", default="8088", help="Code Judge port (default: 8088)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    python_tool_yaml_path = project_dir / "rstar2_agent/config/tool_config/python_tool_config.yaml"
    with python_tool_yaml_path.open() as file:
        python_tool_schema = yaml.safe_load(file)["tools"][0]["tool_schema"]

    tools = [python_tool_schema]
    url = args.base_url  # allow overriding base URL
    budget = int(args.max_tokens)
    prompt = f"You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.\nThis is the problem:\n{args.prompt}"

    messages = [{"role": "user", "content": prompt}]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    parser = RStar2AgentHermesToolParser(tokenizer)

    prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    print(prompt)

    input_ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)
    prompt_len = len(input_ids)

    payload = {
        "model": (args.remote_model or args.model),
        "prompt": input_ids,
        "temperature": 1.0,
        "max_tokens": budget,
        "skip_special_tokens": False,
        "include_stop_str_in_output": True
    }
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    response = requests.post(url, json=payload, headers=headers).json()
    response: str = response["choices"][0]["text"]
    print(response)
    _, tool_calls = asyncio.run(parser.extract_tool_calls(responses_ids=tokenizer.encode(response)))

    while tool_calls:
        # execute tool call
        total_tool_responses, filtered_tool_calls, pending_pos = [], [], []
        for i, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, ToolResponse):
                total_tool_responses.append(tool_call.text)
            else:
                total_tool_responses.append(None)
                pending_pos.append(i)
                filtered_tool_calls.append(tool_call)

        if filtered_tool_calls:
            filtered_tool_calls = [{
                "name": tool_call.name,
                "arguments": json.loads(tool_call.arguments),
            } for tool_call in filtered_tool_calls]
            filtered_tool_responses = asyncio.run(run_tool_calls(filtered_tool_calls, args.judge_host, args.judge_port))
            for i, tool_response in zip(pending_pos, filtered_tool_responses):
                total_tool_responses[i] = tool_response

        # append assistant response to messages
        if response.endswith("<|im_end|>"):
            response = response[: -len("<|im_end|>")]
        assistant_msg = f"<reason>{response}"
        messages.append({"role": "assistant", "content": assistant_msg})

        prefix_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)

        # append tool responses to messages
        for tool_response in total_tool_responses:
            messages.append({"role": "tool", "content": tool_response})

        entire_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        print(entire_text[len(prefix_text):])

        # next turn generation
        input_ids = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True, add_generation_prompt=True)
        if budget > (len(input_ids) - prompt_len):
            payload = {
                "model": (args.remote_model or args.model),
                "prompt": input_ids,
                "temperature": 1.0,
                "max_tokens": budget - (len(input_ids) - prompt_len),
                "skip_special_tokens": False,
                "include_stop_str_in_output": True
            }
            response = requests.post(url, json=payload, headers=headers).json()
            response: str = response["choices"][0]["text"]
            print(response)
            _, tool_calls = asyncio.run(parser.extract_tool_calls(responses_ids=tokenizer.encode(response)))
        else:
            print(f"[Generation end: reached the maximum generation token number {(len(input_ids) - prompt_len)}]")
            break
