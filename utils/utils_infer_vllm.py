from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import *
import os
import time


def load_model_vllm(
    model_name_or_path: str, task: Literal["generate"] = "generate"
) -> Tuple[LLM, AutoTokenizer, Dict]:

    cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    print(f"set CUDA_VISIBLE_DEVICES = {cuda_devices}")
    num_gpus = len(cuda_devices.split(",")) if cuda_devices else 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    # Create an LLM.
    llm_config = {
        "model": model_name_or_path,
        "dtype": "auto",
        # dtype="bfloat16",
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": num_gpus,
        "max_num_seqs": 10,
        "max_num_batched_tokens": 8192,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.6 if num_gpus > 1 else 0.8,
        "trust_remote_code": True,
        "disable_custom_all_reduce": True,
    }
    while llm_config["max_model_len"] >= 2048:
        try:
            print(f"Loading model with max_model_len = {llm_config['max_model_len']}")
            llm = LLM(**llm_config)
            break
        except Exception:
            print("Loading model failed, retrying with smaller max_model_len")
            time.sleep(10)
            llm_config["max_model_len"] //= 2
            continue

    if llm_config["max_model_len"] < 2048:
        raise Exception("Failed to load model")
    return llm, tokenizer, llm_config


def get_responses_vllm(
    prompts: List[str],
    model_name: str,
    model: LLM,
    tokenizer: AutoTokenizer,
    temperature: float = 0.2,
    n: int = 1,
    max_tokens: int = 512,
    stop: Optional[Union[str, List[str]]] = None,
) -> List[List[str]]:
    if os.path.isdir(model_name):
        model_name = os.path.basename(model_name)
    if "chat" in model_name.lower():
        prompts: List[str] = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

    # Create a sampling params object.
    if temperature == 0.0:
        n = 1
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=50,
        top_p=0.95,
        stop=stop,
    )
    outputs = model.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    responses: List[List[str]] = []
    for i in range(len(prompts)):
        response: List[str] = []
        assert len(outputs[i].outputs) == n, (
            "\n"
            + "-" * 20
            + f"\n{prompts[i]}\n"
            + "-" * 20
            + f"\nlen(prompt_token_ids) = {len(outputs[i].prompt_token_ids)}\n"
            + f'prompt_token_ids_shape = {tokenizer(prompts[i], return_tensors="pt").input_ids.shape}\n'
            + f"-" * 20
            + f"\n{outputs[i].outputs[0]}\n"
            + "-" * 20
            + "\n"
        )
        for j in range(n):
            response.append(outputs[i].outputs[j].text)
        responses.append(response)
    return responses


if __name__ == "__main__":
    model_name_or_path = "../hf_models/codellama/CodeLlama-7b-Instruct-hf"
    model, tokenizer, _ = load_model_vllm(model_name_or_path)
    sample_n = 5
    prompts = [
        "bubble sort algorithm in python\n```python\n",
        "select sort algorithm in python\n```python\n",
        "quick sort algorithm in python\n```python\n",
        "insertion sort algorithm in python\n```python\n",
    ]
    responses = get_responses_vllm(
        prompts=prompts,
        model_name=model_name_or_path,
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        n=sample_n,
        max_tokens=128,
        stop=["\n```"],
    )
    assert len(responses) == len(prompts)
    for i in range(len(responses)):
        print(f"prompt[{i}]=\n")
        assert len(responses[i]) == sample_n
        for id, r in enumerate(responses[i]):
            print(f"response[{id}]=\n{r}\n")
