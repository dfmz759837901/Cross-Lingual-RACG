from typing import *
import requests
import time
import json

LLM_CIP = {
    "Qwen2.5-72B-Instruct-GPTQ-Int4",
    "DeepSeek-R1-Distill-Qwen-32B",
    "DeepSeek-R1",
    "QwQ-32B",
}


def get_messages_api(
    messages: List[Dict],
    model_name: str = "Qwen2.5-72B-Instruct-GPTQ-Int4",
    url: str = "your url",
    api_key: str = None,
    temperature: float = None,
    n: int = 1,
    max_tokens: int = None,
    max_new_tokens: int = None,
    stream: bool = False,
    debug: bool = False,
) -> List[str]:
    """
    Send a request to a remote model API to generate responses based on the input messages.
    """
    headers = {"Content-Type": "application/json"}
    if api_key is None:
        raise ValueError("API_key is None")
    headers["Authorization"] = f"Bearer {api_key}"

    if model_name in {"DeepSeek-R1"}:
        n = 1
    data = {
        "model": model_name,
        "messages": messages,
        "n": n,
        "stream": stream,
    }
    if temperature is not None:
        data["temperature"] = temperature
    if max_tokens is not None:
        data["max_tokens"] = max_tokens
    if max_new_tokens is not None:
        data["max_new_tokens"] = max_new_tokens

    if debug:
        print(f"data={data}")

    message = ""
    if not isinstance(message, Dict) or "choices" not in message:
        repeat_index = 0
        while not isinstance(message, Dict) or "choices" not in message:
            if repeat_index > 5:
                raise ConnectionError(f"{url} Error\nmessage=\n{message}")
            if debug:
                print(f"message=\n{message}")
            time.sleep(5)
            response = requests.post(url, json=data, headers=headers)
            if debug:
                print(f"response=\n{response.text}")
            message = json.loads(response.text)
            repeat_index += 1
        if not isinstance(message, Dict) or "choices" not in message:
            raise ConnectionError(f"{url} Error\nmessage=\n{message}")
    if len(message["choices"]) != n:
        raise ValueError(f"{model_name} response num error")
    return [message["choices"][i]["message"]["content"] for i in range(n)]


if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people solve problems.",
        },
        {
            "role": "user",
            "content": "Who are you?",
        },
    ]
    responses = get_messages_api(
        messages=messages,
        model_name="Qwen2.5-72B-Instruct-GPTQ-Int4",
        api_key="your api_key",
        temperature=0.8,
        max_tokens=2048,
        n=2,
        debug=False,
    )

    for response in responses:
        print("-" * 20 + "\n" + response)
