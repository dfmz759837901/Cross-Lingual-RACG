from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np
from typing import *
import torch
import math
import os
from tqdm.auto import tqdm


def load_model_hug(
    model_name_or_path: str, task: Literal["generate", "encode"] = "generate"
) -> Tuple[Union[AutoModelForCausalLM, AutoModel], AutoTokenizer, Dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if task == "generate":
        print("Loading model as AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(device)
    elif task == "encode":
        print("Loading model as AutoModel")
        model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(device)
    model.eval()
    print(f"Loaded model {model_name_or_path} on {device}")
    return model, tokenizer, model.config


def get_responses_hug(
    prompts: List[str],
    model_name: str,
    model: AutoModelForCausalLM,
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
    if temperature == 0.0:
        n = 1

    responses: List[List[str]] = []
    for p in tqdm(prompts):
        inputs = tokenizer.encode(p, return_tensors="pt").to(model.device)

        with torch.no_grad():
            if temperature > 0.0:
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=n,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
        assert len(outputs) == n

        response: List[str] = []
        for j in range(n):
            _response = tokenizer.decode(
                outputs[j][len(inputs[0]) :], skip_special_tokens=True
            )
            response.append(_response)
        responses.append(response)

    return responses


def embedding_sentences_hug(
    sentences: Sequence[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    convert_to_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    # Tokenize sentences
    model.eval()
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    ).to(model.device)
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    assert len(sentence_embeddings) == len(sentences)
    if convert_to_numpy and isinstance(sentence_embeddings, torch.Tensor):
        sentence_embeddings_np: np.ndarray = sentence_embeddings.cpu().numpy()
        return sentence_embeddings_np
    return sentence_embeddings


def get_corpus_embedding_hug(
    corpus: Sequence[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
) -> np.ndarray:
    for i in tqdm(range(math.ceil(len(corpus) / batch_size))):
        corpus_embeddings = embedding_sentences_hug(
            sentences=corpus[i * batch_size : (i + 1) * batch_size],
            model=model,
            tokenizer=tokenizer,
            convert_to_numpy=True,
        )
        if i == 0:
            corpus_embeddings_list = corpus_embeddings
        else:
            corpus_embeddings_list = np.concatenate(
                (corpus_embeddings_list, corpus_embeddings), axis=0
            )
    assert len(corpus_embeddings_list) == len(corpus)
    return corpus_embeddings_list


if __name__ == "__main__":
    model_name_or_path = "../hf_models/codellama/CodeLlama-7b-Instruct-hf"

    model, tokenizer = load_model_hug(model_name_or_path)
    responses = get_responses_hug(
        prompts=[
            "bubble sort algorithm in python\n```python\n",
            "select sort algorithm in python\n```python\n",
            "quick sort algorithm in python\n```python\n",
        ],
        model_name=model_name_or_path,
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        n=5,
        max_tokens=128,
    )
    assert len(responses) == 3
    for i in range(len(responses)):
        print(f"prompt[{i}]=\n")
        for id, r in enumerate(responses[i]):
            print(f"response[{id}]=\n{r}\n")
