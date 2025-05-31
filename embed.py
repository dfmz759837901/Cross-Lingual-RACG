import os

os.environ["HF_HOME"] = ".cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = ".cache/huggingface"
os.environ["HF_TOKEN"] = "your TOKEN"

from typing import *
import pickle
import re
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer
from utils import (
    load_model_hug,
    get_corpus_embedding_hug,
    stream_json_all,
)

#                                      datasets/{dataset}/f"{dataset}_{correct/poisoned/}_{lang}.{jsonl}"
# embed_results/{dense_bge/dense_CodeRankEmbed}/{dataset}/f"{dataset}_{correct/poisoned/}_{lang}.{npy/pkl}"
DATASET_PATH = "datasets"
EMBED_PATH = "embed_results"


def embed(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    retrieval_mode: Literal["dense_bge", "dense_CodeRankEmbed"],
    without_nl: bool,
    query_prefix: str,
    poisoned: Literal["logic", "syntax", "lexicon", "control_flow", "None"],
    batch_size: int,
):
    assert retrieval_mode in ["dense_bge", "dense_CodeRankEmbed"]
    assert poisoned in ["logic", "syntax", "lexicon", "control_flow", "None"]
    assert without_nl in [True, False] and isinstance(without_nl, bool)
    if True == without_nl and retrieval_mode.startswith("dense_"):
        retrieval_mode += "_without_nl"

    for dataset in os.listdir(DATASET_PATH):
        if dataset not in ["mbxp", "mceval", "multilingual_humaneval"]:
            continue
        for jsonl_filename in os.listdir(os.path.join(DATASET_PATH, dataset)):
            if not jsonl_filename.endswith(".jsonl"):
                continue
            if not jsonl_filename.startswith(dataset):
                continue
            if not "_correct_" in jsonl_filename:
                continue
            jsonl_filepath = os.path.join(DATASET_PATH, dataset, jsonl_filename)
            lang = jsonl_filename.split(".")[0].split("_")[-1]
            for data_type in ["query", "corpus"]:
                print(f"Embed {jsonl_filepath} for {data_type} {dataset} {lang} ...")
                npy_filepath = os.path.join(
                    EMBED_PATH,
                    retrieval_mode,
                    data_type,
                    dataset,
                    jsonl_filename.replace(".jsonl", ".npy"),
                )
                pkl_filepath = os.path.join(
                    EMBED_PATH,
                    retrieval_mode,
                    data_type,
                    dataset,
                    jsonl_filename.replace(".jsonl", ".pkl"),
                )
                os.makedirs(os.path.dirname(npy_filepath), exist_ok=True)
                if os.path.exists(npy_filepath) and os.path.exists(pkl_filepath):
                    embeddings = np.load(npy_filepath)
                    print(f"Shape = {embeddings.shape}")
                    datapoints = pickle.load(open(pkl_filepath, "rb"))
                    for d in datapoints:
                        if data_type == "corpus":
                            example = (
                                f"{d['prompt']}\n{d['canonical_solution']}"
                                if False == without_nl
                                else d["code_without_comments"]
                            )
                        elif data_type == "query":
                            example = query_prefix + d["prompt"]
                        example = re.sub(r"\n{2,}", "\n", example)
                        print(f"Example: {data_type}[0]=\n{example}\n")
                        break
                    continue

                datapoints: List[Dict] = stream_json_all(jsonl_filepath)

                corpus = []
                if poisoned == "None":
                    for d in datapoints:
                        if data_type == "corpus":
                            example = (
                                f"{d['prompt']}\n{d['canonical_solution']}"
                                if False == without_nl
                                else d["code_without_comments"]
                            )
                        elif data_type == "query":
                            example = query_prefix + d["prompt"]
                        example = re.sub(r"\n{2,}", "\n", example)
                        corpus.append(example)
                else:
                    raise NotImplementedError(f"poisoned: {poisoned}")
                assert len(corpus) == len(datapoints)
                embeddings = get_corpus_embedding_hug(
                    corpus=corpus,
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=batch_size,
                )
                assert len(corpus) == len(embeddings)
                print(f"Embeddings Shape = {embeddings.shape}")
                print(f"Example: {data_type}[0]=\n{corpus[0]}\n")
                np.save(npy_filepath, embeddings)
                with open(pkl_filepath, "wb") as f:
                    pickle.dump(datapoints, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="dense_CodeRankEmbed",
        choices=["random", "dense_bge", "sparse_bm25", "dense_CodeRankEmbed"],
    )
    parser.add_argument("--without_nl", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--poison",
        type=str,
        default="None",
        choices=["logic", "syntax", "lexicon", "control_flow", "None"],
    )
    args = parser.parse_args()
    print(args)

    model_name_or_path = None
    query_prefix = ""
    if args.retrieval_mode == "dense_CodeRankEmbed":
        model_name_or_path = "your path"
        query_prefix = "Represent this query for searching relevant code:\n"
    elif args.retrieval_mode == "dense_bge":
        model_name_or_path = "../hf_models/BAAI/bge-large-en-v1.5"
        query_prefix = "Represent this sentence for searching relevant passages:\n"
    else:
        raise NotImplementedError(f"retrieval_mode: {args.retrieval_mode}")

    model, tokenizer, _ = load_model_hug(
        model_name_or_path=model_name_or_path, task="encode"
    )
    embed(
        model=model,
        tokenizer=tokenizer,
        retrieval_mode=args.retrieval_mode,
        query_prefix=query_prefix,
        without_nl=args.without_nl,
        poisoned=args.poison,
        batch_size=args.batch_size,
    )

    # queries = ['Represent this query for searching relevant code: Calculate the n-th factorial\n\ndef fact(n):\n']
    # codes = ['\n if n < 0:\n  raise ValueError\n return 1 if n == 0 else n * fact(n - 1)', '\n if n < 0:\n  raise ValueError\n return 1 if n == 0 else n + fact(n - 1)']
    # queries_embeddings = get_corpus_embedding_hug(corpus=queries, model=model, tokenizer=tokenizer, batch_size=args.batch_size)
    # codes_embeddings = get_corpus_embedding_hug(corpus=codes, model=model, tokenizer=tokenizer, batch_size=args.batch_size)
    # print(queries_embeddings.shape)
    # print(codes_embeddings.shape)
    # print(queries_embeddings @ codes_embeddings.T)
