from utils import CodeRAG, CodeRAG_bm25, get_unique_ordered, stream_json_all
import numpy as np
import pickle
import os
import json
from typing import *
import argparse

EMBED_PATH = "embed_results"
RETRIEVE_PATH = "retrieve_results"
TOP_K = 30
DATASETS = ["mbxp", "mceval", "multilingual_humaneval"]
LANGUAGES = [
    "python",
    "java",
    "javascript",
    "typescript",
    "kotlin",
    "ruby",
    "php",
    "cpp",
    "csharp",
    "go",
    "perl",
    "scala",
    "swift",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Retrieval")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        choices=[
            "dense_CodeRankEmbed",
            "dense_CodeRankEmbed_without_nl",
            "dense_bge_without_nl",
            "sparse_bm25_without_nl",
        ],
    )
    args = parser.parse_args()
    retrieval_mode: str = args.retrieval_mode

    retrieval_doc: Dict[str, List[str]] = {}

    if retrieval_mode.startswith("dense_"):
        # load query embeddings and queries
        query_embed_result_all: np.ndarray = None
        query_list_all: List[Dict] = []
        for lang in LANGUAGES:
            for dataset in DATASETS:
                query_embed_result_path = os.path.join(
                    EMBED_PATH,
                    retrieval_mode,
                    "query",
                    dataset,
                    f"{dataset}_correct_{lang}",
                )
                if not os.path.exists(query_embed_result_path + ".npy"):
                    continue
                query_embed_result: np.ndarray = np.load(
                    query_embed_result_path + ".npy"
                )
                query_list: List[Dict] = pickle.load(
                    open(query_embed_result_path + ".pkl", "rb")
                )
                for q in query_list:
                    if "dataset" not in q:
                        q["dataset"] = dataset
                    assert q["dataset"] == dataset
                assert len(query_embed_result) == len(query_list)
                if query_embed_result_all is None:
                    query_embed_result_all = query_embed_result
                else:
                    assert (
                        query_embed_result_all.shape[1] == query_embed_result.shape[1]
                    )
                    query_embed_result_all = np.concatenate(
                        (query_embed_result_all, query_embed_result), axis=0
                    )
                query_list_all.extend(query_list)
                assert len(query_embed_result_all) == len(query_list_all)
        print(f"len of queries: {len(query_list_all)}")

        # load corpus embeddings and corpus
        corpus_embed_result_all: np.ndarray = None
        corpus_list_all: List[Dict] = []
        for lang in LANGUAGES:
            for dataset in DATASETS:
                corpus_embed_result_path = os.path.join(
                    EMBED_PATH,
                    retrieval_mode,
                    "corpus",
                    dataset,
                    f"{dataset}_correct_{lang}",
                )
                if not os.path.exists(corpus_embed_result_path + ".npy"):
                    continue
                corpus_embed_result: np.ndarray = np.load(
                    corpus_embed_result_path + ".npy"
                )
                corpus_list: List[Dict] = pickle.load(
                    open(corpus_embed_result_path + ".pkl", "rb")
                )
                for c in corpus_list:
                    if "dataset" not in c:
                        c["dataset"] = dataset
                    assert c["dataset"] == dataset
                assert len(corpus_embed_result) == len(corpus_list)
                if corpus_embed_result_all is None:
                    corpus_embed_result_all = corpus_embed_result
                else:
                    assert (
                        corpus_embed_result_all.shape[1] == corpus_embed_result.shape[1]
                    )
                    corpus_embed_result_all = np.concatenate(
                        (corpus_embed_result_all, corpus_embed_result), axis=0
                    )
                corpus_list_all.extend(corpus_list)
                assert len(corpus_embed_result_all) == len(corpus_list_all)
        print(f"len of corpus: {len(corpus_list_all)}")

        # dense retrieve
        ref_problems_list = CodeRAG(
            corpus_embeddings=corpus_embed_result_all,
            query_embeddings=query_embed_result_all,
            corpus_infos=corpus_list_all,
            queries=query_list_all,
            top_k=TOP_K,
            remove_groundtruth=False,
            split_langs=False,
        )
        assert len(ref_problems_list) == len(query_list_all)

        # record golden docs per programming language
        for id in range(len(ref_problems_list)):
            query_key = (
                f'<{query_list_all[id]["dataset"]}><{query_list_all[id]["task_id"]}>'
            )
            if query_key not in retrieval_doc:
                retrieval_doc[query_key] = []
            ref_problems = ref_problems_list[id]
            for ref_problem in ref_problems:
                corpus_key = f'<{ref_problem["dataset"]}><{ref_problem["task_id"]}>'
                retrieval_doc[query_key].append(corpus_key)
    elif retrieval_mode.startswith("sparse_bm25"):
        # load queries and corpus
        problem_list: List[Dict] = []
        for dataset in DATASETS:
            for file in os.listdir(os.path.join("datasets", dataset)):
                if not file.endswith(".jsonl") or "_correct_" not in file:
                    continue
                problems = stream_json_all(os.path.join("datasets", dataset, file))
                assert isinstance(problems, list)
                for problem in problems:
                    if "dataset" not in problem:
                        problem["dataset"] = dataset
                    assert problem["dataset"] == dataset
                problem_list.extend(problems)
        print(f"len of problem: {len(problem_list)}")

        retrieval_doc = CodeRAG_bm25(
            corpus_infos=problem_list,
            queries=problem_list,
            top_k=TOP_K,
            without_nl=retrieval_mode.endswith("without_nl"),
        )
    else:
        raise NotImplementedError

    for g in retrieval_doc:
        retrieval_doc[g] = get_unique_ordered(retrieval_doc[g])

    save_dir = os.path.join(RETRIEVE_PATH, "RAG_cross_lang", retrieval_mode)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval_doc, f, ensure_ascii=False, indent=4)
