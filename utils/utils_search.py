import faiss
import numpy as np
import math
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from typing import *


def search_hits(
    index: faiss.IndexFlatIP,
    query_embeddings: np.ndarray,
    top_k: int,
    batch_size: int = 4000,
) -> List[Dict]:
    hits = []
    for i in range(math.ceil(len(query_embeddings) / batch_size)):
        q_emb_matrix = query_embeddings[i * batch_size : (i + 1) * batch_size]
        res_dist, res_p_id = index.search(q_emb_matrix.astype("float32"), top_k)
        assert len(res_dist) == len(q_emb_matrix)
        assert len(res_p_id) == len(q_emb_matrix)

        for i in range(len(q_emb_matrix)):
            passages = []
            assert len(res_p_id[i]) == len(res_dist[i])
            for j in range(min(top_k, len(res_p_id[i]))):
                pid = res_p_id[i][j]
                score = res_dist[i][j]
                passages.append({"corpus_id": int(pid), "score": float(score)})
            hits.append(passages)
    return hits


def build_engine(corpus_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using the provided corpus embeddings.

    Args:
        corpus_embeddings (np.ndarray): A numpy array containing the embeddings of the corpus.

    Returns:
        faiss.IndexFlatIP: A FAISS index built using the inner product metric.
    """
    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(corpus_embeddings.astype("float32"))
    return index


def CodeRAG(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    corpus_infos: List[Dict],
    queries: List[Dict],
    top_k: int,
    batch_size: int = 4000,
    remove_groundtruth: bool = True,
    split_langs: bool = True,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    index: faiss.IndexFlatIP = build_engine(corpus_embeddings)
    assert len(corpus_infos) == index.ntotal
    assert top_k + 3 <= len(corpus_infos)
    hits = search_hits(
        index, query_embeddings, top_k + 3, batch_size
    )  # +3 for the ground truth
    assert len(hits) == len(queries)

    ref_problems_list: List[Dict[str, List[Dict[str, Any]]]] = []
    for query_id in range(len(hits)):
        if split_langs:
            ref_problems: Dict[str, List[Dict[str, Any]]] = {}
        else:
            ref_problems: List[Dict[str, Any]] = []
        cur_total = 0
        for p_dict in hits[query_id]:
            id = p_dict["corpus_id"]
            assert id < len(corpus_infos) and id >= 0
            assert "language" in corpus_infos[id]
            if remove_groundtruth:
                if corpus_infos[id]["task_id"] == queries[query_id]["task_id"]:
                    continue

            if split_langs:
                if corpus_infos[id]["language"] not in ref_problems:
                    ref_problems[corpus_infos[id]["language"]] = []
                ref_problems[corpus_infos[id]["language"]].append(corpus_infos[id])
            else:
                ref_problems.append(corpus_infos[id])

            cur_total += 1
            if cur_total >= top_k:
                break
        assert cur_total == top_k, f"{cur_total} != {top_k}"
        ref_problems_list.append(ref_problems)
    return ref_problems_list


def build_bm25(corpus: List[str]) -> BM25Okapi:
    tokenized_corpus = [d.lower().split() for d in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def hit_bm25(l: List[float], top_k: int) -> List[int]:
    indexed_list = list(enumerate(l))
    indexed_list.sort(key=lambda x: x[1], reverse=True)
    top_k_indices = [index for index, _ in indexed_list[:top_k]]
    return top_k_indices


def CodeRAG_bm25(
    corpus_infos: List[Dict[str, str]],
    queries: List[Dict[str, str]],
    top_k: int,
    without_nl: bool,
) -> List[Dict]:
    top_k = min(top_k, len(corpus_infos))

    corpus: List = []
    for d in corpus_infos:
        example = (
            f"{d['prompt']}\n{d['canonical_solution']}"
            if False == without_nl
            else d["code_without_comments"]
        )
        corpus.append(example)
    bm25 = build_bm25(corpus)

    result_dict: Dict[str, List[str]] = {}
    for query in tqdm(queries):
        tokenized_query = query["prompt"].lower().split()
        scores = bm25.get_scores(tokenized_query)
        assert len(scores) == len(corpus_infos)

        query_key = f'<{query["dataset"]}><{query["task_id"]}>'
        result_dict[query_key] = []
        top_k_indices = hit_bm25(scores, top_k)
        for id in top_k_indices:
            corpus_key = (
                f'<{corpus_infos[id]["dataset"]}><{corpus_infos[id]["task_id"]}>'
            )
            result_dict[query_key].append(corpus_key)

    return result_dict
