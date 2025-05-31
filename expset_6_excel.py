from utils import stream_json_all, calc_precision, calc_recall, get_unique_ordered
from tqdm.auto import tqdm
import os
from typing import *

GOLDEN_PATH = "retrieve_results/golden_doc.json"
RETRIEVE_PATH = "retrieve_results/RAG_cross_lang"
RETRIEVE_MODES = [
    "dense_CodeRankEmbed",
    "sparse_bm25_without_nl",
    "dense_bge_without_nl",
    "dense_CodeRankEmbed_without_nl",
]


def calc_retrieve(
    golden_doc: Dict[str, List[str]],
    retrieve_doc: Dict[str, List[str]],
    Top_K: Union[int, List[int]] = 10,
) -> Tuple[float, float]:

    if isinstance(Top_K, int):
        Top_K = [Top_K]

    for top_k in Top_K:
        precison_all = 0
        recall_all = 0
        for q_key in tqdm(retrieve_doc.keys()):
            if q_key not in golden_doc:
                raise ValueError(f"Query {q_key} not in golden doc")
            golden_doc[q_key] = list(set(golden_doc[q_key]))
            retrieve_doc[q_key] = get_unique_ordered(retrieve_doc[q_key])
            try:
                precison = calc_precision(
                    golden_doc[q_key], retrieve_doc[q_key][:top_k], top_k
                )
                recall = calc_recall(
                    golden_doc[q_key], retrieve_doc[q_key][:top_k], top_k
                )
            except:
                print(
                    f"Query {q_key} wrong, len = {len(retrieve_doc[q_key])}, {retrieve_doc[q_key]}"
                )
                continue
            precison_all += precison
            recall_all += recall
        print(f"Precison@{top_k} = {round(precison_all / len(retrieve_doc) * 100, 2)}%")
        print(f"Recall@{top_k} = {round(recall_all / len(retrieve_doc) * 100, 2)}%\n")


if __name__ == "__main__":

    golden_doc = stream_json_all(GOLDEN_PATH)

    for retrieve_mode in RETRIEVE_MODES:
        retrieve_doc_path = os.path.join(RETRIEVE_PATH, retrieve_mode, "result.json")
        try:
            retrieve_doc = stream_json_all(retrieve_doc_path)
        except:
            continue
        print(f"Calc {retrieve_mode}")
        calc_retrieve(golden_doc, retrieve_doc, Top_K=[1, 3, 5, 10, 20])
