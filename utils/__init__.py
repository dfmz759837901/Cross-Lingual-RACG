__version__ = "0.0"


def _init_package():
    """
    Initialize the package.
    """
    print("Import utils of Cross-Lingual-RACG")


_init_package()

from utils.utils_json import dicts_to_jsonl, stream_json_all, display_dict
from utils.utils_infer_vllm import load_model_vllm, get_responses_vllm
from utils.utils_infer_hug import (
    load_model_hug,
    get_responses_hug,
    get_corpus_embedding_hug,
)
from utils.utils_infer_api import get_messages_api
from utils.utils_metric import (
    estimate_pass_at_k,
    count_pass_at_k,
    calc_precision,
    calc_recall,
    get_unique_ordered,
)
from utils.utils_eval import cleanup_code, process_test
from utils.utils_excel import (
    write_dict_to_excel,
    write_pd_to_excel,
    write_pdlist_to_excel,
    init_sheet,
)
from utils.utils_poison import poison_code, is_changed_code
from utils.utils_search import CodeRAG, CodeRAG_bm25
from utils.utils_comments import remove_comments
from utils.utils_venn import draw_venn

__all__ = [
    "stream_json_all",
    "dicts_to_jsonl",
    "display_dict",
    "load_model_vllm",
    "get_responses_vllm",
    "get_messages_api",
    "load_model_hug",
    "get_responses_hug",
    "get_corpus_embedding_hug",
    "estimate_pass_at_k",
    "count_pass_at_k",
    "calc_precision",
    "calc_recall",
    "get_unique_ordered",
    "check_correctness",
    "cleanup_code",
    "process_test",
    "write_dict_to_excel",
    "write_pd_to_excel",
    "write_pdlist_to_excel",
    "init_sheet",
    "poison_code",
    "is_changed_code",
    "CodeRAG",
    "CodeRAG_bm25",
    "remove_comments",
    "draw_venn",
]
