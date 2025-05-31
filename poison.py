from utils import poison_code, stream_json_all, dicts_to_jsonl, is_changed_code
from typing import *
from tqdm.auto import tqdm
import os

DATASET_PATH = "datasets"
#                     datasets/{dataset}/f"{dataset}_correct_{lang}.jsonl"
POISON_PATH = "poisoned_results"
# poisoned_results/{dataset}/f"{dataset}_poisoned_{poisoned}_{lang}.jsonl"


def batch_poison_save(
    dataset: str,
    lang: Literal["python", "java"],
    poisoned: Literal["logic", "syntax", "lexicon", "control_flow"],
    debug: bool = False,
):
    assert poisoned in ["logic", "syntax", "lexicon", "control_flow"]
    assert lang in ["python", "java"]

    jsonl_filename = os.path.join(
        DATASET_PATH, dataset, f"{dataset}_correct_{lang}.jsonl"
    )
    poisoned_path = os.path.join(
        POISON_PATH, dataset, f"{dataset}_poisoned_{poisoned}_{lang}.jsonl"
    )
    os.makedirs(os.path.dirname(poisoned_path), exist_ok=True)

    datapoints: List[Dict] = stream_json_all(jsonl_filename)
    total_poisoned = 0
    for d in tqdm(datapoints):
        assert "canonical_solution" in d
        if debug:
            print(f"\n[{dataset}][correct][None][{lang}]")
            print(d["canonical_solution"])
        try:
            poisoned_code = poison_code(
                code=d["canonical_solution"], poisoned=poisoned, lang=lang, debug=False
            )
        except:
            raise Exception(
                f"[task_id]\n{d['task_id']}\n[canonical_solution]\n{d['canonical_solution']}"
            )
        is_changed = is_changed_code(d["canonical_solution"], poisoned_code)
        d["canonical_solution"] = poisoned_code
        d["poisoned"] = poisoned
        d["is_changed"] = is_changed
        if is_changed:
            total_poisoned += 1
        if debug:
            print(
                f"\n[{dataset}][poisoned][{poisoned}][{lang}][is_changed={is_changed}]"
            )
            print(d["canonical_solution"])
    print(f"Dataset = {dataset}, Lang = {lang}, Poisoned = {poisoned}")
    print(
        f"Rate = {total_poisoned}/{len(datapoints)} = {round(total_poisoned/len(datapoints)*100,2)} %"
    )
    dicts_to_jsonl(datapoints, poisoned_path, compress=False)


if __name__ == "__main__":
    for dataset in ["humaneval-x", "mbxp", "mceval", "multilingual_humaneval"]:
        for lang in ["python", "java"]:
            for poisoned in ["logic", "syntax", "lexicon", "control_flow"]:
                batch_poison_save(dataset, lang, poisoned, debug=True)
