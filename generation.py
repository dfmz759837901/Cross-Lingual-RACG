import os
from utils import (
    load_model_vllm,
    get_responses_vllm,
    dicts_to_jsonl,
    stream_json_all,
    poison_code,
    is_changed_code,
    CodeRAG,
)
from typing import *
import argparse
from copy import deepcopy
import random
import re
import numpy as np
import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

DATASET_ROOT_PATH = "datasets"
FEW_SHOT_PROMPT_ROOT_PATH = os.path.join(
    "third_party", "mxeval", "data", "mbxp", "fewshot_prompts"
)
GENERATION_PATH = "generation_results"
EMBED_PATH = "embed_results"
POISION_PATH = "poisoning_results"


def get_result_filepath(
    model_name: str,
    lang: str,
    dataset: str,
    retrieval_mode: str,
    generation_mode: str,
    poison: Literal["logic", "syntax", "lexicon", "control_flow", "None"],
    poisoned_lang: Literal["None", "python", "java"],
) -> str:
    result_filename = f"result_{lang}.jsonl"
    if poison == "None":
        result_filename = os.path.join(
            GENERATION_PATH,
            dataset,
            retrieval_mode,
            generation_mode,
            model_name,
            result_filename,
        )
    else:
        result_filename = os.path.join(
            POISION_PATH,
            dataset,
            retrieval_mode,
            generation_mode,
            model_name,
            poison,
            poisoned_lang,
            result_filename,
        )
    os.makedirs(os.path.dirname(result_filename), exist_ok=True)
    return result_filename


def get_fewshot_prompts(lang: str) -> str:
    few_shot_prompts = None
    for file in os.listdir(FEW_SHOT_PROMPT_ROOT_PATH):
        if file.startswith(f"{lang}_fewshot"):
            file_path = os.path.join(FEW_SHOT_PROMPT_ROOT_PATH, file)
            with open(file_path, "r", encoding="utf-8") as f:
                few_shot_prompts = f.read()
            break
    if few_shot_prompts is None:
        raise FileNotFoundError(f"No few shot prompt found for {lang}")
    return few_shot_prompts.rstrip()


def create_prompt(
    problem: Dict[str, Any],
    lang: str,
    without_nl: bool,
    poison: Literal["logic", "syntax", "lexicon", "control_flow", "None"],
    poisoned_lang: Literal["None", "python", "java"],
    ref_problems: Dict[str, List[Dict[str, Any]]] = {},
    few_shot_text: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, Any]]]]:
    prompt = (
        "You will be given a question (problem specification) and will generate a correct program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program. "
        "Please do not output any additional comments, explanations, or text.\n\n"
    )
    if len(ref_problems) > 0:
        prompt += (
            "Here are some retrieved code snippets or documents that may be helpful.\n"
        )
        for ref_lang in ref_problems.keys():
            for ref_problem in ref_problems[ref_lang]:
                code_doc = (
                    deepcopy(ref_problem["code_without_comments"])
                    if without_nl
                    else deepcopy(ref_problem["canonical_solution"])
                )
                if ref_lang.lower() == poisoned_lang.lower() and poison != "None":
                    poisoned_code = poison_code(
                        poisoned=poison,
                        lang=ref_lang,
                        code=deepcopy(code_doc),
                        debug=False,
                    )
                    is_changed = is_changed_code(code_doc, poisoned_code)
                    code_doc = poisoned_code
                    ref_problem["is_changed"] = is_changed
                if not without_nl:
                    code_doc = f"{ref_problem['prompt']}\n{code_doc}"
                prompt += "\n" + code_doc + "\n\n"
    if few_shot_text is not None:
        prompt += f"Please complete the function using {lang} in the format of the example following."
        prompt += "\n" + few_shot_text + "\n"
    prompt += f"\n{problem['prompt']}\n"
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt, ref_problems


def generate_ref_ids(
    n: int, problem_len: int, x: int = None, seed: int = None
) -> List[int]:
    if n > problem_len:
        raise ValueError("n must be less than or equal to problem_len")
    all_integers = list(range(problem_len))
    random_integers = []
    if x is not None:
        if x < 0 or x >= problem_len:
            raise ValueError("x must be in the range [0, problem_len)")
        random_integers = [x]
        all_integers.remove(x)
        n -= 1
    if seed is not None:
        random.seed(seed)
    random_integers.extend(random.sample(all_integers, n))
    random.shuffle(random_integers)
    return random_integers


def infer_model(
    model_name_or_path: str,
    problem_crossLang: Dict[str, List[Dict]],
    dataset: str,
    generation_mode: str,
    poison: Literal["logic", "syntax", "lexicon", "control_flow", "None"],
    poisoned_lang: Literal["None", "python", "java"],
    retrieval_mode: str,
    retrieve_size: int,
    remove_groundtruth: bool,
    without_nl: bool,
    sample_n: int,
    temperature: float,
    debug: bool,
):
    assert retrieval_mode in [
        "random",
        "dense_bge",
        "sparse_bm25",
        "dense_CodeRankEmbed",
    ]
    assert remove_groundtruth in [True, False] and isinstance(remove_groundtruth, bool)
    assert without_nl in [True, False] and isinstance(without_nl, bool)
    if True == without_nl and "dense_CodeRankEmbed" == retrieval_mode:
        retrieval_mode = "dense_CodeRankEmbed_without_nl"
    assert poison in {"logic", "syntax", "lexicon", "control_flow", "None"}
    assert poisoned_lang in {"None", "python", "java"}
    assert poison == "None" or poisoned_lang != "None"

    if model_name_or_path.endswith("std"):
        model_name = "std"
    else:
        model_name = os.path.basename(model_name_or_path)
        model = None
        tokenizer = None

    print(
        f"Start inference {model_name} using mode -g {generation_mode} and -r {retrieval_mode} ..."
    )
    for lang in problem_crossLang:
        print(f"Infer {lang} data...")
        result_filename = get_result_filepath(
            model_name=model_name,
            lang=lang,
            dataset=dataset,
            retrieval_mode=retrieval_mode,
            generation_mode=generation_mode,
            poison=poison,
            poisoned_lang=poisoned_lang,
        )
        if os.path.exists(result_filename):
            print(f"{result_filename} already exists")
            result_data = stream_json_all(result_filename)
            if len(result_data) == len(problem_crossLang[lang]) * sample_n:
                if debug:
                    print(f"Result file is complete, skip inference.")
                    for r in result_data:
                        print(f"[task_id] {r['task_id']}")
                        if "ref_problems" in r:
                            print(f"[ref_problems]\n{r['ref_problems']}")
                        print(f"[input]\n{r['input']}")
                        print(f"[generation]\n{r['generation']}")
                        print()
                continue
            else:
                print(f"Result file is incomplete, start inference.")

        if not model_name_or_path.endswith("std"):
            if model is None or tokenizer is None:
                model, tokenizer, model_config = load_model_vllm(model_name_or_path)

        prompts: List[str] = []
        problem_len = len(problem_crossLang[lang])
        fewshot_prompt = get_fewshot_prompts(lang)
        stop_flag = fewshot_prompt[fewshot_prompt.rfind("\n") :]

        if retrieval_mode == "random":
            ref_problems_list: List[Dict[str, List[Dict[str, Any]]]] = []

            for id in range(problem_len):
                ref_problems: Dict[str, List[Dict[str, Any]]] = {}
                if generation_mode == "baseline_fewshot":
                    pass
                elif generation_mode.startswith("RAG_lang_"):
                    ref_lang = generation_mode[len("RAG_lang_") :]
                    ref_ids = generate_ref_ids(
                        n=retrieve_size, problem_len=problem_len, x=id, seed=42
                    )
                    ref_problems = {
                        ref_lang: [
                            problem_crossLang[ref_lang][ref_id] for ref_id in ref_ids
                        ]
                    }
                elif generation_mode == "RAG_cross_lang":
                    ref_langs = [
                        "cpp",
                        "java",
                        "python",
                    ] 
                    if lang in ref_langs:
                        ref_langs.remove(lang)  # remove return None
                        ref_langs.append("javascript")
                    ref_problems = {
                        ref_lang: [problem_crossLang[ref_lang][id]]
                        for ref_lang in ref_langs
                    }
                elif generation_mode.startswith("Poisoned_RAG_lang_"):
                    ref_lang = generation_mode[len("Poisoned_RAG_lang_") :]
                    ref_problems = {
                        ref_lang: [deepcopy(problem_crossLang[ref_lang][id])]
                    }
                    for ref_lang in ref_problems:
                        for ref_problem in ref_problems[ref_lang]:
                            ref_problem["canonical_solution"] = poison_code(
                                code=ref_problem["canonical_solution"],
                                poisoned="logic",
                                lang=ref_lang,
                            )
                else:
                    raise ValueError(
                        f"Unknown mode: {generation_mode} in {retrieval_mode} retrieval_mode"
                    )
                ref_problems_list.append(ref_problems)

            for id in range(problem_len):
                prompt, ref_problems_list[id] = create_prompt(
                    problem=problem_crossLang[lang][id],
                    lang=lang,
                    ref_problems=deepcopy(ref_problems_list[id]),
                    without_nl=without_nl,
                    poison=poison,
                    poisoned_lang=poisoned_lang,
                    few_shot_text=fewshot_prompt,
                )
                prompts.append(prompt)
        elif retrieval_mode.startswith("dense_CodeRankEmbed"):
            query_embed_result_path = os.path.join(
                EMBED_PATH,
                retrieval_mode,
                "query",
                dataset,
                f"{dataset}_correct_{lang}",
            )
            query_embed_result: np.ndarray = np.load(query_embed_result_path + ".npy")
            query_list: List[Dict] = pickle.load(
                open(query_embed_result_path + ".pkl", "rb")
            )
            assert len(query_embed_result) == len(query_list)
            assert len(query_list) == problem_len
            for id in range(problem_len):
                assert (
                    problem_crossLang[lang][id]["task_id"] == query_list[id]["task_id"]
                )

            corpus_embed_result_all: np.ndarray = None
            corpus_list_all: List[Dict] = []
            if generation_mode == "baseline_fewshot":
                pass
            elif generation_mode.startswith("RAG_lang_"):
                ref_lang: str = generation_mode[len("RAG_lang_") :]
                for ref_dataset in ["mceval", "multilingual_humaneval", "mbxp"]:
                    corpus_embed_result_path = os.path.join(
                        EMBED_PATH,
                        retrieval_mode,
                        "corpus",
                        ref_dataset,
                        f"{ref_dataset}_correct_{ref_lang}",
                    )
                    if not os.path.exists(corpus_embed_result_path + ".npy"):
                        continue
                    corpus_embed_result: np.ndarray = np.load(
                        corpus_embed_result_path + ".npy"
                    )
                    corpus_list = pickle.load(
                        open(corpus_embed_result_path + ".pkl", "rb")
                    )
                    assert len(corpus_embed_result) == len(corpus_list)
                    for c in corpus_list:
                        if "language" not in c:
                            c["language"] = ref_lang
                        assert c["language"] == ref_lang
                    if corpus_embed_result_all is None:
                        corpus_embed_result_all = corpus_embed_result
                    else:
                        assert (
                            corpus_embed_result_all.shape[1]
                            == corpus_embed_result.shape[1]
                        )
                        corpus_embed_result_all = np.concatenate(
                            (corpus_embed_result_all, corpus_embed_result), axis=0
                        )
                    corpus_list_all.extend(corpus_list)
                    assert len(corpus_embed_result_all) == len(corpus_list_all)
            elif generation_mode == "RAG_cross_lang":
                for ref_dataset in ["mceval", "multilingual_humaneval", "mbxp"]:
                    cur_root = os.path.join(
                        os.path.join(EMBED_PATH, retrieval_mode, "corpus", ref_dataset)
                    )
                    for ref_lang_file in os.listdir(cur_root):
                        if not ref_lang_file.endswith(".npy"):
                            continue
                        corpus_embed_result: np.ndarray = np.load(
                            os.path.join(cur_root, ref_lang_file)
                        )
                        corpus_list = pickle.load(
                            open(
                                os.path.join(cur_root, ref_lang_file).replace(
                                    ".npy", ".pkl"
                                ),
                                "rb",
                            )
                        )
                        assert len(corpus_embed_result) == len(corpus_list)
                        ref_lang: str = ref_lang_file.split(".")[0].split("_")[-1]
                        for c in corpus_list:
                            if "language" not in c:
                                c["language"] = ref_lang
                            assert c["language"] == ref_lang
                        if corpus_embed_result_all is None:
                            corpus_embed_result_all = corpus_embed_result
                        else:
                            assert (
                                corpus_embed_result_all.shape[1]
                                == corpus_embed_result.shape[1]
                            )
                            corpus_embed_result_all = np.concatenate(
                                (corpus_embed_result_all, corpus_embed_result), axis=0
                            )
                        corpus_list_all.extend(corpus_list)
                        assert len(corpus_embed_result_all) == len(corpus_list_all)
            elif generation_mode.startswith("Poisoned_RAG_lang_"):
                raise NotImplementedError(f"{generation_mode} not implemented")
            else:
                raise ValueError(
                    f"Unknown mode: {generation_mode} in {retrieval_mode} retrieval_mode"
                )

            ref_problems_list = [{} for _ in range(problem_len)]
            if len(corpus_list_all) > 0:
                ref_problems_list = CodeRAG(
                    corpus_embeddings=corpus_embed_result_all,
                    query_embeddings=query_embed_result,
                    corpus_infos=corpus_list_all,
                    queries=problem_crossLang[lang],
                    top_k=retrieve_size,
                    remove_groundtruth=remove_groundtruth,
                )
                assert len(ref_problems_list) == problem_len
            for query_id in range(problem_len):
                assert (
                    problem_crossLang[lang][query_id]["task_id"]
                    == query_list[query_id]["task_id"]
                )
                prompt, ref_problems_list[query_id] = create_prompt(
                    problem=problem_crossLang[lang][query_id],
                    lang=lang,
                    ref_problems=deepcopy(ref_problems_list[query_id]),
                    without_nl=without_nl,
                    poison=poison,
                    poisoned_lang=poisoned_lang,
                    few_shot_text=fewshot_prompt,
                )
                prompts.append(prompt)
        else:
            raise NotImplementedError(f"Unsupported retrieval_mode: {retrieval_mode}")

        prompts = [p.rstrip() + "\n" for p in prompts]

        if model_name_or_path.endswith("std"):
            completions: List[List[str]] = []
            for id in range(problem_len):
                completion: List[str] = []
                for _ in range(sample_n):
                    completion.append(problem_crossLang[lang][id]["canonical_solution"])
                completions.append(completion)
        else:
            completions: List[List[str]] = get_responses_vllm(
                prompts=prompts,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                n=sample_n,
                max_tokens=1024,
                stop=[stop_flag, "\ndef ", "/**", "\nimport", "\n\n\n", "```"],
            )

        assert len(completions) == problem_len and len(prompts) == problem_len
        result_data = []
        for i in range(problem_len):
            ref_problems = deepcopy(ref_problems_list[i])
            for ref_lang in ref_problems.keys():
                if poison != "None":
                    ref_problems[ref_lang] = [
                        {"id": r["task_id"], "is_changed": r.get("is_changed", None)}
                        for r in ref_problems[ref_lang]
                    ]
                else:
                    ref_problems[ref_lang] = [
                        r["task_id"] for r in ref_problems[ref_lang]
                    ]

            if debug:
                print(f"[task_id] {problem_crossLang[lang][i]['task_id']}")
                print(f"[ref_problems]\n{ref_problems}")
                print(f"[input]\n{prompts[i]}\n")
            for c in completions[i]:
                result_dict = deepcopy(problem_crossLang[lang][i])
                result_dict["ref_problems"] = ref_problems
                result_dict["input"] = prompts[i]
                result_dict["model"] = model_name
                c += stop_flag
                result_dict["generation"] = c
                if debug:
                    print(f"[generation]\n{c}\n")
                result_data.append(result_dict)

        print(f"\nSaved to {result_filename}")
        dicts_to_jsonl(result_data, result_filename, compress=False)


def main(
    model_name_or_path: str,
    dataset: str,
    retrieval_mode: str,
    retrieve_size: int,
    remove_groundtruth: bool,
    without_nl: bool,
    generation_mode: str,
    poison: Literal["logic", "syntax", "lexicon", "control_flow", "None"],
    poisoned_lang: Literal["None", "python", "java"],
    sample_n: int,
    temperature: float,
):
    dataset_path = os.path.join(DATASET_ROOT_PATH, dataset)
    problem_crossLang = {}
    for file in os.listdir(dataset_path):
        if file.endswith(".jsonl") and "_correct_" in file:
            assert file.startswith(dataset)
            lang = file.split("_")[-1].split(".")[0]
            problem_crossLang[lang] = stream_json_all(os.path.join(dataset_path, file))
    print(f"language_list = {problem_crossLang.keys()}")
    infer_model(
        model_name_or_path=model_name_or_path,
        problem_crossLang=problem_crossLang,
        dataset=dataset,
        retrieval_mode=retrieval_mode,
        retrieve_size=retrieve_size,
        remove_groundtruth=remove_groundtruth,
        without_nl=without_nl,
        generation_mode=generation_mode,
        sample_n=sample_n,
        temperature=temperature,
        poison=poison,
        poisoned_lang=poisoned_lang,
        debug=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Infer Generation")
    parser.add_argument("--model_name_or_path", type=str, default="std")

    parser.add_argument("--retrieval_mode", type=str, default="random")
    parser.add_argument("--retrieve_size", type=int, default=3)
    parser.add_argument("--remove_groundtruth", action="store_true", default=False)
    parser.add_argument("--without_nl", action="store_true", default=False)


    parser.add_argument("--generation_mode", type=str, default="baseline_fewshot")
    parser.add_argument("--sample_n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)

    parser.add_argument("--dataset", type=str, default="humaneval-x")

    parser.add_argument(
        "--poison",
        type=str,
        default="None",
        choices=["logic", "syntax", "lexicon", "control_flow", "None"],
    )
    parser.add_argument(
        "--poisoned_lang",
        type=str,
        default="None",
        choices=["None", "python", "java"],
    )

    args = parser.parse_args()
    print("\n" + "=" * 20)
    print(args)
    print("=" * 20)
    main(
        model_name_or_path=args.model_name_or_path,
        dataset=args.dataset,
        retrieval_mode=args.retrieval_mode,
        retrieve_size=args.retrieve_size,
        remove_groundtruth=args.remove_groundtruth,
        without_nl=args.without_nl,
        generation_mode=args.generation_mode,
        sample_n=args.sample_n,
        temperature=args.temperature,
        poison=args.poison,
        poisoned_lang=args.poisoned_lang,
    )
