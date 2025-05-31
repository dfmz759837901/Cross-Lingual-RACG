from typing import *
from tqdm.auto import tqdm
import os
import re
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from utils import stream_json_all, dicts_to_jsonl, count_pass_at_k

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from mxeval.execution import (
    check_correctness,
    check_correctness_cpp,
    check_correctness_csharp,
    check_correctness_go,
    check_correctness_java,
    check_correctness_javascript,
    check_correctness_kotlin,
    check_correctness_perl,
    check_correctness_php,
    check_correctness_ruby,
    check_correctness_scala,
    check_correctness_swift,
    check_correctness_typescript,
)

CHECK_CORRECTNESS_FUNCTION_MAP = {
    "python": check_correctness,
    "java": check_correctness_java,
    "javascript": check_correctness_javascript,
    "typescript": check_correctness_typescript,
    "kotlin": check_correctness_kotlin,
    "ruby": check_correctness_ruby,
    "php": check_correctness_php,
    "cpp": check_correctness_cpp,
    "csharp": check_correctness_csharp,
    "go": check_correctness_go,
    "perl": check_correctness_perl,
    "scala": check_correctness_scala,
    "swift": check_correctness_swift,
}

GENERATION_PATH = "generation_results"
EVAL_PATH = "eval_results"
POISION_PATH = "poisoning_results"
DEBUG = True


def task(result: Dict, lang: str, k: int) -> Dict:
    try:
        for _ in range(k):
            exec_result = CHECK_CORRECTNESS_FUNCTION_MAP[lang](
                result, result["generation"], timeout=20.0
            )
            result["passed"] = exec_result["passed"]
            result["result"] = exec_result["result"]
            if result["passed"] == True:
                if len(result["result"]) > 0 and result["result"] != "passed":
                    print("=" * 20)
                    print(result["task_id"])
                    print(result["result"])
                result["result"] = "passed"
                break
        return result
    except Exception as e:
        raise Exception(f"[Language] {lang}\n[Exception]\n{e}\n[result]\n{result}")


def process_tasks_concurrently(
    result_data: List[Dict], task: Callable, lang: str, k: int, max_workers: int = 1
) -> List[Dict]:
    result_list: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, result, lang, k) for result in result_data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result_list.append(future.result())
        wait(futures)
    assert len(result_data) == len(result_list)
    return result_list


def handle_generation(generation: str, lang: str, dataset: str) -> str:
    if "\n/**" in generation:
        generation = generation.split("\n/**", 1)[0]

    if lang == "python":
        stop_words = ["\ndef ", "\ns\n", "\n\n        return True"]
        for stop_word in stop_words:
            generation = generation.split(stop_word, 1)[0]
        firstline = "    return"
        for line in generation.splitlines():
            line = line.rstrip()
            if len(line) > 0:
                firstline = line
                break
        indent = firstline[: len(firstline) - len(firstline.lstrip())]
        newgeneration = []
        for line in generation.splitlines():
            line = line.rstrip()
            if len(line) > 0:
                newgeneration.append(line)
            if line.startswith(indent + "return") or line.startswith("\treturn"):
                break
        generation = "\n".join(newgeneration)

    if lang == "java":
        stop_words = ["\nimport ", "\nclass "]
        for stop_word in stop_words:
            generation = generation.split(stop_word, 1)[0]
        if "\n}" in generation:
            generation = generation[: generation.find("\n}")].rstrip() + "\n}"

    if lang == "php":
        stop_words = ["?>", "</s>"]
        for stop_word in stop_words:
            generation = generation.split(stop_word, 1)[0]
        if "\n}" in generation:
            generation = generation[: generation.find("\n}")].rstrip() + "\n}"

    if lang == "scala":
        if "\n}" in generation:
            generation = generation[: generation.find("\n}")].rstrip() + "\n}"
        if "object Main" in generation:
            generation = generation[: generation.find("object Main")].rstrip()

    if dataset == "mceval":
        if lang == "php":
            if not generation.lstrip().startswith("{"):
                generation = "{\n" + generation
        if lang == "java" or lang == "scala":
            if not generation.lstrip().startswith("{"):
                generation = "  {\n" + generation
            if generation.count("}") > generation.count("{"):
                generation = generation[: generation.rfind("}")].rstrip()
        if lang == "perl":
            while generation.count("}") < generation.count("{") + 1:
                if not generation.endswith("\n"):
                    generation += "\n"
                generation = generation + "}"

    if dataset == "multilingual_humaneval" or dataset == "mbxp":
        if lang == "scala":
            if generation.count("}") > generation.count("{") + 1:
                generation = generation[: generation.rfind("}")].rstrip()

    generation = "\n" + generation + "\n"
    generation = re.sub(r"\n{2,}", "\n", generation)
    return generation


def main(
    model_name: str,
    dataset: str,
    retrieval_mode: str,
    generation_mode: str,
    poison: Literal["logic", "syntax", "lexicon", "control_flow", "None"],
    poisoned_lang: Literal["None", "python", "java"],
    sample_n: int,
    debug: bool,
):
    if poison == "None":
        generation_path = os.path.join(
            GENERATION_PATH, dataset, retrieval_mode, generation_mode, model_name
        )
    else:
        generation_path = os.path.join(
            POISION_PATH,
            dataset,
            retrieval_mode,
            generation_mode,
            model_name,
            poison,
            poisoned_lang,
        )
    if not os.path.exists(generation_path):
        return

    for file in os.listdir(generation_path):
        if file.endswith(".jsonl"):
            assert file.startswith("result_")
            lang = file.split(".")[0][len("result_") :]
            if lang not in CHECK_CORRECTNESS_FUNCTION_MAP:
                print(f"======= Unsupported language: {lang} =======")
                continue
            file_path = os.path.join(generation_path, file)
            if poison == "None":
                eval_path = file_path.replace(GENERATION_PATH, EVAL_PATH)
            else:
                eval_path = file_path.replace(POISION_PATH, EVAL_PATH)

            print(f"Processing {file_path}...")
            if not os.path.exists(eval_path):
                result_data = stream_json_all(file_path)
                for result in result_data:
                    result["generation"] = handle_generation(
                        result["generation"], lang, dataset
                    )

                try:
                    max_workers = max(os.cpu_count() // 4 - 1, 1)
                    if dataset in {"mceval", "humaneval-x"} and lang in {"go"}:
                        max_workers = 1
                    result_list = process_tasks_concurrently(
                        result_data=result_data,
                        task=task,
                        lang=lang,
                        k=5 if lang in ["perl", "go"] else 1,
                        # Hash table in perl is unordered, Map in Go is unordered
                        max_workers=max_workers,
                        # CPU cores
                    )
                except KeyboardInterrupt:
                    break
                except:
                    time.sleep(5)
                    result_list = process_tasks_concurrently(
                        result_data=result_data,
                        task=task,
                        lang=lang,
                        k=5 if lang in ["perl", "go"] else 1,
                        # Hash table in perl is unordered, Map in Go is unordered
                        max_workers=1,
                        # CPU cores
                    )

                assert len(result_list) == len(
                    result_data
                ), f"{lang} {len(result_list)} != {len(result_data)}"
                result_data = result_list
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                dicts_to_jsonl(result_data, eval_path, compress=False)
            else:
                if debug:
                    result_data = stream_json_all(eval_path)

            if debug:
                for r in result_data:
                    print("\n[task_id]")
                    print(r["task_id"])
                    print("[passed]")
                    print(r["passed"])
                    print("[result]")
                    print(r["result"])
                    print("[code]")
                    display_code = r["prompt"] + r["generation"] + r["test"]
                    display_code = re.sub(r"\n{2,}", "\n", display_code)
                    print(display_code)
                print("=======================================")
                pass_at_k, len_data = count_pass_at_k(result_data, pass_k=sample_n)
                print(
                    f"Eval Completed {file_path}\n{dataset} {lang} pass@{sample_n} = {round(pass_at_k * 100, 2)}, len = {len_data}\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Generation Eval")
    parser.add_argument("--model_name_or_path", type=str, default="std")
    parser.add_argument("--retrieval_mode", type=str, default="random")
    parser.add_argument("--generation_mode", type=str, default="baseline_fewshot")
    parser.add_argument("--sample_n", type=int, default=5)
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

    model_name = os.path.basename(args.model_name_or_path)
    main(
        model_name=model_name,
        dataset=args.dataset,
        retrieval_mode=args.retrieval_mode,
        generation_mode=args.generation_mode,
        sample_n=args.sample_n,
        poison=args.poison,
        poisoned_lang=args.poisoned_lang,
        debug=DEBUG,
    )
