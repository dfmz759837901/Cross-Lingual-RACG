import os
from utils import stream_json_all, get_messages_api, display_dict, dicts_to_jsonl
from typing import *
from tqdm.auto import tqdm
import re
import sys

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
DATASET_ROOT_PATH = "datasets"
FEW_SHOT_PROMPT_ROOT_PATH = os.path.join(
    "third_party", "mxeval", "data", "mbxp", "fewshot_prompts"
)
DEBUG = False


def confirm(data_correct: List[Dict[str, Any]], lang: str):
    for problem in tqdm(data_correct):
        try:
            k = (
                50 if lang in ["perl", "go"] else 5
            )  # Hash table in perl is unordered, Map in Go is unordered
            correct = False
            for _ in range(k):
                exec_result = CHECK_CORRECTNESS_FUNCTION_MAP[lang](
                    problem, problem["canonical_solution"], timeout=30.0
                )
                if exec_result["passed"] == True:
                    correct = True
                    break
            assert correct == True
        except:
            print("\n" + "-" * 20)
            display_dict(problem)
            print(f"[exec_result]\n{exec_result}")
            print("-" * 20 + "\n")
            raise AssertionError


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
    return few_shot_prompts


def find_ref_problems(
    problem: Dict[str, Any], data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    ref_problems = {}
    for ref_lang in data.keys():
        for ref_problem in data[ref_lang]:
            if (
                ref_problem["task_id"].split("/")[-1]
                == problem["task_id"].split("/")[-1]
            ):
                ref_problems[ref_lang] = ref_problem
                break
    return ref_problems


def create_messages(
    problem: Dict[str, Any],
    lang: str,
    ref_problems: Optional[Dict[str, Dict[str, Any]]] = None,
    few_shot_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    prompt = (
        "You will be given a question (problem specification) and will generate a correct program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program. "
        "Please do not output any additional comments, explanations, or text.\n\n"
    )
    if ref_problems is not None:
        prompt += "Here is some retrieved examples that may be helpful.\n"
        for ref_lang in ref_problems.keys():
            ref_problem = ref_problems[ref_lang]
            prompt += (
                f"\n{ref_problem['prompt']}\n{ref_problem['canonical_solution']}\n\n"
            )
    if few_shot_text is not None:
        prompt += (
            f"Please complete the function using {lang} in the format of the example following. "
            f"Please continue writing and just provide the function implementation using {lang}. "
            "Please complete the function *WITHOUT* repeating function declaration and comments.\n"
        )
        prompt += "\n" + few_shot_text + "\n"
    prompt += f"\n{problem['prompt']}\n"
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    messages = [{"role": "user", "content": prompt}]
    return messages


def deal_response(response: str, ref_str: str, lang: str) -> str:
    start_flag = False
    response_dealed = ""
    for line in response.splitlines():
        line = line.rstrip()
        if line.startswith("```"):
            continue
        if line == "":
            continue
        if line not in ref_str:
            start_flag = True
        if start_flag:
            response_dealed += line + "\n"

    if lang == "csharp":
        if "\n    }\n}" in response_dealed:
            response_dealed = response_dealed.replace("\n    }\n}", "")
    if lang == "scala":
        if "\n}" in response_dealed:
            response_dealed = response_dealed.replace("\n}", "")

    return "\n" + response_dealed


if __name__ == "__main__":
    dataset_list = ["mbxp", "mceval", "multilingual_humaneval", "humaneval-x"]
    dataset_list = ["mbxp", "mceval", "multilingual_humaneval"]
    total_data = 0
    total_correct_data = 0
    for lang in CHECK_CORRECTNESS_FUNCTION_MAP.keys():
        print("-" * 20 + f"\nLANG = {lang}")
        total_lang_data = 0
        total_lang_correct_data = 0
        few_shot_text = get_fewshot_prompts(lang)
        for dataset_name in dataset_list:
            print(f"DATASET = {dataset_name}")
            data_path = os.path.join(
                DATASET_ROOT_PATH, dataset_name, f"{dataset_name}_{lang}.jsonl"
            )
            data_correct_path = os.path.join(
                DATASET_ROOT_PATH, dataset_name, f"{dataset_name}_correct_{lang}.jsonl"
            )
            data = []
            data_correct = []
            if os.path.exists(data_path):
                data = stream_json_all(data_path)
            if os.path.exists(data_correct_path):
                data_correct = stream_json_all(data_correct_path)
            print(f"len(data) = {len(data)}")
            print(f"len(data_correct) = {len(data_correct)}")
            ref_langs = ["python", "java"]
            ref_datas = {}
            for ref_lang in ref_langs:
                ref_data_correct_path = os.path.join(
                    DATASET_ROOT_PATH,
                    dataset_name,
                    f"{dataset_name}_correct_{ref_lang}.jsonl",
                )
                ref_datas[ref_lang] = stream_json_all(ref_data_correct_path)

            if lang not in [
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
            ]:
                confirm(data_correct, lang)

            # if lang in CHECK_CORRECTNESS_FUNCTION_MAP.keys():
            if lang in []:
                print("Enhancing dataset...")
                for problem in tqdm(data):
                    if problem in data_correct:
                        continue

                    # if lang == "python":
                    #     problem["prompt"] = problem["prompt"].replace("\t", "    ")
                    #     problem["test"] = problem["test"].replace("\t", "    ")
                    # if lang == "python" and problem["entry_point"] == "check":
                    #     problem["entry_point"] = "check_py"
                    #     problem["prompt"] = problem["prompt"].replace(
                    #         "check(", "check_py("
                    #     )

                    ref_problems = find_ref_problems(problem, ref_datas)
                    messages = create_messages(
                        problem=problem,
                        lang=lang,
                        ref_problems=ref_problems,
                        few_shot_text=few_shot_text,
                    )
                    if DEBUG:
                        print("\n```prompt\n" + messages[0]["content"] + "\n```")
                    try:
                        responses = get_messages_api(
                            messages=messages,
                            temperature=0.8,
                            n=10,
                        )
                    except Exception as e:
                        print(f"Exception: {e}")
                        continue
                    for response in responses:
                        response = deal_response(
                            response=response, ref_str=messages[0]["content"], lang=lang
                        )
                        exec_result = CHECK_CORRECTNESS_FUNCTION_MAP[lang](
                            problem, response, timeout=30.0
                        )
                        if exec_result["passed"] == True:
                            problem["canonical_solution"] = response
                            data_correct.append(problem)
                            print(f"\n{problem['task_id']} passed!")
                            break
                    if problem not in data_correct:
                        print(
                            f"\n```[prompt]\n{problem['prompt']}\n[response]{response}\n[exec_result]\n{exec_result['result']}\n```"
                        )

                print(f"new len(data) = {len(data)}")
                print(f"new len(data_correct) = {len(data_correct)}")

            total_lang_data += len(data)
            total_lang_correct_data += len(data_correct)
            if len(data_correct) > 0:
                dicts_to_jsonl(data_correct, data_correct_path, compress=False)
            if len(data) > 0:
                dicts_to_jsonl(data, data_path, compress=False)

        print(f"total_lang_data = {total_lang_data}")
        print(f"total_lang_correct_data = {total_lang_correct_data}")
        total_data += total_lang_data
        total_correct_data += total_lang_correct_data
    print(f"total_data = {total_data}")
    print(f"total_correct_data = {total_correct_data}")
