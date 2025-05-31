from typing import *


def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
):
    """
    Cleans up the generated code.
    """
    if language_type is None or dataset is None:
        return code
    dataset = dataset.lower()
    language_type = language_type.lower()
    if "humaneval" in dataset or "mbxp" == dataset:
        if language_type == "python":
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
            for w in end_words:
                if w in code:
                    code = code[: code.rfind(w)]
        elif language_type == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + "}"
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
            if "\n}" in code:
                code = code[: code.rfind("\n}")] + "\n}"
            if code.count("{") + 1 == code.count("}"):
                code += "\n}"
        elif language_type == "go":
            end_words = ["\n//", "\nfunc main("]
            for w in end_words:
                if w in code:
                    code = code[: code.rfind(w)]
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
            if "\n}" in code:
                code = code[: code.rfind("\n}")] + "\n}"
        elif language_type == "cpp":
            main_pos = code.find("\nint main")
            if main_pos != -1:
                code = code[:main_pos]
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
            if "\n}" in code:
                code = code[: code.rfind("\n}")] + "\n}"
        elif language_type == "js" or language_type == "javascript":
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
            if "\n}" in code:
                code = code[: code.rfind("\n}")] + "\n}"
        elif language_type == "rust":
            main_pos = code.find("\nfn main()")
            if main_pos != -1:
                code = code[:main_pos]
            if "}" in code:
                code = code[: code.rfind("}")] + "}"
            if "\n}" in code:
                code = code[: code.rfind("\n}")] + "\n}"

    return code


IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}


def process_test(
    sample: Dict, problem: Dict, dataset: str = None, example_test: bool = False
) -> str:
    if dataset is None:
        raise ValueError("dataset cannot be None")
    dataset = dataset.lower()
    if "humaneval" not in dataset and "mbxp" != dataset:
        raise ValueError("dataset type must be humaneval or mbxp")

    if "language" in sample:
        language = sample["language"].lower()
    else:
        language = sample["task_id"].split("/")[0].lower()

    prompt = sample["prompt"]
    if example_test and "example_test" in problem and problem["example_test"] != "":
        test = problem["example_test"]
    else:
        test = problem["test"]
    code = cleanup_code(sample["generation"], language_type=language, dataset=dataset)

    # Pre-process for different languages
    if language == "python":
        code_ = []
        for line in code.split("\n"):
            if len(line.strip()) > 0 and line[0] != " " and line[0] != "\t":
                break
            code_.append(line)
        code = "\n".join(code_)
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + prompt + code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        test_string = test_set_up + "\n" + prompt + code + "\n" + test
    elif language == "java":
        test_string = prompt + code + "\n" + test
    elif language == "js" or language == "javascript":
        test_string = prompt + code + "\n" + test
    elif language == "go":
        import_string = problem["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and "example_test" in problem:
            test = problem["example_test"]
        else:
            test = problem["test"]
        test_setup = problem["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in code:
                    other_pkgs.append(f'"{pkg}"')
        if other_pkgs:
            import_other_pkgs = (
                "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            )
            test_string = (
                test_setup
                + "\n"
                + import_other_pkgs
                + "\n"
                + prompt
                + code
                + "\n"
                + test
            )
        else:
            test_string = test_setup + "\n" + prompt + code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = problem["declaration"]
        test_string = main + declaration + prompt + code + test

    return test_string
