import os
import sys
from typing import *
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy
from utils import stream_json_all, write_pd_to_excel, init_sheet, count_pass_at_k

pd.options.display.float_format = "{:.2f}".format

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

DATASETS = {"humaneval-x"}
RETRIEVAL_MODES = {"random"}
SAMPLE_N = 1
EVAL_RESULTS_DIR = "eval_results"
MODELS = [
    "CodeLlama-7b-Instruct-hf",
    "deepseek-coder-6.7b-instruct",
    "Qwen2.5-Coder-7B-Instruct",
    "phi-1",
    "phi-1_5",
]
LANGUAGES = [
    "python",
    "java",
    "javascript",
    "cpp",
    "go",
]
POISONS = ["logic", "control_flow", "syntax", "lexicon"]
POISONED_LANGS = ["python", "java"]


def count(
    root_path: str, sample_n: int
) -> Dict[Tuple[str, str, str, str, str], Dict[str, Union[float, int]]]:
    table_dict = {}
    for poisoned_lang in POISONED_LANGS:
        for model_name in os.listdir(
            os.path.join(root_path, f"RAG_lang_{poisoned_lang}")
        ):
            for poison in POISONS:

                baseline_modes = ["baseline_fewshot", f"RAG_lang_{poisoned_lang}"]
                for baseline_mode in baseline_modes:
                    dir = os.path.join(root_path, baseline_mode, model_name)
                    for result_lang_file in os.listdir(dir):
                        if not result_lang_file.endswith(".jsonl"):
                            continue
                        lang = result_lang_file.split(".")[0][len("result_") :]
                        if (baseline_mode, model_name, lang) in table_dict:
                            continue
                        result_file_path = os.path.join(dir, result_lang_file)
                        print(f"Processing {result_file_path}")
                        result_data = stream_json_all(result_file_path)
                        pass_at_k, data_num = count_pass_at_k(
                            result_data, pass_k=sample_n
                        )
                        table_dict[(baseline_mode, model_name, lang)] = {
                            f"pass@{sample_n}": pass_at_k,
                            "num": data_num,
                        }
                        print(
                            f"pass@{sample_n} = {round(pass_at_k * 100, 2)}, num = {data_num}"
                        )

                dir = os.path.join(
                    root_path,
                    f"RAG_lang_{poisoned_lang}",
                    model_name,
                    poison,
                    poisoned_lang,
                )
                for result_lang_file in os.listdir(dir):
                    if not result_lang_file.endswith(".jsonl"):
                        continue
                    lang = result_lang_file.split(".")[0][len("result_") :]
                    result_file_path = os.path.join(dir, result_lang_file)
                    print(f"Processing {result_file_path}")
                    result_data = stream_json_all(result_file_path)
                    pass_at_k, data_num = count_pass_at_k(result_data, pass_k=sample_n)
                    table_dict[
                        (
                            f"RAG_lang_{poisoned_lang}",
                            model_name,
                            poison,
                            poisoned_lang,
                            lang,
                        )
                    ] = {
                        f"pass@{sample_n}": pass_at_k,
                        "num": data_num,
                    }
                    print(
                        f"pass@{sample_n} = {round(pass_at_k * 100, 2)}, num = {data_num}"
                    )

    return table_dict


def calc_mean(data: Optional[pd.DataFrame], axis: int = 0) -> pd.DataFrame:
    if data is None:
        return None
    data = deepcopy(data)
    if axis == 1:
        means = data.mean(axis=axis)
        means.name = "mean"
        return pd.concat([data, means], axis=axis)
    elif axis == 0:
        means = data.mean(axis=axis)
        data.loc["mean"] = means
        return data
    else:
        raise ValueError("axis must be 0 or 1")


def display(
    table_dict: Dict[Tuple[str, str, str, str, str], Dict[str, Union[float, int]]],
    sheet_name: str,
    save_path: str,
    mode: Literal["w", "a"] = "w",
):

    init_sheet(sheet_name, save_path, mode)

    generation_mode_list = ["baseline_fewshot"]
    generation_mode_list += [
        f"RAG_lang_{poisoned_lang}" for poisoned_lang in POISONED_LANGS
    ]
    generation_mode_list += [
        f"RAG_lang_{poisoned_lang}_{poison}"
        for poisoned_lang in POISONED_LANGS
        for poison in POISONS
    ]

    gen_lang_list = set()
    model_list = set()
    for key in table_dict.keys():
        model_list.add(key[1])
        gen_lang_list.add(key[-1])
    model_list = sorted(list(model_list))
    gen_lang_list = sorted(list(gen_lang_list))

    startrow = 0
    for model in model_list:
        data = pd.DataFrame(index=generation_mode_list, columns=gen_lang_list)
        data.index.name = model + "%"

        data_add = pd.DataFrame(index=generation_mode_list, columns=gen_lang_list)
        data_add.index.name = model + "+%"

        baseline_modes = ["baseline_fewshot"] + [
            f"RAG_lang_{poisoned_lang}" for poisoned_lang in POISONED_LANGS
        ]

        for baseline_mode in baseline_modes:
            for gen_lang in gen_lang_list:
                key = (baseline_mode, model, gen_lang)
                data.loc[baseline_mode, gen_lang] = (
                    table_dict[key][f"pass@{SAMPLE_N}"] * 100
                    if key in table_dict
                    else None
                )
                data_add.loc[baseline_mode, gen_lang] = None

        for poisoned_lang in POISONED_LANGS:
            for poison in POISONS:
                for gen_lang in gen_lang_list:
                    key = (
                        f"RAG_lang_{poisoned_lang}",
                        model,
                        poison,
                        poisoned_lang,
                        gen_lang,
                    )
                    data.loc[f"RAG_lang_{poisoned_lang}_{poison}", gen_lang] = (
                        table_dict[key][f"pass@{SAMPLE_N}"] * 100
                        if key in table_dict
                        else None
                    )
                    data_add.loc[f"RAG_lang_{poisoned_lang}_{poison}", gen_lang] = (
                        data.loc[f"RAG_lang_{poisoned_lang}_{poison}", gen_lang]
                        - data.loc[f"RAG_lang_{poisoned_lang}", gen_lang]
                    )

        write_pd_to_excel(
            df=calc_mean(calc_mean(data, axis=1), axis=0),
            sheet_name=sheet_name,
            save_path=save_path,
            startrow=startrow,
            startcol=0,
            mode="a",
            if_sheet_exists="overlay",
        )
        write_pd_to_excel(
            df=calc_mean(calc_mean(data_add, axis=1), axis=0),
            sheet_name=sheet_name,
            save_path=save_path,
            startrow=startrow,
            startcol=data.shape[1] + 3,
            mode="a",
            if_sheet_exists="overlay",
        )
        startrow += data.shape[0] + 3


if __name__ == "__main__":

    for dataset in DATASETS:
        for retrieval_mode in RETRIEVAL_MODES:
            table_dict = count(
                os.path.join(EVAL_RESULTS_DIR, dataset, retrieval_mode), SAMPLE_N
            )

            baseline_modes = ["baseline_fewshot"] + [
                f"RAG_lang_{poisoned_lang}" for poisoned_lang in POISONED_LANGS
            ]

            for baseline_mode in baseline_modes:
                for language in LANGUAGES:

                    for model in MODELS:
                        assert (
                            table_dict[(baseline_mode, model, language)]["num"]
                            == table_dict[(baseline_mode, "phi-1", language)]["num"]
                        )

                    table_dict[(baseline_mode, "mono-lingual-model", language)] = {
                        f"pass@{SAMPLE_N}": (
                            table_dict[(baseline_mode, "phi-1", language)][
                                f"pass@{SAMPLE_N}"
                            ]
                            + table_dict[(baseline_mode, "phi-1_5", language)][
                                f"pass@{SAMPLE_N}"
                            ]
                        )
                        / 2,
                        "num": table_dict[(baseline_mode, "phi-1", language)]["num"],
                    }
                    table_dict[(baseline_mode, "multi-lingual-model", language)] = {
                        f"pass@{SAMPLE_N}": (
                            table_dict[
                                (baseline_mode, "CodeLlama-7b-Instruct-hf", language)
                            ][f"pass@{SAMPLE_N}"]
                            + table_dict[
                                (
                                    baseline_mode,
                                    "deepseek-coder-6.7b-instruct",
                                    language,
                                )
                            ][f"pass@{SAMPLE_N}"]
                            + table_dict[
                                (baseline_mode, "Qwen2.5-Coder-7B-Instruct", language)
                            ][f"pass@{SAMPLE_N}"]
                        )
                        / 3,
                        "num": table_dict[
                            (baseline_mode, "CodeLlama-7b-Instruct-hf", language)
                        ]["num"],
                    }

            for poisoned_lang in POISONED_LANGS:
                for poison in POISONS:
                    for language in LANGUAGES:

                        for model in MODELS:
                            assert (
                                table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        model,
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ]["num"]
                                == table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        "phi-1",
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ]["num"]
                            )

                        table_dict[
                            (
                                f"RAG_lang_{poisoned_lang}",
                                "mono-lingual-model",
                                poison,
                                poisoned_lang,
                                language,
                            )
                        ] = {
                            f"pass@{SAMPLE_N}": (
                                table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        "phi-1",
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ][f"pass@{SAMPLE_N}"]
                                + table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        "phi-1_5",
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ][f"pass@{SAMPLE_N}"]
                            )
                            / 2,
                            "num": table_dict[
                                (
                                    f"RAG_lang_{poisoned_lang}",
                                    "phi-1",
                                    poison,
                                    poisoned_lang,
                                    language,
                                )
                            ]["num"],
                        }

                        table_dict[
                            (
                                f"RAG_lang_{poisoned_lang}",
                                "multi-lingual-model",
                                poison,
                                poisoned_lang,
                                language,
                            )
                        ] = {
                            f"pass@{SAMPLE_N}": (
                                table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        "CodeLlama-7b-Instruct-hf",
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ][f"pass@{SAMPLE_N}"]
                                + table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        "deepseek-coder-6.7b-instruct",
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ][f"pass@{SAMPLE_N}"]
                                + table_dict[
                                    (
                                        f"RAG_lang_{poisoned_lang}",
                                        "Qwen2.5-Coder-7B-Instruct",
                                        poison,
                                        poisoned_lang,
                                        language,
                                    )
                                ][f"pass@{SAMPLE_N}"]
                            )
                            / 3,
                            "num": table_dict[
                                (
                                    f"RAG_lang_{poisoned_lang}",
                                    "Qwen2.5-Coder-7B-Instruct",
                                    poison,
                                    poisoned_lang,
                                    language,
                                )
                            ]["num"],
                        }

            display(
                table_dict, sheet_name=dataset, save_path="table_G_exp_4.xlsx", mode="w"
            )
