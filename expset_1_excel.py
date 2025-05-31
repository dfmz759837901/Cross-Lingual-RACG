import os
import pickle
from typing import *
import pandas as pd
from copy import deepcopy
from utils import stream_json_all, write_pd_to_excel, init_sheet, count_pass_at_k

pd.options.display.float_format = "{:.2f}".format


def count(root_path: str, sample_n: int) -> Dict[Tuple[str, str, str], float]:
    table = {}
    for generation_mode in os.listdir(root_path):
        for model_name in os.listdir(os.path.join(root_path, generation_mode)):
            for result_lang_file in os.listdir(
                os.path.join(root_path, generation_mode, model_name)
            ):
                if result_lang_file.endswith(".jsonl"):
                    lang = result_lang_file.split(".")[0][len("result_") :]
                    result_file_path = os.path.join(
                        root_path, generation_mode, model_name, result_lang_file
                    )
                    print(f"Processing {result_file_path}")
                    result_data = stream_json_all(result_file_path)
                    pass_at_k, _ = count_pass_at_k(result_data, pass_k=sample_n)
                    table[(generation_mode, model_name, lang)] = pass_at_k
                    print(f"pass@{sample_n} = {round(pass_at_k * 100, 2)}")
    return table


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


def output_dflist(
    data_list: List[Optional[pd.DataFrame]],
    sheet_name: str,
    save_path: str,
    startrow: int = 0,
):
    lines = data_list[0].shape[0]
    for data in data_list[1:]:
        if data is None:
            continue
        if data.shape[0] != lines:
            raise ValueError("data_list must have the same number of lines")
    startcol = 0
    for data in data_list:
        if data is None:
            continue
        write_pd_to_excel(
            df=data,
            sheet_name=sheet_name,
            save_path=save_path,
            startrow=startrow,
            startcol=startcol,
            mode="a",
            if_sheet_exists="overlay",
        )
        startcol += data.shape[1] + 2


def display(table: Dict[Tuple[str, str, str], float], sheet_name: str, save_path: str):

    init_sheet(sheet_name, save_path)

    generation_mode_list = set()
    model_list = set()
    gen_lang_list = set()
    ref_lang_list = set()

    for key in table.keys():
        generation_mode_list.add(key[0])
        model_list.add(key[1])
        gen_lang_list.add(key[2])

    generation_mode_list = sorted(list(generation_mode_list))
    model_list = sorted(list(model_list))
    gen_lang_list = sorted(list(gen_lang_list))

    for m in ["baseline_fewshot"]:
        if m in generation_mode_list:
            generation_mode_list.remove(m)
            generation_mode_list.insert(0, m)

    startrow = 0
    baseline_generation_mode_df = None
    for generation_mode in generation_mode_list:
        if generation_mode.startswith("RAG_lang_"):
            ref_lang_list.add(generation_mode[len("RAG_lang_") :])
            continue

        data = pd.DataFrame(index=model_list, columns=gen_lang_list)
        data.index.name = generation_mode + "%"
        for model in model_list:
            for lang in gen_lang_list:
                key = (generation_mode, model, lang)
                data.loc[model, lang] = table[key] * 100 if key in table else None

        data_add = None
        data_add_ratio = None
        if baseline_generation_mode_df is None:
            if "baseline" in generation_mode:
                baseline_generation_mode_df = data
        else:
            data_add = deepcopy(data)
            data_add.index.name += "+"
            data_add_ratio = deepcopy(data)
            data_add_ratio.index.name += "+/%"
            for r in model_list:
                for c in gen_lang_list:
                    if data_add.loc[r, c] is not None:
                        data_add.loc[r, c] -= baseline_generation_mode_df.loc[r, c]
                        data_add_ratio.loc[r, c] = (
                            (
                                data_add.loc[r, c]
                                * 100
                                / baseline_generation_mode_df.loc[r, c]
                            )
                            if baseline_generation_mode_df.loc[r, c] > 0
                            else None
                        )
        output_dflist(
            [
                calc_mean(calc_mean(d, axis=1), axis=0)
                for d in [data, data_add, data_add_ratio]
            ],
            sheet_name=sheet_name,
            save_path=save_path,
            startrow=startrow,
        )
        startrow += data.shape[0] + 3

    ref_lang_list = sorted(list(ref_lang_list))
    for model in model_list:
        data = pd.DataFrame(
            index=["ref_" + r for r in ref_lang_list], columns=gen_lang_list
        )
        data.index.name = model + "%"
        for lang in ref_lang_list:
            for gen_lang in gen_lang_list:
                key = ("RAG_lang_" + lang, model, gen_lang)
                data.loc["ref_" + lang, gen_lang] = (
                    table[key] * 100 if key in table else None
                )
        assert baseline_generation_mode_df is not None
        data_add = deepcopy(data)
        data_add.index.name += "+"
        data_add_ratio = deepcopy(data)
        data_add_ratio.index.name += "+/%"
        for r in ref_lang_list:
            for c in gen_lang_list:
                if r == c:
                    data_add.loc["ref_" + r, c] = None
                    data_add_ratio.loc["ref_" + r, c] = None
                if data_add.loc["ref_" + r, c] is not None:
                    data_add.loc["ref_" + r, c] -= baseline_generation_mode_df.loc[
                        model, c
                    ]
                    data_add_ratio.loc["ref_" + r, c] = (
                        (
                            data_add.loc["ref_" + r, c]
                            * 100
                            / baseline_generation_mode_df.loc[model, c]
                        )
                        if baseline_generation_mode_df.loc[model, c] > 0
                        else None
                    )
        output_dflist(
            [
                calc_mean(calc_mean(d, axis=1), axis=0)
                for d in [data, data_add, data_add_ratio]
            ],
            sheet_name=sheet_name,
            save_path=save_path,
            startrow=startrow,
        )
        startrow += data.shape[0] + 3


if __name__ == "__main__":
    DATASET = "humaneval-x"
    RETRIEVAL_MODE = "random"
    SAMPLE_N = 1
    EVAL_RESULTS_DIR = "eval_results"

    root_path = os.path.join(EVAL_RESULTS_DIR, DATASET, RETRIEVAL_MODE)
    # table_G_cache = os.path.join(root_path, "table_G_exp_1.pkl")
    # if os.path.exists(table_G_cache):
    #     with open(table_G_cache, "rb") as f:
    #         table = pickle.load(f)
    # else:
    table = count(root_path, sample_n=SAMPLE_N)
    generation_mode_list = set()
    gen_lang_list = set()
    for key in table.keys():
        generation_mode_list.add(key[0])
        gen_lang_list.add(key[2])
    generation_mode_list = sorted(list(generation_mode_list))
    gen_lang_list = sorted(list(gen_lang_list))
    for generation_mode in generation_mode_list:
        for language in gen_lang_list:
            table[(generation_mode, "mono-lingual-model", language)] = (
                table[(generation_mode, "phi-1", language)]
                + table[(generation_mode, "phi-1_5", language)]
            ) / 2
            table[(generation_mode, "multi-lingual-model", language)] = (
                table[(generation_mode, "CodeLlama-7b-Instruct-hf", language)]
                + table[(generation_mode, "deepseek-coder-6.7b-instruct", language)]
                + table[(generation_mode, "Qwen2.5-Coder-7B-Instruct", language)]
            ) / 3
    # with open(table_G_cache, "wb") as f:
    #     pickle.dump(table, f)
    display(table, sheet_name="Experiment_1", save_path="table_G_exp_1.xlsx")
