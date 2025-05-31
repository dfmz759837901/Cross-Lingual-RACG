import pandas as pd
from typing import *
import os


def init_sheet(
    sheet_name: str,
    save_path: str,
    mode: Literal["w", "a"] = "w",
):
    assert mode in ["w", "a"], "mode must be 'w' or 'a'"
    with pd.ExcelWriter(save_path, engine="openpyxl", mode=mode) as writer:
        writer.book.create_sheet(sheet_name)


def write_pd_to_excel(
    df: pd.DataFrame,
    sheet_name: str,
    save_path: str,
    startrow: int = 0,
    startcol: int = 0,
    mode: Literal["w", "a"] = "w",
    if_sheet_exists: Optional[Literal["error", "new", "replace", "overlay"]] = None,
    header: bool = True,
):
    if mode == "a" and not os.path.exists(save_path):
        init_sheet(sheet_name, save_path)
    with pd.ExcelWriter(
        save_path, engine="openpyxl", mode=mode, if_sheet_exists=if_sheet_exists
    ) as writer:
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=startrow,
            startcol=startcol,
            header=header,
            float_format="%.2f",
        )
    print(f"Data saved to {save_path}")


def write_dict_to_excel(
    data: Dict[str, List],
    sheet_name: str,
    save_path: str,
    startrow: int = 0,
    startcol: int = 0,
    mode: Literal["w", "a"] = "w",
    if_sheet_exists: Literal["error", "new", "replace", "overlay"] = "error",
):
    df = pd.DataFrame(data).T
    write_pd_to_excel(
        df=df,
        sheet_name=sheet_name,
        save_path=save_path,
        startrow=startrow,
        startcol=startcol,
        mode=mode,
        if_sheet_exists=if_sheet_exists,
        header=False,
    )


def write_pdlist_to_excel(
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
        startcol += data.shape[1] + 3


if __name__ == "__main__":
    data = {
        "1-shot": ["cpp", "go", "java", "js", "python", "mean"],
        "CodeLlama-7b-Instruct-hf": [31.10, 25.73, 35.85, 34.02, 35.98, 32.54],
        "CodeLlama-13b-Instruct-hf": [36.71, 23.17, 41.59, 36.10, 32.56, 34.03],
        "deepseek-coder-6.7b-instruct": [62.44, 58.41, 66.95, 67.93, 73.41, 65.83],
        "phi-1_5": [14.63, 7.20, 16.71, 19.02, 38.54, 19.22],
    }
    write_dict_to_excel(
        data,
        "Sheet1",
        "temp_output.xlsx",
        startrow=18,
        startcol=1,
        mode="a",
        if_sheet_exists="overlay",
    )
