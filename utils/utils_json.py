import gzip
import json
import os
import re
from typing import Dict, List, Union, TextIO, Optional


def dicts_to_jsonl(data_list: List[Dict], filename: str, compress: bool = True) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = ".jsonl"
    sgz = ".gz"
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data

    if compress:
        filename = filename + sgz
        with gzip.open(filename, "w") as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict, ensure_ascii=False) + "\n"
                jout = jout.encode("utf-8")
                compressed.write(jout)
    else:
        with open(filename, "w", encoding="utf-8") as out:
            for ddict in data_list:
                jout = json.dumps(ddict, ensure_ascii=False) + "\n"
                out.write(jout)


def stream_json_all(filename: str) -> Union[List[Dict], Dict]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    results = []
    open_func = gzip.open if filename.endswith(".gz") else open

    with open_func(filename, "rt", encoding="utf-8") as fp:
        if filename.endswith((".jsonl", ".gz")):
            for line in fp:
                if line.strip():  # Check if the line contains non-whitespace characters
                    results.append(json.loads(line))
        elif filename.endswith(".json"):
            results = json.load(fp)  # Load the entire JSON structure
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    return results


def display_dict(d: Dict, file: Optional[TextIO] = None):
    for k, v in d.items():
        if isinstance(v, str):
            v = "\n".join(line.rstrip() for line in v.splitlines() if line.rstrip())
            v = v.rstrip()
        line = f"[{k}]\n{v}\n"
        if file:
            file.write(line)
        else:
            print(line, end="")
