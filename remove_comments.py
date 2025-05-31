import os
from utils import stream_json_all, remove_comments, dicts_to_jsonl

DATASET_PATH = "datasets"

if __name__ == "__main__":
    langs = [
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
    ]
    datasets = ["multilingual_humaneval", "mbxp", "mceval", "humaneval-x"]

    for lang in langs:
        for dataset in datasets:
            filepath = os.path.join(
                DATASET_PATH, dataset, f"{dataset}_correct_{lang}.jsonl"
            )
            try:
                dicts = stream_json_all(filepath)
            except:
                print(f"File not found: {filepath}")
                continue

            print(f"Start removing comments for {dataset} {lang}\n\nExample 0:")
            for d in dicts:
                code = f"\n{d['prompt']}\n{d['canonical_solution']}\n"
                code = remove_comments(code)
                if d == dicts[0]:
                    print(f"[task_id] {d['task_id']}")
                    print(f"[origin code]\n{d['prompt']}\n{d['canonical_solution']}")
                    print(f"[code]\n{code}")
                    print("-" * 20)
                d["code_without_comments"] = code
            dicts_to_jsonl(dicts, filepath, compress=False)
