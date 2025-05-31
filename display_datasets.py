from utils import stream_json_all, display_dict, dicts_to_jsonl
import os

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
                continue
            change_flag = False
            for d in dicts:
                if "language" not in d:
                    d["language"] = lang
                    change_flag = True
                assert d["language"] == lang
            if change_flag:
                dicts_to_jsonl(dicts, filepath, compress=False)
            outputpath = f"datadisplay/{dataset}/{dataset}_{lang}.txt"
            os.makedirs(os.path.dirname(outputpath), exist_ok=True)
            with open(outputpath, "w", encoding="utf-8") as f:
                for d in dicts:
                    display_dict(d, file=f)
                    f.write("\n")
