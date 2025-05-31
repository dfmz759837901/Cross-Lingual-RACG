from utils import stream_json_all, draw_venn

import sys

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if __name__ == "__main__":
    gen_langs = [
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
    ref_langs = [
        "python",
        "java",
    ]
    model_names = [
        "CodeLlama-7b-Instruct-hf",
        "deepseek-coder-6.7b-instruct",
        "Qwen2.5-Coder-7B-Instruct",
    ]
    datasets = [
        # "humaneval-x",
        "mceval",
        "multilingual_humaneval",
        "mbxp",
    ]
    retrieval_modes = [
        # "random",
        "dense_CodeRankEmbed",
    ]
    poisons = [
        "logic",
        "control_flow",
        "syntax",
        "lexicon",
    ]

    enhance_set = {p: [] for p in poisons}
    for dataset in datasets:
        for retrieval_mode in retrieval_modes:
            for model_name in model_names:
                for gen_lang in gen_langs:

                    try:
                        baselinefile = f"eval_results/{dataset}/dense_CodeRankEmbed_without_nl/baseline_fewshot/{model_name}/result_{gen_lang}.jsonl"
                        baseline = stream_json_all(baselinefile)
                    except KeyboardInterrupt:
                        break
                    except:
                        continue

                    for ref_lang in ref_langs:
                        if ref_lang != gen_lang:

                            try:
                                filepath1 = f"eval_results/{dataset}/{retrieval_mode}/RAG_lang_{ref_lang}/{model_name}/result_{gen_lang}.jsonl"
                                dict_list1 = stream_json_all(filepath1)
                            except KeyboardInterrupt:
                                break
                            except:
                                continue

                            for poison in poisons:
                                try:
                                    filepath2 = f"eval_results/{dataset}/{retrieval_mode}/RAG_lang_{ref_lang}/{model_name}/{poison}/{ref_lang}/result_{gen_lang}.jsonl"
                                    dict_list2 = stream_json_all(filepath2)
                                except KeyboardInterrupt:
                                    break
                                except:
                                    continue

                                # print(f"Gen_Lang = {gen_lang}")
                                # print(f"Ref_Lang = {ref_lang}")
                                # print(f"Model = {model_name}")
                                # print(f"Dataset = {dataset}")
                                # print(f"Retrieval Mode = {retrieval_mode}")
                                # print(f"Poison = {poison}")

                                for d2 in dict_list2:
                                    if d2["passed"]:
                                        for d1 in dict_list1:
                                            if d1["task_id"] == d2["task_id"]:
                                                if not d1["passed"]:
                                                    for d3 in baseline:
                                                        if (
                                                            d3["task_id"]
                                                            == d2["task_id"]
                                                        ):
                                                            if not d3["passed"]:
                                                                enhance_set[
                                                                    poison
                                                                ].append(
                                                                    f"<{model_name}><{dataset}><{retrieval_mode}><ref_{ref_lang}><gen_{gen_lang}><{d3['task_id']}>"
                                                                )
                                                            break
                                                break
                                # print("=" * 20)

    total_set = []
    for p in poisons:
        enhance_set[p] = list(set(enhance_set[p]))
        with open(f"figures/{p}.txt", "w", encoding="utf-8") as f:
            for item in enhance_set[p]:
                f.write(f"{item}\n")
        total_set.extend(enhance_set[p])

    total_set = list(set(total_set))
    total_cases = len(total_set)
    print(f"Total Cases = {total_cases}")
    enhance_set["logic_all"] = list(
        set((enhance_set["logic"] + enhance_set["control_flow"]))
    )
    for p in poisons + ["logic_all"]:
        print(f"Poison = {p}")
        print(f"Enhance Set Len = {len(enhance_set[p])}")
        print(f"Ratio / Total Cases = {len(enhance_set[p]) / total_cases * 100:.2f} %")
        # print(f"Enhance Set =\n{enhance_set[p]}")

    title = "Venn Diagram for Perturbation Types"
    title = ""

    draw_venn(
        subsets=[
            set(enhance_set["logic_all"]),
            set(enhance_set["syntax"]),
            set(enhance_set["lexicon"]),
        ],
        set_labels=("Semantics", "Syntax", "Lexicon"),
        save_path="figures/venn_perturbation_1.png",
        title=title,
        colors=['#0072B2', '#D55E00', '#CC79A7']
    )

    draw_venn(
        subsets=[
            set(enhance_set["logic"]),
            set(enhance_set["syntax"]),
            set(enhance_set["control_flow"]),
        ],
        set_labels=("Logic words", "Syntax", "Control Flow"),
        save_path="figures/venn_perturbation_2.png",
        title=title,
        colors=['#0072B2', '#D55E00', '#CC79A7']
    )

    draw_venn(
        subsets=[
            set(enhance_set["logic"]),
            set(enhance_set["syntax"]),
        ],
        set_labels=("Logic keyword", "Syntax"),
        save_path="figures/venn_perturbation_3.png",
        title=title,
        colors=['#0072B2', '#CC79A7']
    )
