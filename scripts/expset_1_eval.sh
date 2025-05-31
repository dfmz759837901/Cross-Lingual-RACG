#!/bin/bash

generation_model_paths=(
    "../hf_models/codellama/CodeLlama-7b-Instruct-hf"
    "../hf_models/deepseek-ai/deepseek-coder-6.7b-instruct"
    "../hf_models/Qwen/Qwen2.5-Coder-7B-Instruct"
    "../hf_models/microsoft/phi-1"
    "../hf_models/microsoft/phi-1_5"
)
langs=(
    "cpp"
    "java"
    "python"
    "go"
    "javascript"
)
dataset="humaneval-x"
retrieval_modes=(
    "random" 
)
generation_modes=(
    "baseline_fewshot"
)

for lang in "${langs[@]}"; do
    generation_modes+=("RAG_lang_$lang")
done

rm -r third_party/mxeval/mxeval/*_eval

for retrieval_mode in "${retrieval_modes[@]}"; do
    for generation_model_name_or_path in "${generation_model_paths[@]}"; do
        for generation_mode in "${generation_modes[@]}"; do

            generation_model_name=$(basename "$generation_model_name_or_path")
            
            log_filepath="logs/eval/${dataset}/${retrieval_mode}/${generation_mode}/log_${generation_model_name}.txt"

            log_dir=$(dirname "$log_filepath")
            if [ ! -d "$log_dir" ]; then
                mkdir -p "$log_dir"
            fi

            rm -r third_party/mxeval/mxeval/*_eval
            echo "Eval Start: $generation_model_name_or_path with mode -g $generation_mode -r $retrieval_mode"

            python eval.py \
                --model_name_or_path "$generation_model_name_or_path" \
                --generation_mode "$generation_mode" \
                --retrieval_mode "$retrieval_mode" \
                --dataset "$dataset" \
                --sample_n 1 \
                > "$log_filepath"
            
            echo "Eval Completed: $generation_model_name_or_path with mode -g $generation_mode -r $retrieval_mode"
        done
    done
done

rm -r third_party/mxeval/mxeval/*_eval