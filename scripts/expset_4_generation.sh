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
    "go"
    "java"
    "javascript"
    "python"
)
dataset="humaneval-x"
retrieval_modes=(
    "random" 
)
poisons=(
    "logic"
    "syntax"
    "lexicon"
    "control_flow"
)
poisoned_langs=(
    "python"
    "java"
)

for retrieval_mode in "${retrieval_modes[@]}"; do
    for generation_model_name_or_path in "${generation_model_paths[@]}"; do
        for poison in "${poisons[@]}"; do
            for poisoned_lang in "${poisoned_langs[@]}"; do
                generation_modes=(
                    "RAG_lang_${poisoned_lang}"
                )
                for generation_mode in "${generation_modes[@]}"; do

                    generation_model_name=$(basename "$generation_model_name_or_path")

                    log_filepath="logs/poisoning/${dataset}/${retrieval_mode}/${generation_mode}/${poison}/${poisoned_lang}/log_${generation_model_name}.txt"

                    log_dir=$(dirname "$log_filepath")
                    if [ ! -d "$log_dir" ]; then
                        mkdir -p "$log_dir"
                    fi
                    
                    echo "$dataset Generate Start: $generation_model_name_or_path with mode -g $generation_mode -r $retrieval_mode"
                    echo "Poison: $poison, Poisoned Lang: $poisoned_lang"

                    python generation.py \
                        --model_name_or_path "$generation_model_name_or_path" \
                        --generation_mode "$generation_mode" \
                        --retrieval_mode "$retrieval_mode" \
                        --retrieve_size 1 \
                        --poison "$poison" \
                        --poisoned_lang "$poisoned_lang" \
                        --dataset "$dataset" \
                        --sample_n 1 \
                        --temperature 0.0 \
                        > "$log_filepath"

                    echo "$dataset Generate Completed: $generation_model_name_or_path with mode -g $generation_mode -r $retrieval_mode"
                    echo "Poison: $poison, Poisoned Lang: $poisoned_lang"
                done
            done
        done
    done
done