#!/bin/bash

generation_model_paths=(
    "../hf_models/codellama/CodeLlama-7b-Instruct-hf"
    "../hf_models/deepseek-ai/deepseek-coder-6.7b-instruct"
    "../hf_models/Qwen/Qwen2.5-Coder-7B-Instruct"
    "../hf_models/microsoft/phi-1"
    "../hf_models/microsoft/phi-1_5"
)
langs=(
    "python"
    "java"
    "javascript"
    "typescript"
    "kotlin"
    "ruby"
    "php"
    "cpp"
    "csharp"
    "go"
    "perl"
    "scala"
    "swift"
)
datasets=(
    "mceval"
    "multilingual_humaneval"
    "mbxp"
)
retrieval_modes=(
    "dense_CodeRankEmbed" 
)
generation_modes=(
    "baseline_fewshot"
)
generation_modes+=("RAG_cross_lang")
for lang in "${langs[@]}"; do
    generation_modes+=("RAG_lang_$lang")
done

for dataset in "${datasets[@]}"; do
    for retrieval_mode in "${retrieval_modes[@]}"; do
        for generation_model_name_or_path in "${generation_model_paths[@]}"; do
            for generation_mode in "${generation_modes[@]}"; do

                generation_model_name=$(basename "$generation_model_name_or_path")
                
                log_filepath="logs/generation/${dataset}/${retrieval_mode}_without_nl/${generation_mode}/log_${generation_model_name}.txt"

                log_dir=$(dirname "$log_filepath")
                if [ ! -d "$log_dir" ]; then
                    mkdir -p "$log_dir"
                fi
                
                echo "$dataset Generate Start: $generation_model_name_or_path with mode -g $generation_mode -r $retrieval_mode without_nl"

                python generation.py \
                    --model_name_or_path "$generation_model_name_or_path" \
                    --generation_mode "$generation_mode" \
                    --retrieval_mode "$retrieval_mode" \
                    --without_nl \
                    --dataset "$dataset" \
                    --sample_n 1 \
                    --temperature 0.0 \
                    > "$log_filepath"
                
                echo "$dataset Generate Completed: $generation_model_name_or_path with mode -g $generation_mode -r $retrieval_mode without_nl"
            done
        done
    done
done