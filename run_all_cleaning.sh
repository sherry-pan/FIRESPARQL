#!/bin/bash

MODELS=(
    # "llama3.2_3b_lora_sft_7epochs"
    # "llama3.2_3b_lora_sft_10epochs"
    # "llama3.2_3b_lora_sft_20epochs"
    # "llama3_8b_lora_sft_7epochs"
    # "llama3_8b_lora_sft_10epochs"
    # "llama3_8b_lora_sft_15epochs"
    "llama-3.2-3b-Instruct"
    "llama-3-8b-Instruct"
)

INPUT_DIR="results/step1_generated_text/vanilla_rag"
OUTPUT_DIR="results/step2_clean_sparql/vanilla_rag"
ROUNDS=(
    "round1"
    "round2"
    "round3"
)

for MODEL in "${MODELS[@]}"; do
    for ROUND in "${ROUNDS[@]}"; do
        INPUT_PATH="${INPUT_DIR}/${MODEL}_${ROUND}"
        OUTPUT_PATH="${OUTPUT_DIR}/${MODEL}_${ROUND}"
        echo "Running sparql cleaning on results over model: $MODEL, round: $ROUND"
        python sparql-cleaning-llm.py "$INPUT_PATH" "$OUTPUT_PATH"
    done   
done
echo "All cleaning tasks completed."