#!/bin/bash

MODELS=(
    "llama3_8b_lora_sft_20epochs"
    "llama3_8b_lora_sft_5epochs"
    # "llama3_8b_lora_sft_3epochs"
    "llama3.2_3b_lora_sft_3epochs"
    "llama3.2_3b_lora_sft_5epochs"
    "llama3.2_3b_lora_sft_20epochs"
)

INPUT_DIR="results/step1_generated_text/ft"
OUTPUT_DIR="results/step2_clean_sparql/ft"

for MODEL in "${MODELS[@]}"
do
    INPUT_PATH="${INPUT_DIR}/${MODEL}"
    OUTPUT_PATH="${OUTPUT_DIR}/${MODEL}"
    echo "Running sparql cleaning on results over model: $MODEL"
    python sparql-cleaning-llm.py "$INPUT_PATH" "$OUTPUT_PATH"
done
