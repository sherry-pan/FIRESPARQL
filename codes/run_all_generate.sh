#!/bin/bash

MODELS=(
    "llama3.2_3b_lora_sft_3epochs"
    "llama3.2_3b_lora_sft_5epochs"
    "llama3.2_3b_lora_sft_20epochs"
    "llama3_8b_lora_sft_3epochs"
    "llama3_8b_lora_sft_5epochs"
    "llama3_8b_lora_sft_20epochs"
)

TEST_DATA="xueli_data/sciqa/project_data/test_questions.csv"
OUTPUT_DIR="results/step1_generated_text/ft"
ROUNDS=(
    "round1"
    "round2"
    "round3"
)

for MODEL in "${MODELS[@]}"; do
    for ROUND in "${ROUNDS[@]}"; do
        OUTPUT_PATH="${OUTPUT_DIR}/${MODEL}_${ROUND}"
        echo "Running model: $MODEL"
        python generate_sparql_cuda.py "merge_models/$MODEL" "$TEST_DATA" "$OUTPUT_PATH"
    done
done
echo "All generation tasks completed."