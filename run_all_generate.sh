#!/bin/bash

MODELS=(
    "llama3_8b_lora_sft_20epochs"
    "llama3_8b_lora_sft_5epochs"
    "llama3_8b_lora_sft_3epochs"
)

TEST_DATA="xueli_data/SciQA/project_data/test_questions.csv"
OUTPUT_DIR="results/step1_generated_text/ft"

for MODEL in "${MODELS[@]}"
do
    OUTPUT_PATH="${OUTPUT_DIR}/${MODEL}"
    echo "Running model: $MODEL"
    python generate_sparql_mps.py "models/$MODEL" "$TEST_DATA" "$OUTPUT_PATH"
done
