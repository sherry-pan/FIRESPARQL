#!/bin/bash

# --------- for fine-tuned models -----------------
# OUTPUT_CSV="bleu_rouge_results_ft.csv"
# BASE_DIR="results/step2_clean_sparql/ft"

# MODELS=(
#     "llama3.2_3b"
#     "llama3_8b"
# )

# EPOCHS=(
#     "3epochs" 
#     "5epochs"
#     "7epochs"
#     "10epochs"
#     "15epochs"
#     "20epochs"
#     )

# ROUNDS=(
#     "round1"
#     "round2"
#     "round3"
# )

# CMD="python bleu_rouge.py $OUTPUT_CSV"

# for model in "${MODELS[@]}"; do
#   for epoch in "${EPOCHS[@]}"; do
#     for round in "${ROUNDS[@]}"; do
#       FOLDER="${BASE_DIR}/${model}_lora_sft_${epoch}_${round}"
#       if [ -d "$FOLDER" ]; then
#         CMD+=" $FOLDER"
#       else
#         echo "Skipping missing folder: $FOLDER"
#       fi
#     done
#   done
# done

# eval $CMD



# --------- for vanilla models -----------------

OUTPUT_CSV="results/step2_clean_sparql/bleu_rouge_results_vanilla_rag.csv"
BASE_DIR="results/step2_clean_sparql/vanilla_rag"

MODELS=(
    "llama-3.2-3b-Instruct"
    "llama-3-8b-Instruct"
    # "llama3.2_3b_lora_sft_20epochs"
    # "llama3_8b_lora_sft_15epochs"
)


ROUNDS=(
    "round1"
    "round2"
    "round3"
)

CMD="python bleu_rouge.py $OUTPUT_CSV"

for model in "${MODELS[@]}"; do
  for round in "${ROUNDS[@]}"; do
    FOLDER="${BASE_DIR}/${model}_${round}"
        if [ -d "$FOLDER" ]; then
        CMD+=" $FOLDER"
        else
        echo "Skipping missing folder: $FOLDER"
        fi
  done
done

eval $CMD
