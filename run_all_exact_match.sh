#!/bin/bash

# --------------->for fine-tuned models<----------------
# # Output file
# OUTPUT_CSV="exact_match_summary_ft.csv"
# > $OUTPUT_CSV  # Empty it first

# # Define model variants, epochs, and rounds
# models=(
#     "llama3.2_3b"
#     "llama3_8b"
# )

# epochs_list=(
#     "3epochs" 
#     "5epochs"
#     "7epochs"
#     "10epochs"
#     "15epochs"
#     "20epochs"
#     )

# rounds=(
#     "round1"
#     "round2"
#     "round3"
# )

# # Loop through each combination
# for model in "${models[@]}"; do
#   for epochs in "${epochs_list[@]}"; do
#     for round in "${rounds[@]}"; do
#       model_id="${model}_lora_sft_${epochs}_${round}"
#       csv_path="results/step3_sparql_running_against_orkg/ft/${model_id}/sparql_results.csv"

#       if [ -f "$csv_path" ]; then
#         echo "Processing $csv_path ..."
#         python exact_match.py --file "$csv_path" --model_id "$model_id" --output_csv "$OUTPUT_CSV"
#       else
#         echo "⚠️  File not found: $csv_path"
#       fi
#     done
#   done
# done



# --------------->for vanilla models<----------------
OUTPUT_CSV="exact_match_summary_vanilla.csv"
> $OUTPUT_CSV  # Empty it first

# Define model variants, epochs, and rounds
models=(
    "llama-3.2-3b-Instruct"
    "llama-3-8b-Instruct"
)

rounds=(
    "round1"
    "round2"
    "round3"
)

# Loop through each combination
for model in "${models[@]}"; do
  for round in "${rounds[@]}"; do
    model_id="${model}_${round}"
    csv_path="results/step3_sparql_running_against_orkg/vanilla/${model_id}/sparql_results.csv"

    if [ -f "$csv_path" ]; then
      echo "Processing $csv_path ..."
      python exact_match.py --file "$csv_path" --model_id "$model_id" --output_csv "$OUTPUT_CSV"
    else
      echo "⚠️  File not found: $csv_path"
    fi
  done
done
