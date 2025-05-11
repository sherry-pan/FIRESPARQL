
folder_path = "results/step4_accumulated_success_sparql/ft/"
# Define the filenames for the three rounds
for epoch in [3,5,7,10,15,20]:
    file_names = [
        f"success_ids_llama3.2_3b_lora_sft_{epoch}epochs_round1.txt",
        f"success_ids_llama3.2_3b_lora_sft_{epoch}epochs_round2.txt",
        f"success_ids_llama3.2_3b_lora_sft_{epoch}epochs_round3.txt"
    ]
    # Construct the full paths for each file
    file_names = [folder_path + file_name for file_name in file_names]

    # Use a set to store unique IDs
    unique_ids = set()

    # Read IDs from each file and add them to the set
    for file_name in file_names:
        with open(file_name, "r") as f:
            for line in f:
                id_ = line.strip()
                if id_:  # skip empty lines
                    unique_ids.add(id_)

    # # Write the unique IDs to a new file
    # output_file = "accumulated_unique_ids.txt"
    # with open(output_file, "w") as f:
    #     for id_ in sorted(unique_ids):  # optional: sort the IDs
    #         f.write(id_ + "\n")

    # Output the number of unique IDs
    print(f"{len(unique_ids)}")
