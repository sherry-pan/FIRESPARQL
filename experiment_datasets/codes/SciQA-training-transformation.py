import json


# Load the json file
def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


# Create the output json from the input json data
def create_output(data, output_file):
    output_json = [
    {
        "instruction": "The Open Research Knowledge Graph (ORKG) is a semantic knowledge graph designed to represent, compare, and retrieve scholarly contributions. Given a natural language question, your task is to generate the corresponding SPARQL query that can be used to query the ORKG for the correct answer. \nInput Format: \n    Question: A natural language question related to the ORKG.\nOutput Format:\n    SPARQL Query: The corresponding SPARQL query to retrieve the answer from the ORKG.",
        "input": question["question"]["string"],
        "output": question["query"]["sparql"]
    }
    for question in data["questions"]
]
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)


# Load the json data
data = load_data('SciQA-dataset/train/questions.json')

# Create the output json
create_output(data, 'data/SciQA-training-data-for-finetuning.json')

