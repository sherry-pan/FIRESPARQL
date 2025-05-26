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
        "instruction": "The dblp Knowledge Graph (dblp KG) is a fully semantic view on all the data and relationships that you can find in the dblp computer science bibliography. \nGiven a natural language question, your task is to generate the corresponding SPARQL query that can be used to query the dblp KG for the correct answer. \nInput Format: \n    Question: A natural language question related to the dblp KG.\nOutput Format:\n    SPARQL Query: The corresponding SPARQL query to retrieve the answer from the dblp KG.",
        "input": question["question"]["string"],
        "output": question["query"]["sparql"]
    }
    for question in data["questions"]
]
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)


# Load the json data
data = load_data('xueli_data/dblp/train/questions.json')

# Create the output json
create_output(data, 'xueli_data/dblp/project_data/dblp_training_data4ft.json')

