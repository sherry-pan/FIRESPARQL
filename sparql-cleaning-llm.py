import os
import re
import sys
import pandas as pd
from openai import OpenAI 
from dotenv import load_dotenv


def get_files_in_folder(folder_path):
    """
    Get a list of all files in a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list: List of file names.
    """
    files = []
    try:
        for entry in os.listdir(folder_path):
            entry_path = os.path.join(folder_path, entry)
            # Check if the entry is a file and check if it is a text file
            if os.path.isfile(entry_path) and entry.endswith('.txt'):
                files.append(entry)
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return files

def get_completion(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "developer", "content": "You are an expert in SPARQL query."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content


def clean_sparql(generated_text_path, sparql_folder):
    """
    Clean the generated SPARQL queries.
    """
    # create a folder to save the cleaned sparql if it does not exist
    os.makedirs(sparql_folder, exist_ok=True)

    i = 0
    for file in get_files_in_folder(generated_text_path):
        i += 1
        print(f"Processing file {i} of {len(get_files_in_folder(generated_text_path))}: {file}")
        file_path = os.path.join(generated_text_path, file)
        with open(file_path, 'r') as f:
            question_sparql = f.read()
            prompt = f"Given a question and its corresponding SPARQL query, there might be some error in the SPARQL query such as missing blank space between variable names, unnecessary repetition and so on. Please clean the SPARQL and give me the cleaned SPARQL. Only output the SPARQL, no other text.\n{question_sparql}"
            sparql_query= get_completion(prompt)
            # print(f"genterated sparql: {sparql_query}")
            new_file_path = os.path.join(sparql_folder, f"{file}")
            with open(new_file_path, 'w') as new_file:
                new_file.write(sparql_query)
            print(f"Cleaned SPARQL query saved to {new_file_path}")
            
    return None


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()  # Load variables from .env file
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key = api_key)

    # Path to the folder containing the generated SPARQL queries and the cleaned SPARQL queries
    generated_text_path = sys.argv[1]
    clean_sparql_folder = sys.argv[2]
    clean_sparql(generated_text_path, clean_sparql_folder)
    








