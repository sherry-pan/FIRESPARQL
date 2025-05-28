# import os
# import re
# import sys
# import pandas as pd
# from openai import OpenAI 
# from dotenv import load_dotenv


# def get_files_in_folder(folder_path):
#     """
#     Get a list of all files in a folder.

#     Args:
#         folder_path (str): Path to the folder.

#     Returns:
#         list: List of file names.
#     """
#     files = []
#     try:
#         for entry in os.listdir(folder_path):
#             entry_path = os.path.join(folder_path, entry)
#             # Check if the entry is a file and check if it is a text file
#             if os.path.isfile(entry_path) and entry.endswith('.txt'):
#                 files.append(entry)
#     except FileNotFoundError:
#         print(f"The folder '{folder_path}' does not exist.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
    
#     return files

# def get_completion(prompt):
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "developer", "content": "You are an expert in SPARQL query."},
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )

#     return completion.choices[0].message.content


# def clean_sparql(generated_text_path, sparql_folder):
#     """
#     Clean the generated SPARQL queries.
#     """
#     # create a folder to save the cleaned sparql if it does not exist
#     os.makedirs(sparql_folder, exist_ok=True)

#     i = 0
#     for file in get_files_in_folder(generated_text_path):
#         i += 1
#         print(f"Processing file {i} of {len(get_files_in_folder(generated_text_path))}: {file}")
#         file_path = os.path.join(generated_text_path, file)
#         with open(file_path, 'r') as f:
#             question_sparql = f.read()
#             prompt = f"Given a question and its corresponding SPARQL query, there might be some error in the SPARQL query such as missing blank space between variable names, unnecessary repetition and so on. Please clean the SPARQL and give me the cleaned SPARQL. Only output the SPARQL, no other text.\n{question_sparql}"
#             sparql_query= get_completion(prompt)
#             # print(f"genterated sparql: {sparql_query}")
#             new_file_path = os.path.join(sparql_folder, f"{file}")
#             with open(new_file_path, 'w') as new_file:
#                 new_file.write(sparql_query)
#             print(f"Cleaned SPARQL query saved to {new_file_path}")
            
#     return None


# if __name__ == '__main__':
#     # Load environment variables
#     load_dotenv()  # Load variables from .env file
#     api_key = os.getenv("OPENAI_API_KEY")
#     client = OpenAI(api_key = api_key)

#     # Path to the folder containing the generated SPARQL queries and the cleaned SPARQL queries
#     generated_text_path = sys.argv[1]
#     clean_sparql_folder = sys.argv[2]
#     clean_sparql(generated_text_path, clean_sparql_folder)


import os
import sys
from openai import OpenAI
from dotenv import load_dotenv


def list_text_files(directory: str) -> list:
    """
    Return a list of .txt files in the given directory.

    Args:
        directory (str): Path to the folder.

    Returns:
        list: List of .txt file names.
    """
    try:
        return [f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]
    except FileNotFoundError:
        print(f"Folder '{directory}' not found.")
    except Exception as e:
        print(f"Error reading folder '{directory}': {e}")
    return []


def call_openai_cleaning_api(prompt: str, client: OpenAI) -> str:
    """
    Call the OpenAI API with a cleaning prompt.

    Args:
        prompt (str): The cleaning instruction and input SPARQL.
        client (OpenAI): Authenticated OpenAI client.

    Returns:
        str: Cleaned SPARQL query.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in SPARQL query."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def clean_and_save_sparql(input_folder: str, output_folder: str, client: OpenAI):
    """
    Clean SPARQL queries from files and save them to a new folder.

    Args:
        input_folder (str): Folder with original SPARQL text files.
        output_folder (str): Folder to save cleaned SPARQL files.
        client (OpenAI): OpenAI API client.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = list_text_files(input_folder)
    total_files = len(files)

    for idx, filename in enumerate(files, start=1):
        print(f"Processing file {idx}/{total_files}: {filename}")
        file_path = os.path.join(input_folder, filename)

        with open(file_path, 'r') as f:
            original_text = f.read()

        prompt = (
            "Given a question and its corresponding SPARQL query, there might be "
            "errors such as missing spaces between variable names, unnecessary repetition, etc. "
            "Please clean the SPARQL and return only the cleaned SPARQL (no explanation):\n\n"
            f"{original_text}"
        )

        cleaned_sparql = call_openai_cleaning_api(prompt, client)

        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'w') as out_f:
            out_f.write(cleaned_sparql)

        print(f"Saved cleaned SPARQL to: {output_path}")


def main():
    """
    Entry point for the script. Loads API key and starts cleaning process.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)

    if len(sys.argv) != 3:
        print("Usage: python script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    client = OpenAI(api_key=api_key)

    clean_and_save_sparql(input_folder, output_folder, client)


if __name__ == '__main__':
    main()

    








