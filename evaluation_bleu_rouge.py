import os
import re
import sys
import pandas as pd
# pip install nltk scikit-learn rouge-score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
            if os.path.isfile(entry_path):
                files.append(entry)
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return files


def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score between reference and hypothesis.
    """
    smooth = SmoothingFunction().method1
    # BLEU score with 4-gram smoothing: BLEU-4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smooth, weights=(0.25, 0.25, 0.25, 0.25))


def calculate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    """
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores


def evaluation(sparql_folder, ground_truth_csv_file='xueli_data/test_questions.csv'):
    """
    Evaluate the generated SPARQL queries using BLEU and ROUGE scores.
    """
    # Load the CSV file with the test questions
    df = pd.read_csv(ground_truth_csv_file)

    bleu_score_list = []
    rouge_1_score_list = []
    rouge_2_score_list = []
    rouge_l_score_list = []

    # Loop through each file in the clean_sparql folder
    sparql_list = get_files_in_folder(sparql_folder)

    # Attension: not all the file in the test_questions.csv are in the clean_sparql folder
    for file in sparql_list:
        # Load the content of the text file
        with open(os.path.join(sparql_folder, file), 'r') as f:
            # remove the ```sparql and ``` from the generated sparql
            generated_sparql = f.read().replace('```sparql', ' ').replace('```', ' ').strip()
            # remove the prefix namespace if it exists
            generated_sparql = re.sub(r'PREFIX.*\n', '', generated_sparql)
            # print(f'generated_sparql: {generated_sparql}')
            question_id = file.split('.')[0]
            # print(f'generated_sparql: {generated_sparql}')
        # Get the ground truth SPARQL query
        sparql = df[df['id'] == question_id]['query'].values[0]

        # Calculate BLEU and ROUGE scores
        bleu_score = calculate_bleu(sparql, generated_sparql)
        rouge_scores = calculate_rouge(sparql, generated_sparql)
        rouge_1 = rouge_scores['rouge-1']['f']
        rouge_2 = rouge_scores['rouge-2']['f']
        rouge_l = rouge_scores['rouge-l']['f']

        # Append the scores to the lists
        bleu_score_list.append(bleu_score)
        rouge_1_score_list.append(rouge_1)
        rouge_2_score_list.append(rouge_2)
        rouge_l_score_list.append(rouge_l)

    # calculate the average scores
    avg_bleu_score = sum(bleu_score_list) / len(bleu_score_list)
    avg_rouge_1_score = sum(rouge_1_score_list) / len(rouge_1_score_list)
    avg_rouge_2_score = sum(rouge_2_score_list) / len(rouge_2_score_list)
    avg_rouge_l_score = sum(rouge_l_score_list) / len(rouge_l_score_list)

    return avg_bleu_score, avg_rouge_1_score, avg_rouge_2_score, avg_rouge_l_score


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluation.py <sparql_folder>")
        sys.exit(1)
    # Load environment variables
    load_dotenv()  # Load variables from .env file
    sparql_folder = sys.argv[1]
    avg_bleu_score, avg_rouge_1_score, avg_rouge_2_score, avg_rouge_l_score = evaluation(sparql_folder)
    print(f"Average BLEU4 Score: {avg_bleu_score:.2f}")
    print(f"Average ROUGE-1 Score: {avg_rouge_1_score:.2f}")
    print(f"Average ROUGE-2 Score: {avg_rouge_2_score:.2f}")
    print(f"Average ROUGE-L Score: {avg_rouge_l_score:.2f}")


if __name__ == '__main__':
    main()
    

