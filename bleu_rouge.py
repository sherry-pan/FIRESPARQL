import os
import re
import sys
import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from dotenv import load_dotenv

def list_txt_files(folder):
    try:
        return [f for f in os.listdir(folder) if f.endswith('.txt') and os.path.isfile(os.path.join(folder, f))]
    except FileNotFoundError:
        print(f"Warning: Folder '{folder}' not found.")
        return []

def clean_sparql(text):
    text = text.replace('```sparql', '').replace('```', '').strip()
    return re.sub(r'PREFIX.*\n', '', text)

def bleu_score(ref, hyp):
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref.split()], hyp.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

def rouge_scores(ref, hyp):
    return Rouge().get_scores(hyp, ref, avg=True)

def parse_folder_metadata(folder_name):
    # Assumes format: model_lora_sft_epochs_round (e.g., llama3_8b_lora_sft_3epochs_round1)
    parts = folder_name.split('_lora_sft_')
    if len(parts) != 2:
        return '', '', ''
    model = parts[0]
    try:
        epoch, round_ = parts[1].split('_')
    except ValueError:
        epoch, round_ = '', ''
    return model, epoch, round_

def evaluate(folder, ground_truth_df):
    files = list_txt_files(folder)
    if not files:
        return None

    bleu_scores, rouge_1, rouge_2, rouge_l = [], [], [], []

    for file in files:
        qid = file.split('.')[0]
        if qid not in ground_truth_df.index:
            continue
        with open(os.path.join(folder, file), 'r') as f:
            generated = clean_sparql(f.read())
        reference = ground_truth_df.loc[qid, 'query']

        bleu = bleu_score(reference, generated)
        rouge = rouge_scores(reference, generated)

        bleu_scores.append(bleu)
        rouge_1.append(rouge['rouge-1']['f'])
        rouge_2.append(rouge['rouge-2']['f'])
        rouge_l.append(rouge['rouge-l']['f'])

    if not bleu_scores:
        return None

    avg_metrics = {
        'BLEU-4': round(sum(bleu_scores) / len(bleu_scores), 2),
        'ROUGE-1': round(sum(rouge_1) / len(rouge_1), 2),
        'ROUGE-2': round(sum(rouge_2) / len(rouge_2), 2),
        'ROUGE-L': round(sum(rouge_l) / len(rouge_l), 2)
    }

    folder_name = os.path.basename(folder.rstrip('/'))
    model, epoch, round_ = parse_folder_metadata(folder_name)
    avg_metrics.update({
        'Model': model,
        'Epoch': epoch,
        'Round': round_,
        'Folder': folder_name
    })
    return avg_metrics

def main():
    if len(sys.argv) < 3:
        print("Usage: python bleu_rouge.py <output_csv> <folder1> [<folder2> ...]")
        sys.exit(1)

    load_dotenv()
    output_csv = sys.argv[1]
    folders = sys.argv[2:]

    ground_truth_df = pd.read_csv('xueli_data/sciqa/project_data/test_questions.csv').set_index('id')
    results = []

    for folder in folders:
        print(f"Evaluating folder: {folder}")
        metrics = evaluate(folder, ground_truth_df)
        if metrics:
            results.append(metrics)
        else:
            print(f"Skipped folder '{folder}' (no valid SPARQL files or IDs)")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

if __name__ == '__main__':
    main()
