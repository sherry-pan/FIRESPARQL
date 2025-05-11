# evaluate_em.py

import pandas as pd
import argparse
import os

def process_sparql_result(raw_result: str) -> set:
    if not isinstance(raw_result, str):
        return set()

    lines = raw_result.strip().split('\n')
    data_lines = lines[1:] if lines and lines[0].startswith('?') else lines

    result_set = {
        tuple(col.split('^^')[0] if '^^' in col else col for col in line.strip().split('\t'))
        for line in data_lines if line.strip()
    }
    return result_set


def compute_em(row) -> int:
    return int(process_sparql_result(row['message']) == process_sparql_result(row['gt_message']))


def get_exact_match(file: str, model_id: str, combined_csv: str):
    df = pd.read_csv(file)
    total_queries = len(df)

    df_success = df[df['status'] == 'SUCCESS'].copy()
    success_queries = len(df_success)

    df_success['EM'] = df_success.apply(compute_em, axis=1)

    avg_em_success = df_success['EM'].mean()
    avg_em_total = df_success['EM'].sum() / total_queries

    print(f"[{model_id}] Total: {total_queries}, Success: {success_queries}, EM_SUCCESS: {avg_em_success:.4f}, EM_ALL: {avg_em_total:.4f}")

    # Save successful EM=1 question IDs
    success_ids_file = f"success_ids_{model_id}.txt"
    df_success[df_success['EM'] == 1]['question_id'].to_csv(success_ids_file, index=False, header=False)

    # Append summary
    summary_df = pd.DataFrame([{
        "Model ID": model_id,
        "Total Queries": total_queries,
        "Successful Queries": success_queries,
        "Avg EM (SUCCESS)": round(avg_em_success, 4),
        "Avg EM (ALL)": round(avg_em_total, 4),
    }])
    if os.path.exists(combined_csv):
        summary_df.to_csv(combined_csv, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(combined_csv, mode='w', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to sparql_results.csv")
    parser.add_argument("--model_id", required=True, help="Model identifier (e.g., llama3_8b_lora_sft_3epochs_round1)")
    parser.add_argument("--output_csv", default="combined_sparql_eval_summary.csv", help="Combined summary CSV file")
    args = parser.parse_args()

    get_exact_match(args.file, args.model_id, args.output_csv)
