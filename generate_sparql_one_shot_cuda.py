import os
import sys
import time
import torch
import pandas as pd # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        # device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )
    return model, tokenizer


def generate_sparql(question, question_id, best_train_question, best_train_sparql_query, model,tokenizer, output_dir):

    # question = "Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?"
    # question_id = "Q1"

    prompt = f"""
    The Open Research Knowledge Graph (ORKG) is a semantic knowledge graph designed to represent, 
    compare, and retrieve scholarly contributions.

    Your task is to generate a SPARQL query that correctly answers a given natural language question in English.
    The query should be executable on the ORKG SPARQL endpoint.

    Below is one example to guide you:

    ---
    Example Question:
    {best_train_question}

    Example SPARQL Query:
    {best_train_sparql_query}
    ---

    Now generate the SPARQL query for the following question.

    Input Question:
    {question}

    Give me only the SPARQL query, with no other text.

    SPARQL Query:
    """

    prompt_template = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)

    # Generate text
    gen_tokens = model.generate(inputs["input_ids"], max_length=512, temperature=0.7).to(device)
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # see this reference for not include the input text in the generated text: https://github.com/huggingface/transformers/issues/17117
    generated_text = tokenizer.batch_decode(gen_tokens[:, inputs["input_ids"].shape[1]:])[0]

    #save the question and generated text to a file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{question_id}.txt')

    # Save the question and generated text to a single file
    with open(output_file, 'w') as f:
        f.write("Question:\n")
        f.write(question + "\n\n")
        f.write("Generated SPARQL:\n")
        f.write(generated_text + "\n")

def main():
    local_model_path = sys.argv[1]
    # xueli_data/sciqa/project_data/most_similar_questions.csv
    input_file = sys.argv[2]
    output_dir = sys.argv[3]
    model, tokenizer = load_model(local_model_path)
    data = pd.read_csv(input_file)
    begin = time.time()
    for i in range(len(data)):
        generate_sparql(data['test_question'][i], data['id'][i], data['best_train_question'][i], data["train_query"][i], model, tokenizer, output_dir)
        print(f"Question {i+1} done")
    end = time.time()
    print(f"Time taken: {end-begin}")


if __name__ == "__main__":
    main()

# python generate_sparql_one_shot_cuda.py \
# meta-llama/llama-3.2-3b-Instruct \
# xueli_data/sciqa/project_data/most_similar_questions.csv \
# results/step1_generated_text/one_shot

