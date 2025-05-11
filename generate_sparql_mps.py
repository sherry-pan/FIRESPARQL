import os
import sys
import time
import torch
import pandas as pd # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.mps.empty_cache()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def load_model(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.float16,
        # device_map="auto",
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )
    return model, tokenizer


def generate_sparql(question, question_id, model,tokenizer, output_dir):

    # question = "Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?"
    # question_id = "Q1"

    prompt = f"""
    The Open Research Knowledge Graph (ORKG) is a semantic knowledge graph designed to represent, 
    compare, and retrieve scholarly contributions. Given a natural language question in English, your task 
    is to generate the corresponding SPARQL query to this question. The generated SPARQL query should be 
    able to query the ORKG, getting correct answer to the input question. 
    Give me only the SPARQL query, no other text.
    Input Question: {question}
    Output SPARQL Query:
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
    input_file = sys.argv[2]
    output_dir = sys.argv[3]
    model, tokenizer = load_model(local_model_path)
    data = pd.read_csv(input_file)
    begin = time.time()
    for i in range(len(data)):
        generate_sparql(data['question'][i], data['id'][i], model, tokenizer, output_dir)
        print(f"Question {i+1} done")
    end = time.time()
    print(f"Time taken: {end-begin}")


if __name__ == "__main__":
    main()


# Run the script
# python generate_sparql_transformer.py saves/Llama-3.2-3B-Instruct/lora/train_2024-12-12-13-32-24  xueli_data/test_questions.csv results/generated_text/llama3.2_3b_lora_20250121
