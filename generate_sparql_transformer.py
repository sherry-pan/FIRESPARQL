import os
import sys
import time
import pandas as pd # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore


def load_model(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    return model, tokenizer


def generate_sparql(question, question_id, model,tokenizer, output_dir):

    # question = "Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?"
    # question_id = "Q1"

    prompt = f"""
    The Open Research Knowledge Graph (ORKG) is a semantic knowledge graph designed to represent, compare, and retrieve scholarly contributions. Given a natural language question, your task is to generate the corresponding SPARQL query that can be used to query the ORKG for the correct answer. Give me only the SPARQL query, no other text.
    Input Question: {question}
    Only output the SPARQL query here. No other text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text
    gen_tokens = model.generate(inputs["input_ids"], max_length=1024, temperature=0.7)
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # see this reference for not include the input text in the generated text: https://github.com/huggingface/transformers/issues/17117
    generated_text = tokenizer.batch_decode(gen_tokens[:, inputs["input_ids"].shape[1]:])[0]

    #save the question and generated text to a file
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
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






    

