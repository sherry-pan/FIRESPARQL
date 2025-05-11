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


def generate_sparql(question, question_id, context, model,tokenizer, output_dir):

    # question = "Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?"
    # question_id = "Q1"

    prompt = f"""
You are an expert in querying the Open Research Knowledge Graph (ORKG), a semantic knowledge graph for scholarly knowledge.

Your task is to generate an accurate SPARQL query that retrieves the answer to the given natural language question.

The query should:
- Accurately reflect the intent of the question.
- Use the correct URIs of properties and entities from ORKG.
- Return only the relevant result variables.
- Be executable directly on the ORKG SPARQL endpoint.

Input Question:
{question}

You are provided with the following context that contains relevant properties and entities, extracted via retrieval-augmented generation. Use this context to help match the correct URIs.

Context:
---------
{context}
---------

Output only the SPARQL query. Do not include any explanation, comments, or additional text.

SPARQL Query:
"""
  
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
    attention_mask = inputs['attention_mask']
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Generate text using the model
    gen_tokens = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_new_tokens=1024)
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # see this reference for not include the input text in the generated text: https://github.com/huggingface/transformers/issues/17117
    generated_text = tokenizer.batch_decode(gen_tokens[:, inputs["input_ids"].shape[1]:])[0]

    #save the question and generated text to a file， ensure the directory exists
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
    print("Starting...............................")
    if len(sys.argv) != 5:
        print("Usage: python generate_sparql_rag_mps.py <local_model_path> <input_file for the questions> <context_file from the rag> <output_dir>")
        sys.exit(1)

    local_model_path = sys.argv[1]
    input_file = sys.argv[2]
    context_file = sys.argv[3]
    output_dir = sys.argv[4]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"Loading model from {local_model_path}")
    model, tokenizer = load_model(local_model_path)
    print(f"Loading questions from {input_file}")
    data = pd.read_csv(input_file)
    begin = time.time()
    for i in range(len(data)):
        start = time.time()
        # Load the context file in txt format as a string 
        print(f"Loading context from {context_file}")
        with open(f"{context_file}/{data['id'][i]}.txt", 'r') as f:
            context = f.read()
        print(f"Generating SPARQL query for question {i+1}-{data['id'][i]}\n")
        generate_sparql(data['question'][i], data['id'][i], context, model, tokenizer, output_dir)
        print(f"Question {i+1}-{data['id'][i]} done")
        end = time.time()
        print(f"Time taken: {end-start} seconds")
        print("------------------------------------------------")
    end = time.time()
    # change the time to hours, minutes and seconds
    hours, rem = divmod(end-begin, 3600)
    minutes, seconds = divmod(rem, 60)
    print("------------------------------------------------)")
    print(f"Total time taken: {hours:.0f} hours, {minutes:.0f} minutes and {seconds:.0f} seconds")


if __name__ == "__main__":
    main()


# Run the script
# python ft_rag_generate_sparql.py saves/Llama-3.2-3B-Instruct/lora/train_2024-12-12-13-32-24  xueli_data/test_questions.csv results/rag_groq results/generated_text_ft_rag/Llama-3.2-3B-Instruct


    

