import os
import re
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

def clean_context(context):
    # Remove all content between <think> and </think>, inclusive
    context = re.sub(r'<think>.*?</think>', '', context, flags=re.DOTALL)
    return context

def generate_sparql(question, question_id, context, model, tokenizer, output_dir):
    # Clean the context to remove content between <think> and </think>
    context = clean_context(context)

    #-----------------------prompt1-----------------------
    # prompt = f"""
    # You are an expert in querying the Open Research Knowledge Graph (ORKG), a semantic knowledge graph for scholarly knowledge.

    # Your task is to generate an accurate SPARQL query that retrieves the answer to the given natural language question.

    # The query should:
    # - Accurately reflect the intent of the question.
    # - Use the correct URIs of properties and entities from ORKG.
    # - Return only the relevant result variables.
    # - Be executable directly on the ORKG SPARQL endpoint.

    # Input Question:
    # {question}

    # You are provided with the following context that contains relevant properties, extracted via retrieval-augmented generation. Use this context to help match the correct URIs.

    # Context:
    # ---------
    # {context}
    # ---------

    # Output only the SPARQL query. Do not include any explanation, comments, or additional text.

    # SPARQL Query:
    # """
    #-----------------------prompt2-----------------------
    prompt = f"""
    You are an expert in querying the Open Research Knowledge Graph (ORKG), a semantic knowledge graph for scholarly contributions.

    Your task is to generate an accurate SPARQL query that answers the following natural language question:

    Question:
    {question}

    You are also provided with background context that includes potentially relevant ORKG properties retrieved via a RAG system. You may choose to use this context if it helps improve the accuracy of your query. However, if you already know how to generate the correct query, you may ignore it.

    Context (optional to use):
    ---------
    {context}
    ---------

    Guidelines:
    - Use valid SPARQL syntax and ORKG-compatible URIs.
    - Retrieve only the necessary variables to answer the question.
    - Do not include any explanations, comments, or formatting outside of the SPARQL query itself.

    Output only the SPARQL query below.

    SPARQL Query:
    """
    prompt_template = f"""
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
  
    inputs = tokenizer(prompt_template, return_tensors="pt", padding=True, truncation=False).to(device)
    attention_mask = inputs['attention_mask']
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Generate text using the model
    gen_tokens = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_new_tokens=1024).to(device)
    generated_text = tokenizer.batch_decode(gen_tokens[:, inputs["input_ids"].shape[1]:])[0]

    # Save the question and generated text to a file
    os.makedirs(output_dir, exist_ok=True)

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
        print("Usage: python generate_sparql_rag_mps.py <local_model_path> <input_file for the test questions> <context_file from the rag> <output_dir>")
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
# python generate_sparql_rag_mps.py \
#     merge_models/llama3.2_3b_lora_sft_20epochs \
#     xueli_data/sciqa/project_data/test_questions.csv \
#     results/context_from_rag/deepseek-r1-distill-llama-70b \
#     results/step1_generated_text/ft_rag/llama3.2_3b_lora_sft_20epochs


    

