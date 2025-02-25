import os
import sys
import time
import pandas as pd # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore


def load_model(local_model_path):
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)
    print("Model loaded")
    return model, tokenizer


def generate_sparql(question, question_id, context, model,tokenizer, output_dir):

    # question = "Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?"
    # question_id = "Q1"

    prompt = f"""
    The Open Research Knowledge Graph (ORKG) is a semantic knowledge graph designed to represent, 
    compare, and retrieve scholarly contributions. Given a natural language question in English, your task 
    is to generate the corresponding SPARQL query to this question. The generated SPARQL query should be 
    able to query the ORKG, getting correct answer to the input question. 
    Input Question: {question}
    You are also provided with the following context as background information for linking the correct entities or properties in the ORKG.

    context begins---------
    {context}
    context ends---------

    Output only the SPARQL query, no other free text or explanations.
    Output SPARQL Query:
    """

    # prompt = f"""
    # The Open Research Knowledge Graph (ORKG) is a semantic knowledge graph designed to represent, 
    # compare, and retrieve scholarly contributions. Given a natural language question in English, your task 
    # is to generate the corresponding SPARQL query to this question. The generated SPARQL query should be 
    # able to query the ORKG, getting correct answer to the input question.
    # Give me only the SPARQL query, no other text.
    # Input Question: {question}
    # Output SPARQL Query:
    # """

    # print("**Context:\n", context)
    # print("**Question:\n", question)
    # print("**Prompt:\n", prompt)


    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=False)
    attention_mask = inputs['attention_mask']
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Generate text
    print("Generating SPARQL query...")
    gen_tokens = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_new_tokens=1024)
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # see this reference for not include the input text in the generated text: https://github.com/huggingface/transformers/issues/17117
    generated_text = tokenizer.batch_decode(gen_tokens[:, inputs["input_ids"].shape[1]:])[0]
    print("SPARQL query generated successfully")

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
    print("Starting...............................")
    if len(sys.argv) != 5:
        print("Usage: python ft_rag_generate_sparql.py <local_model_path> <input_file> <context_file> <output_dir>")
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
    print("Generating SPARQL queries...")
    for i in range(len(data)):
        start = time.time()
        # Load the context file in txt format as a string 
        print(f"Loading context from {context_file}")
        with open(f"{context_file}/{data['id'][i]}.txt", 'r') as f:
            context = f.read()
        generate_sparql(data['question'][i], data['id'][i], context, model, tokenizer, output_dir)
        print(f"Question {i+1}-{data['id'][i]} done")
        end = time.time()
        print(f"Time taken: {end-start} seconds")
        print("------------------------------------------------")
    end = time.time()
    print(f"Time taken: {end-begin}")


if __name__ == "__main__":
    main()



# Run the script
# python ft_rag_generate_sparql.py saves/Llama-3.2-3B-Instruct/lora/train_2024-12-12-13-32-24  xueli_data/test_questions.csv results/rag_groq results/generated_text_ft_rag/Llama-3.2-3B-Instruct


    

