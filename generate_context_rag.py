import warnings
warnings.filterwarnings('ignore')

import os
import time
import json
import chromadb
import pandas as pd

from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def rdf_data_indexing_to_chromadb(model, api_key, temperature, embed_model, rag_files, collection_name):
    """
    Indexing the RDF data to ChromaDB
    """
    llm = Groq(model=model, api_key=api_key)
    chroma_client = chromadb.EphemeralClient()

    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    chroma_collection = chroma_client.create_collection(collection_name)
    embed_model = HuggingFaceEmbedding(model_name=embed_model)

    print(f"Loading data from {rag_files}...")
    documents = SimpleDirectoryReader(rag_files).load_data()
    print("Data loaded successfully!")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Indexing data...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    print("Data indexed successfully!")

    Settings.llm = Groq(model=model, temperature=temperature)
    query_engine = index.as_query_engine(llm)
    print("Query Engine created successfully!")
    return query_engine

def get_response_from_llm(query_engine, user_query):
    """
    Get response from LLM
    """
    response = query_engine.query(user_query)
    return response

def read_test_questions(question_file):
    """
    Read test questions from sciqa benchmark dataset
    """
    data = pd.read_csv(question_file)
    questions = data['question'].tolist()
    question_ids = data['id'].tolist()
    return questions, question_ids

def load_property_json_sample(json_path, sample_size=20):
    """
    Load and sample properties from the JSON file
    """
    with open(json_path, 'r') as f:
        properties = json.load(f)
    return json.dumps(properties[:sample_size], indent=2)

def main(query_engine, user_query):
    """
    Main query function
    """
    response = get_response_from_llm(query_engine, user_query)
    return response

if __name__ == "__main__":

    load_dotenv()

    model = "deepseek-r1-distill-llama-70b"
    api_key = os.getenv('GROQ_API_KEY')
    temperature = 0.5
    embed_model = "BAAI/bge-small-en"
    # rag file directory
    rag_files = "xueli_data/sciqa/project_data/rag_files"
    collection_name = 'qa-kg'

    print("Indexing RDF data to ChromaDB...")
    starttime = time.time()
    query_engine = rdf_data_indexing_to_chromadb(model, api_key, temperature, embed_model, rag_files, collection_name)
    endtime = time.time()
    print(f"Time taken for indexing: {endtime - starttime} seconds")
    print("RDF data indexed successfully!\n")

    questions, question_ids = read_test_questions("xueli_data/sciqa/project_data/test_questions.csv")
    print(f"Total number of questions: {len(questions)}")

    json_sample = load_property_json_sample("xueli_data/sciqa/project_data/orkg-property.json", sample_size=20)

    for i in range(len(questions)):
        question = questions[i]
        question_id = question_ids[i]

        user_query = f"""
        You are given a JSON file containing properties and their labels from the ORKG knowledge graph. 
        Your task is to identify the most relevant properties from this file that correspond to the input natural language question.

        --- JSON EXAMPLE START ---
        {json_sample}
        --- JSON EXAMPLE END ---

        Question: {question}

        Please list the most relevant properties from the JSON that could help answer this question. For each match, explain briefly why it is relevant.

        Return format:
        [
        {{
            "property_iri": "<IRI>",
            "label": "<label>",
            "reason": "<brief explanation>"
        }},
        ...
]
        """

        # print(f"\n[User Query for Question ID {question_id}]:\n{user_query}\n")
        print("Getting response from LLM...\n")
        response = main(query_engine, user_query)
        print(f"Response for question {question_id} is: {response}")
        print("------------------------------------------------------------")

        output_dir = f"results/context_from_rag/{model}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/{question_id}.txt", "w") as f:
            f.write(str(response))
        print(f"Response for question {question_id} is saved successfully!")
        print("------------------------------------------------------------")

    print("All questions are processed")
