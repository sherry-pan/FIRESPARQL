import warnings
warnings.filterwarnings('ignore')

import os
import re
import time
import json
import chromadb
import time
import pandas as pd

from dotenv import load_dotenv

from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def rdf_data_indexing_to_chromadb(model, api_key, temperature,embed_model, rag_files, collection_name):
    """
    Indexing the RDF data to ChromaDB
    """
    llm = Groq(model= model, api_key=api_key)
    # create client and a new collection
    chroma_client = chromadb.EphemeralClient()
    # clear past collection
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    # create new collection
    chroma_collection = chroma_client.create_collection(collection_name)

    # define embedding function
    embed_model = HuggingFaceEmbedding(model_name=embed_model)

    # load documents from a specific path(file or folders)
    print(f"Loading data from {rag_files}...")
    documents = SimpleDirectoryReader(rag_files).load_data()
    print(f"Data loaded successfully!")
    
    # set up ChromaVectorStore and load in data
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Indexing data...")
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    print("Data indexed successfully!")

    Settings.llm = Groq(model = model, temperature=temperature)
    query_engine = index.as_query_engine(llm)
    print("Query Engine created successfully!")
    return query_engine


def get_response_from_llm(query_engine, user_query):
    """""
    Get response from LLM
    """
    response = query_engine.query(user_query)
    return response


def read_test_questions(question_file):
    """
    Read 513 test questions from sciqa benchmark dataset
    """
    # data = pd.read_csv("xueli_data/test_questions.csv")
    data = pd.read_csv(question_file)
    questions = data['question'].tolist()
    question_ids = data['id'].tolist()
    return questions, question_ids

def main(query_engine, user_query):
    """
    main
    """
    response = get_response_from_llm(query_engine, user_query)
    return response


if __name__ == "__main__":

    load_dotenv()

    model = "llama-3.3-70b-versatile"
    api_key = os.getenv('GROQ_API_KEY')
    print(f"api_key: {api_key}")
    temperature = 0.5
    embed_model = "BAAI/bge-small-en"
    rag_files = "xueli_data/rdf-dump/"
    collection_name = 'qa-kg'

    # rdf data indexing
    print("Indexing RDF data to ChromaDB...")
    starttime = time.time()
    query_engine = rdf_data_indexing_to_chromadb(model, api_key, temperature,embed_model, rag_files, collection_name)
    endtime = time.time()
    print(f"Time taken for indexing: {endtime - starttime} seconds")
    print("RDF data indexed successfully!\n")

    questions, question_ids = read_test_questions("xueli_data/test_questions.csv")
    print(f"Total number of questions: {len(questions)}")
    print(f"The first question is: {questions[0]}")
    print(f"The first question id is: {question_ids[0]}")

    for i in range(0, len(questions)):
        question = questions[i]
        question_id = question_ids[i]
        user_query = f"""
                given a natual language question, your task is to retrieve the top 5 similar candidate entities or perperties 
                from the given ORKG rdf data dump in turtle format. To finish the task, you need to implement the following steps:
                1. Extract the entities and properties mentioned in the question
                2. Retrieve the top 10 similar entities or properties from the given rdf data dump based on cosine similarity scores
                3. Return the top 10 similar entities or properties
                The output should be a list of dictionaries in JSON format, where each dictionary contains the following keys:
                - uri: the URI of the entity or property
                - label: the label of the entity or property
                - score: the similarity score between the entity or property and the question
                The input question is: {question}
                """
        print(f"User Query: , {user_query}")
        print("Getting response from LLM...\n")
        response = main(query_engine, user_query)
        print(f"Response for question {question_id} is: {response}")
        print("------------------------------------------------------------")
        # save the response to a txt file
        output_dir = "results/rag_groq"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/{question_id}.txt", "w") as f:
            f.write(str(response))
        print(f"Response for question {question_id} is saved successfully!")
        print("------------------------------------------------------------")
    print("All questions are processed")