import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Load CSV files
test_df = pd.read_csv("/Users/sherrypan/GitHub/LLaMa-Factory/xueli_data/sciqa/project_data/test_questions.csv")
train_df = pd.read_csv("/Users/sherrypan/GitHub/LLaMa-Factory/xueli_data/sciqa/project_data/train_questions.csv")

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, and good quality

# Encode all questions using Sentence-BERT
print("Encoding train questions...")
train_embeddings = model.encode(train_df["question"].tolist(), convert_to_tensor=True)

print("Encoding test questions and finding most similar train question...")
results = []

# Iterate through each test question
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    test_id = row["id"]
    test_question = row["question"]
    test_query = row["query"]

    # Encode test question
    test_embedding = model.encode(test_question, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = cosine_similarity([test_embedding.cpu().numpy()], train_embeddings.cpu().numpy())[0]

    # Find index of best match
    best_match_idx = np.argmax(cosine_scores)

    # Get best matched train question and query
    best_train_question = train_df.iloc[best_match_idx]["question"]
    best_train_query = train_df.iloc[best_match_idx]["query"]

    results.append({
        "id": test_id,
        "test_question": test_question,
        "test_query": test_query,
        "best_train_question": best_train_question,
        "train_query": best_train_query
    })

# Save results to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("/Users/sherrypan/GitHub/LLaMa-Factory/xueli_data/sciqa/project_data/most_similar_questions.csv", index=False)
print("Saved to most_similar_questions.csv")
