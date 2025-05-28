### FIRESPARQL, 

# FIRESPARQL: a LLM-based framework for SPARQL Query Generation over Scholarly Knowledge Graphs

This repository contains the code, model, and data associated with our paper:

> **FIRESPARQL: A LLM-based Framework for SPARQL Query Generation over Scholarly Knowledge Graphs**  
> 📎 *Under Review 
> ✨ Fine-tuned LLaMA-3-8B-Instruct via LoRA achieves state-of-the-art performance on NLQ-to-SPARQL generation for the ORKG/SciQA dataset.  

## 🔍 Overview

FIRESPARQL explores the effectiveness of domain-specific fine-tuning for generating executable and accurate SPARQL queries from natural language questions. Our experiments demonstrate that:

- Fine-tuning LLMs (via LoRA) on NLQ–SPARQL pairs significantly improves generation and execution accuracy.
- The fine-tuned **LLaMA-3-8B-Instruct** model (trained for 15 epochs) outperforms both zero-shot and one-shot baselines.
- Incorporating naive RAG modules **does not** improve performance and can introduce noise.
- One-shot prompting using SentenceBERT-based similarity also shows competitive performance in the absence of fine-tuning.

## 📊 Performance on the llama-3-8b-instruct model

| Method            | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | RelaxedEM (All) |
|-------------------|--------|---------|---------|---------|------------------|
| Fine-tuned LLaMA3-8B (15ep) | 0.77   | 0.91    | 0.86    | 0.90    | 0.85   |
| One-shot (SentenceBERT)     | 0.58   | 0.81    | 0.73    | 0.78    | 0.40   |
| Zero-shot (no fine-tuning)  | 0.03   | 0.39    | 0.18    | 0.38    | 0      |

## 🏗️ Model

We release the **best performing model**, fine-tuned LLaMA-3-8B-Instruct (15 epochs), on Hugging Face:

👉 [Meta-Llama-3-8b-ft4sciqa](https://huggingface.co/Sherry791/Meta-Llama-3-8B-Instruct-ft4sciqa)

### Fine-Tuning Details

- Base model: `LLaMA-3-8B-Instruct`
- Fine-tuning technique: LoRA
- Epochs: 15
- Training data: training set from SciQA benchmark
- Test data: test set from SciQA benchmark (513 natural language questions + corresponding SPARQL queries )

## SPARQL execution details
- Using Qlever for sparql execution

## 🧪 Reproducing Results

### Requirements
_Coming soon or to be added here..._

---

## 📁 Code Structure

```plaintext
.
├── codes/                                 # Core scripts for generation, evaluation, and cleaning
│   ├── accumulate_exact_match.py          # Aggregates exact match scores for three runs
│   ├── bleu_rouge.py                      # Computes BLEU and ROUGE metrics
│   ├── exact_match.py                     # Computes exact match scores
│   ├── generate_context_rag.py            # Retrieves context using RAG from ORKG
│   ├── generate_sparql_cuda.py            # Generates SPARQL queries using CUDA
│   ├── generate_sparql_mps.py             # Generates SPARQL queries using Apple MPS
│   ├── generate_sparql_one_shot_cuda.py   # One-shot SPARQL generation using CUDA
│   ├── generate_sparql_rag_cuda.py        # SPARQL generation with RAG using CUDA
│   ├── generate_sparql_rag_mps.py         # SPARQL generation with RAG using MPS
│   ├── merge_sparql.ipynb                 # Merges SPARQL results with ground truth for error analysis
│   ├── ploting.ipynb                      # Visualization of evaluation results
│   ├── readme.md                          # Additional documentation on the codes
│   ├── run_all_bleu_rouge.sh              # Script to run all BLEU/ROUGE evaluations on snellius
│   ├── run_all_cleaning.sh                # Script to clean generated SPARQL on snellius
│   ├── run_all_exact_match.sh             # Script to compute exact match scores on snellius
│   ├── run_all_generate.sh                # Script to generate SPARQL in batch on snellius
│   └── sparql-cleaning-llm.py             # LLM-based SPARQL query cleaning
│
├── experiment_datasets/                   # Dataset directory
│   ├── codes/                             # Code-related preprocessing the dataset
│   ├── dblp/                              # DBLP dataset
│   └── sciqa/                             # SciQA benchmark data
│
├── results/                               # All output results
│   ├── context_from_rag/                  # Retrieved context from ORKG using RAG
│   ├── step1_generated_text/              # Output of generated SPARQL queries
│   ├── step2_clean_sparql/                # Cleaned/generated SPARQL files
│   ├── step3_sparql_running_against_qlever/ # Results from QLever SPARQL execution
│   ├── step4_accumulated_success_metrics/ # Aggregated metrics (e.g., RelaxedEM)
│   └── step5_error_analysis/              # Failed cases and syntax error breakdown
│
├── .gitignore                             # Git ignore rules
└── README.md                              # Project overview and documentation

