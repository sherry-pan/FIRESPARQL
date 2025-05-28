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
