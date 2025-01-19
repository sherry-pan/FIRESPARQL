### Abstract
Scholarly knowledge graphs (SKG) represent the bibliographic metadata and scientific elements such as research problems, theories, approaches, evaluations. Question answering (QA) over SKGs demonstrates significant challenges due to the intricate nature of scholarly data and the complex structure of SKGs. The task of QA over SKGs usually takes a natural language question (NLQ) as the input and generates a corresponding SPARQL query to determine its correctness[1,2].
The emergence of large language models (LLMs) has inspired a growing body of research exploring their potential to address the challenges of QA over SKGs. However, LLMs face limitations when handling KG-specific parsing due to their lack of direct access to entities within the knowledge graph and insufficient understanding of the ontological schema, particularly for low-resource SKGs like the Open Research Knowledge Graph (ORKG). This results in suboptimal performance for QA tasks over SKGs. Insights from a pilot experiment using GPT-4 to generate SPARQL queries for handcrafted (NLQs in the SciQA Benchmark) revealed two major categories of errors in this task: semantic inaccuracies and structural inconsistencies.

Semantic inaccuracies occur when LLMs fail to link the correct properties and entities in ORKG, despite generating SPARQL queries with correct structure. Our observations reveal that LLMs tend to rely on examples provided in the few-shot learning process to generate the correct structure for a certain type of questions, but often struggle with linking the correct properties and entities because LLMs do not learn the content of the ORKG. 
We propose a RAG approach to generate the top k candidate properties or entities from ORKG based on the properties and entities mentioned in the NLQs, for LLMs to use as a context while generating the SPARQL queries.
Structural inconsistencies arise due to LLMs’ lack of ontological knowledge of the ORKG, leading to errors in query structure, such as missing or abundant links (triples). We suggest that fine-tuning LLMs with ontological information from the ORKG can help address these structural issues, allowing the model to generate more accurate queries with appropriate multi-hop relations. We proposed to address these problems by fine-tuning LLMs with two different datasets: 1) the NL-SPARQL pairs in SciQA benchmark dataset and 2) the triples in ORKG.
Additionally, we highlight the limitations of traditional machine translation evaluation metrics like BLEU and ROUGE, which rely on n-gram token overlap and fail to detect semantic issues in generated queries. These evaluation metrics lead to high scores despite low execution accuracy when queries contain incorrect properties or entities. To address this, we propose a more nuanced metric to evaluate the generated SPARQL queries, considering both structural correctness and semantic accuracy.

We conducte experiments on the SciQA Benchmark dataset and compare our results with state-of-the-art approaches.

References:

[1] iang, L., Yan, X., Usbeck, R.: A structure and content prompt-based method for knowledge graph question answering over scholarly data. In: QALD/SemREC@ISWC (2023)
[2] Taffa, T.A., Usbeck, R.: Leveraging llms in scholarly knowledge graph question answering. In: QALD/SemREC@ ISWC (2023)



### Analysis of the generated SPARQL queries to handcrafted questions
https://docs.google.com/spreadsheets/d/1aAC9POjjjmql8HPK8a3lFUykZiGa0sa1dv2vOjzZqpQ/edit?usp=sharing


### Methodology
![QAoverSKGs](https://github.com/user-attachments/assets/3ce7e851-e62a-4b70-8e9e-0002610d5ac5)



### Preliminary results
