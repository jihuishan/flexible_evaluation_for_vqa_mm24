# Towards Flexible Evaluation for Generative Visual Question Answering

Welcome to the repository for our work on **Flexible Evaluation for Generative Visual Question Answering (VQA)**, an oral in ACM Multimedia 2024. In this project, we aim to tackle the challenges posed by current evaluation metrics used for assessing multimodal large language models (MLLMs) in VQA tasks.

Traditional evaluation methods, such as Exact Match, often fail to account for the rich and varied responses that MLLMs can generate. To address this, we propose a **Semantically Flexible VQA Evaluator (SFVE)**, which evaluates responses based on semantic similarity rather than exact word matches. This approach allows for a more accurate and flexible evaluation of open-ended VQA responses.

## Key Features

- **SFVE Model**: A semantically-based evaluator designed to handle diverse response types, surpassing traditional methods.
- **AVE Dataset**: The **Assessing VQA Evaluators (AVE)** dataset provides human-annotated responses, enabling comprehensive analysis of different semantic evaluators.

Our goal is to provide a framework that ensures fair and consistent evaluation of generative models on VQA tasks, accommodating the complexity and richness of MLLM-generated responses. This flexible evaluation tool is useful for researchers looking to assess models in open-ended VQA tasks.


## Models Overview

We have trained two models with different sizes: **SFVE-base** and **SFVE-large**. Below are the details of each model along with their performance on our proposed AVE dataset.

| Model Name  | Backbone     | HuggingFace URL                                                | Alignment Performance (AVE) |
|-------------|--------------|----------------------------------------------------------------|-------------------|
| SFVE-base   | RoBERTa-base | [SFVE-base on HuggingFace](https://huggingface.co/Huishan/SFVE-base)  | 56.5              |
| SFVE-large  | RoBERTa-large| [SFVE-large on HuggingFace](https://huggingface.co/Huishan/SFVE-large) | 59.2              |


## How to Use

The SFVE models are designed for flexibly evaluating the generated results from MLLMs on VQA datasets. Specifically, the expected usage involves extracting sentence embeddings from both the question and the model's generated answer. By computing the cosine similarity between these embeddings, users can quantitatively assess the semantic relevance of the generated answers to the input questions. This method allows for a flexible and scalable approach to evaluate model outputs across various VQA tasks, providing an efficient metric to gauge model performance on semantic understanding and alignment.

In practice, users can input pairs of questions and generated answers into the SFVE model, extract the corresponding embeddings, and calculate their similarity to determine how closely the model's answer aligns with the question's intent. This makes the SFVE models particularly suitable for automated evaluations in scenarios where human evaluation may be costly or time-consuming.

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

# Load the model
MODEL_SIZE = 'base'
model = AutoModel.from_pretrained("Huishan/SFVE-"+MODEL_SIZE).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("Huishan/SFVE-"+MODEL_SIZE)

# Example data for sentence pairs
batch_data = {
    'sentence1': ["Question: How many birds are there? Answer: 3", "Question: What is the color of the boy's hair? Answer: yellow"],
    'sentence2': ["Question: How many birds are there? Answer: three", "Question: What is the color of the boy's hair? Answer: golden"]
}

# Tokenize the sentences
input_ids1 = tokenizer([text for text in batch_data['sentence1']], truncation=True,
                            padding='longest', max_length=128, return_tensors='pt', add_special_tokens=True).to('cuda')
input_ids2 = tokenizer([text for text in batch_data['sentence2']], truncation=True,
                            padding='longest', max_length=128, return_tensors='pt', add_special_tokens=True).to('cuda')

# Get sentence embeddings
with torch.no_grad():
    logits1 = model(**input_ids1).pooler_output
    logits2 = model(**input_ids2).pooler_output

# Compute cosine similarity
cosine_sim = F.cosine_similarity(logits1, logits2)

# Print the similarity scores for each sentence pair
for i, score in enumerate(cosine_sim):
    print(f"Similarity between sentence pair {i + 1}: {score.item()}")

```


If you find this work helpful, please cite us:

```bibtex
@inproceedings{ji2024towards, 
  title={Towards Flexible Evaluation for Generative Visual Question Answering},
  author={Ji, Huishan and Si, Qingyi and Lin, Zheng and Wang, Weiping},
  booktitle={ACM Multimedia 2024}
}
```


More details coming soon! Welcome any questions to jihuishan@iie.ac.cn.
