# Towards Flexible Evaluation for Generative Visual Question Answering
This is the official repository for paper "Towards Flexible Evaluation for Generative Visual Question Answering", an oral in ACM Multimedia 2024.


## Models Overview

We have trained two models with different sizes: **SFVE-base** and **SFVE-large**. Below are the details of each model along with their performance on our proposed AVE dataset.

| Model Name  | Backbone     | HuggingFace URL                                                | Performance (AVE) |
|-------------|--------------|----------------------------------------------------------------|-------------------|
| SFVE-base   | RoBERTa-base | [SFVE-base on HuggingFace](https://huggingface.co/Huishan/SFVE-base)  | 56.3              |
| SFVE-large  | RoBERTa-large| [SFVE-large on HuggingFace](https://huggingface.co/Huishan/SFVE-large) | 59.4              |


## How to Use the Models

The SFVE models are designed for flexibly evaluating the generated results from MLLMs on VQA datasets. Specifically, the expected usage

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

More details coming soon! Welcome any questions to jihuishan@iie.ac.cn.
