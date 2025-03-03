# Machine Learning - NLP: Skip-Gram Word Embeddings

## Overview

This notebook focuses on implementing word embeddings, compact and dense vector representations of words that encode their semantic meaning. Specifically, we implement the **Word2Vec Skip-Gram** model and apply **negative sampling** to optimize training efficiency.

## Objectives

- Understand the theoretical foundation of Word2Vec and Skip-Gram architecture.
- Implement a neural network model to generate high-quality word embeddings.
- Utilize negative sampling to efficiently train the Skip-Gram model.
- Evaluate the embeddings using similarity metrics and downstream tasks.

## Dependencies & Installation

Ensure the following libraries are installed before running the notebook:

```bash
pip install nltk gensim tensorflow
```

## Key Components

### 1. Data Preprocessing

- **Text Normalization & Tokenization**: Cleaning and preparing raw text data.
- **Stopword Removal & Frequency-Based Filtering**: Improving the quality of input data.
- **Subsampling of Frequent Words**: Reducing the dominance of common words in the training process.

### 2. Word2Vec Skip-Gram Implementation

- **Context Window Selection**: Defining how surrounding words influence embeddings.
- **Negative Sampling Strategy**: Efficiently training the model by reducing computational complexity.
- **Model Training & Optimization**: Using stochastic gradient descent for parameter updates.

### 3. Embedding Evaluation

- **Cosine Similarity Measurement**: Assessing the quality of learned embeddings.
- **T-SNE Visualization**: Projecting high-dimensional embeddings into a lower-dimensional space.

## Metadata Files

- **Generated metadata files** contain extracted embeddings as well as test data for further evaluation.
- These files facilitate reproducibility and ensure that learned embeddings can be reused for additional experiments.

## Execution Steps

1. Ensure all dependencies are installed and properly configured.
2. Run all cells sequentially to initialize data processing and model training.
3. Validate the learned embeddings using similarity analysis and visualization.
4. Use stored metadata files for further experimentation and evaluation.

## Notes

- The implementation is based on **TensorFlow**, and usage of PyTorch is **not permitted** for consistency.
- Before submission, ensure all cells have been executed to maintain reproducibility and correctness.
- Negative sampling significantly improves the efficiency of Skip-Gram training without sacrificing performance.

