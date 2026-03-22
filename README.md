# Question Pair Similarity Detector

A natural language processing (NLP) application that predicts whether two questions are semantically similar, built using Word2Vec embeddings and a Random Forest classifier, deployed as a Streamlit web app.

## Overview

This is a self-learning project inspired by the Quora Question Pairs dataset. The goal is to identify duplicate or semantically equivalent questions — a common problem in community Q&A platforms, customer support systems, and search engines.

The application combines classical NLP feature engineering with Word2Vec sentence embeddings to classify question pairs as similar or dissimilar, achieving approximately 90% accuracy.

## Features

- Real-time question similarity prediction via an interactive Streamlit interface
- Word2Vec model for generating semantic word and sentence embeddings
- Rich feature engineering pipeline including token features, fuzzy matching, and length-based features
- Random Forest classifier optimized for classification performance
- Preprocessing pipeline with contraction expansion, HTML tag removal, and stopword filtering

## NLP Pipeline

1. **Text Preprocessing** — lowercasing, contraction expansion, HTML cleaning, punctuation removal
2. **Feature Engineering** — common word counts, stopword overlap, token ratios, fuzzy matching scores (FuzzyWuzzy)
3. **Word2Vec Embeddings** — sentence-level vector representations via mean pooling of word vectors
4. **Classification** — Random Forest classifier trained on combined feature set

## Tech Stack

- Python
- Gensim (Word2Vec)
- Scikit-learn (Random Forest)
- FuzzyWuzzy
- NLTK
- Streamlit
- BeautifulSoup
- Google Drive (model hosting via gdown)

## Project Structure

```
my_question_pairs/
├── app.py               # Streamlit application
├── my-code.ipynb        # Model training and EDA notebook
├── train.csv            # Training dataset
├── requirements.txt     # Python dependencies
├── Procfile             # Deployment configuration
└── start.sh             # Startup script
```

## How to Run Locally

```bash
git clone https://github.com/gyanaranjanmishra/my_question_pairs.git
cd my_question_pairs
pip install -r requirements.txt
streamlit run app.py
```

> Note: Models are downloaded automatically from Google Drive on first run.

## Results

| Metric | Score |
|---|---|
| Accuracy | ~90% |
| Classifier | Random Forest |
| Embeddings | Word2Vec (Gensim) |

## Author

**Gyanaranjan Mishra, PhD**
Applied Data Scientist | Materials Engineer
[LinkedIn](https://www.linkedin.com/in/gyanaranjanmishra/) | gyanaranjanmishra06@gmail.com
