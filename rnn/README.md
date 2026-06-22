# Sentiment Analysis with RNN, GRU, and LSTM

## Overview
This repository implements a production-ready, comparative study of **recurrent network architectures** for **binary sentiment classification** using the **IMDB large movie reviews dataset**. 

The models are developed using **PyTorch**, optimized with **Bidirectional** processing, and enhanced with advanced aggregation strategies such as **Max Pooling over time** and an **MLP-based Self-Attention mechanism**. Semantic text representation is achieved using **pretrained Google News Word2Vec embeddings (300d)**.

The project follows clean-code software engineering principles by decoupling core architectural definitions from execution environments.

---

## Code Structure

* **models.py** - PyTorch implementations of MLP (Attention) and the generic RNN/GRU/LSTM class.
* **utils.py** - Text preprocessing, vocabulary builder, PyTorch Dataset, and train/test loops.
* **main.ipynb** - Interactive execution notebook containing pipelines, visualization, and persistent metrics.

---

## Models Implemented

The following architectures are implemented and evaluated:

- **Vanilla RNN**
- **GRU (Gated Recurrent Unit)**
- **LSTM (Long Short-Term Memory)**

Each architecture is tested with:
- **Max Pooling over time**
- **MLP-based Attention mechanism**

This results in a total of **six models**:
- RNN + Max Pooling  
- GRU + Max Pooling  
- LSTM + Max Pooling  
- RNN + Attention  
- GRU + Attention  
- LSTM + Attention
---

## Dataset
- **IMDB Large Movie Review Dataset**
- Binary sentiment classification:
  - `0`: Negative  
  - `1`: Positive  

The dataset is split into:
- Training set
- Validation (development) set
- Test set  

---

## Text Representation

- Tokenization and preprocessing with regular expressions
- Vocabulary built from training data with frequency thresholding
- Fixed-length sequences with padding and masking
- **Pretrained Word2Vec embeddings (Google News, 300d)**
- Padding and unknown tokens handled explicitly

---

## Training Details

- Loss function: **Cross Entropy Loss**
- Optimizer: **Adam**
- Bidirectional recurrent layers
- Best model selection based on **validation loss**
- Training and validation loss curves are visualized

---

## Evaluation Metrics

Models are evaluated on the test set using:
- Accuracy
- Precision, Recall, F1-score (Micro & Macro)
- Full classification report

---

## Results
- In main.ipynb file there are the results of the final run before commit in this repo.
- Analytic results from different combinations of hyperparameters can be found in: https://drive.google.com/drive/folders/1p0zjjSEbGFUO3WnQ5tjgVsMRTmLMrVUF?usp=drive_link
- Folders in Google Drive present the results from the runs with the hyperparameters of their name.

### Mapping of hyperparameters

- hl = hidden layers
- lr = learning rate
- hs = hidden states
- bs = batch size
- %dev data
- max words per critisism

## Language and Libraries Used

- Python
- PyTorch
- Gensim (Word2Vec)
- scikit-learn
- NumPy
- Matplotlib

---

## Purpose
This project was developed for educational purposes to:
- Gain hands-on experience with sequence models
- Understand the impact of attention mechanisms
- Compare different recurrent architectures in a controlled setting
- Practice end-to-end NLP model development in PyTorch

---

## Environment & Dependency Setup

Ensure you have a Python environment running (preferably with GPU/CUDA capabilities).

1. Clone or download the repository files (`models.py`, `utils.py`, `main.ipynb`) into your workspace folder.
2. Install the necessary system dependencies:

```bash
pip install torch gensim scikit-learn numpy matplotlib
