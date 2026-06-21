# Sentiment Analysis with RNN, GRU, and LSTM

## Overview
This project presents a comparative study of **recurrent neural network architectures** for **binary sentiment classification** on the **IMDB movie reviews dataset**.  
The models are implemented in **PyTorch** and enhanced with **max pooling** and **attention mechanisms**, while leveraging **pretrained Word2Vec embeddings** to capture semantic information.

The goal of the project is to evaluate how different sequence modeling architectures and aggregation strategies affect classification performance.

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
- 
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
- Analytic results can be found in: https://drive.google.com/drive/folders/1p0zjjSEbGFUO3WnQ5tjgVsMRTmLMrVUF?usp=drive_link

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

