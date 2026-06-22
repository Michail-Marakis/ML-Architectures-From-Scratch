
import os, glob, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, f1_score, precision_recall_fscore_support
from gensim.models import KeyedVectors

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.vocab = vocab
        self.max_length = max_length
        self.texts = [self.tokenize(t) for t in texts]
        self.labels = labels

    def tokenize(self, text):
        words = re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()
        ids = [self.vocab.get(w, self.vocab['UNK']) for w in words]
        ids = ids[:self.max_length]
        pad_len = self.max_length - len(ids)
        ids = ids + [self.vocab['PAD']] * pad_len
        return ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.tensor(self.texts[idx], dtype=torch.long)          # (T,)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return ids, label



def load_aclImdb_split(split_dir, maxes):
    texts, labels = [], []
    for label_name, y in [("neg", 0), ("pos", 1)]:
        pattern = os.path.join(split_dir, label_name, "*.txt")
        for fp in glob.glob(pattern):
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                labels.append(y)
                words = re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()
                maxes.append(len(words))
    return texts, labels, maxes


def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        words = re.sub(r'[^a-zA-Z]', ' ', t.lower()).split()
        counter.update(words)

    vocab = {"PAD": 0, "UNK": 1}
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab



  #-----------------------------------------------------------------------------------------------------------------

def build_embedding_matrix(vocab, w2v, emb_dim=300):
    V = len(vocab)
    pad_id = vocab["PAD"]
    unk_id = vocab["UNK"]

    #mean vector για UNK
    mean_vec = np.mean(w2v.vectors, axis=0)  # (300,)

    M = np.zeros((V, emb_dim), dtype=np.float32)

    for word, idx in vocab.items():
        if idx == pad_id:
            M[idx] = np.zeros(emb_dim)                 #PAD -> 0
        elif word in w2v:
            M[idx] = w2v[word]                         #γνωστή λέξη
        else:
            M[idx] = mean_vec                          #UNK -> mean

    M[unk_id] = mean_vec  #σιγουριά για UNK

    return torch.tensor(M, dtype=torch.float)   #vocanulary X 300

#--------------------------------------------------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        total_train_loss = 0

        for ids, labels in train_loader:
            ids = ids.to(device)
            labels = labels.to(device)              # (B,) long

            optimizer.zero_grad()
            logits = model(ids)                     # (B, 2)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for ids, labels in val_loader:
                ids = ids.to(device)
                labels = labels.to(device)

                logits = model(ids)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        #BEST EPOCH (dev-based selection)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Dev Loss: {avg_val_loss:.4f}")

    # restore best model
    model.load_state_dict(best_state)

    return train_losses, val_losses


def test_model(model, test_loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for ids, labels in test_loader:
            ids = ids.to(device)
            labels = labels.to(device)

            logits = model(ids)                 # (B, 2)
            preds = torch.argmax(logits, dim=1) # (B,)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    #Metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro"
    )

    report = classification_report(y_true, y_pred, digits=4)

    return (
        accuracy,
        precision_micro,
        recall_micro,
        f1_micro,
        precision_macro,
        recall_macro,
        f1_macro,
        report
    )
#-----------------------------------------------------------------------------------------------------------------