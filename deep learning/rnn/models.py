import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden_dim, attn_hidden_dim=128):
        super().__init__()

        self.attn = nn.Sequential(nn.Linear(hidden_dim, attn_hidden_dim), nn.Tanh(), nn.Linear(attn_hidden_dim, 1))

    def forward(self, rnn_outputs, mask):
        #rnn_outputs: (batch_size X T X D=2H)
        #mask: (batch_size X T) with 1 for real tokens, 0 for PAD

        scores = self.attn(rnn_outputs).squeeze(-1)  # (batch_size X T)
        scores = scores.masked_fill(mask == 0, -1e9) #ignore PAD
        weights = torch.softmax(scores, dim=1)       # (batch_size X T)

        context = torch.sum(rnn_outputs * weights.unsqueeze(-1), dim=1)  # (batch_size X D=2H)

        return weights, context



class RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, embedding_matrix,
                 hidden_size, num_layers, num_classes,
                 use_pooling, use_attention, model_type,
                 attn_hidden_dim=128, pad_id=0, freeze=True):
        super(RNN, self).__init__()

        self.pad_id = pad_id
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_pooling = use_pooling
        self.use_attention = use_attention
        self.model_type = model_type
        D = self.hidden_size *2 #(BIDERECTIONAL DIMENION FOR EVERY HIDDEN STATE)

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze, padding_idx=pad_id
        )


        if self.model_type == 'GRU':
            self.gru = nn.GRU(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        elif self.model_type == 'LSTM':
            self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.RNN(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)

        if self.use_attention == True:
            self.attention = MLP(D, attn_hidden_dim=128)
        else:
            self.attention = None

        self.fc = nn.Linear(D, num_classes)


    def forward(self, ids):
        batch_size = ids.size(0)
        device = ids.device
        #ids = batch_size X T
        mask = (ids != self.pad_id).long()   # (batch_size X T)
        emb = self.embedding(ids)

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)


        if self.model_type == 'GRU':
            out, _ = self.gru(emb, h0)
        elif self.model_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size,device=device)
            out, _ = self.lstm(emb, (h0, c0))
            #last hidden state from every layer is useless
        else:
            out, _ = self.rnn(emb,h0)

        if self.use_pooling:
            #max pooling, but ignore PAD
            out_masked = out.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            pooled = torch.max(out_masked, dim=1).values     # (batch_size X D)
            logits = self.fc(pooled)                         # (batch_size X Categories = 2)
            return logits


        elif self.use_attention:          #context vector c = Σah
            attention_weights, context_vec = self.attention(out, mask)
            logits = self.fc(context_vec)
            return logits

        return out

