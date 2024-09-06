import torch
import pandas as pd
from torch import nn
import tokenmonster

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length=32):
        self.vocab = tokenmonster.load("english-24000-consistent-v1")
        self.sequence_length=sequence_length
        self.words = self.load_words()

    def load_words(self):
        output = []
        data = pd.read_parquet(f"data/data.parquet")
        for chunk in data["text"]:
            if len(chunk) > 0:
                for i in self.vocab.tokenize(chunk):
                    output.append(int(i))
        return output
    
    def __len__(self):
        return len(self.words) - self.sequence_length

    def __getitem__(self, i):
        return(
            torch.tensor(self.words[i:i+self.sequence_length]),
            torch.tensor(self.words[i+1:i+self.sequence_length+1]),
        )

class LSTMModel(nn.Module):
    def __init__(self, dataset, model_path="LSTMmodel.pth",n_lstm = 8192, n_layers = 64, dropout = 0.2):
        super(LSTMModel, self).__init__()
        self.n_layers = n_layers
        self.n_lstm = n_lstm
        self.dropout = dropout
        self.n_vocab = len(dataset.vocab)
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=self.n_lstm)
        self.lstm = nn.LSTM(input_size=self.n_lstm,hidden_size=self.n_lstm,num_layers=self.n_layers,dropout=self.dropout)
        self.fc = nn.Linear(self.n_lstm,self.n_vocab)
        self.model_path = model_path

        try:
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded at location {model_path}")
        except:
            print("Model not found, initializing new model")
    
    def forward(self, x, prev_state):
        y = self.embedding(x)
        output, state = self.lstm(y, prev_state)
        logits = self.fc(output)
        return logits, state
    
    def _init_state(self, seq_length):
        return(torch.zeros(self.n_layers,seq_length,self.n_lstm),torch.zeros(self.n_layers,seq_length,self.n_lstm))