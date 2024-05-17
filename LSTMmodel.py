import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length=4):
        self.sequence_length=sequence_length
        self.words = self.load_words()
        self.vocab = self.load_vocab()

        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        data = pd.read_parquet("data.parquet")
        text = data["text"].str.cat(sep=" ")
        return text.split(" ")
        
    def load_vocab(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def __len__(self):
        return len(self.words_indexes) - self.sequence_length

    def __getitem__(self, i):
        return(
            torch.tensor(self.words_indexes[i:i+self.sequence_length]),
            torch.tensor(self.words_indexes[i+1:i+self.sequence_length+1]),
        )

class LSTMModel(nn.Module):
    def __init__(self, dataset, model_path="model2.pth"):
        super(LSTMModel, self).__init__()
        self.n_layers = 6
        self.n_lstm = 128
        self.embedding_dim = 128
        self.dropout = 0.2
        n_vocab = len(dataset.vocab)

        self.embedding = nn.Embedding(num_embeddings=n_vocab,embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.n_lstm,hidden_size=self.n_lstm,num_layers=self.n_layers,dropout=self.dropout)
        self.fc = nn.Linear(self.n_lstm,n_vocab)
        self.model_path = model_path
        try:
            self.load_state_dict(torch.load(model_path))
        except:
            pass
    def forward(self, x, prev_state):
        y = self.embedding(x)
        output, state = self.lstm(y, prev_state)
        logits = self.fc(output)
        return logits, state
    
    def _init_state(self, seq_length):
        return(torch.zeros(self.n_layers,seq_length,self.n_lstm),torch.zeros(self.n_layers,seq_length,self.n_lstm))

class LSTMStack(nn.Module):
    def __init__(self, dataset, model_path="model2stack.pth"):
        super(LSTMStack, self).__init__()
        self.n_layers = 6
        self.n_lstm = 128
        self.embedding_dim = 128
        self.dropout = 0.2
        n_vocab = len(dataset.vocab)

        self.embedding = nn.Embedding(num_embeddings=n_vocab,embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.n_lstm,hidden_size=self.n_lstm,num_layers=self.n_layers,dropout=self.dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.n_lstm,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,n_vocab)
        )
        self.model_path = model_path
        try:
            self.load_state_dict(torch.load(model_path))
        except:
            pass
    def forward(self, x, prev_state):
        y = self.embedding(x)
        output, state = self.lstm(y, prev_state)
        logits = self.fc(output)
        return logits, state
    
    def _init_state(self, seq_length):
        return(torch.zeros(self.n_layers,seq_length,self.n_lstm),torch.zeros(self.n_layers,seq_length,self.n_lstm))
    
def train_loop(dataset,model,batch_size=1024,seq_length=4,max_epochs=50):
    model.train()

    train_data = DataLoader(dataset,batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(max_epochs):
        a, b = model._init_state(seq_length)
        total_loss = 0
        average_loss = 0

        for batch, (x, y) in enumerate(train_data):
            optimizer.zero_grad()

            y_prediction, (a, b) = model(x, (a,b))
            loss = loss_fn(y_prediction.transpose(1,2), y)

            a = a.detach()
            b = b.detach()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.25)
            optimizer.step()
            total_loss += loss.item()
            average_loss = total_loss / (batch+1)
            torch.save(model.state_dict(),model.model_path)
            print(f"Epoch: {epoch}, batch: {batch}, loss: {loss.item()}")
        
        print(f"Epoch : {epoch}, avg_loss : {average_loss}")

def generate(dataset,model,input_text,n_words):
    text = input_text.split(" ")
    model.eval()

    a, b = model._init_state(len(text))

    for i in range(0,n_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in text[i:]]])
        y_prediction, (a, b) = model(x, (a,b))

        output = y_prediction[0][-1]
        logits = torch.nn.functional.softmax(output, dim=0).detach()
        topk_values = torch.nn.functional.softmax(torch.topk(logits,40).values,dim=0).detach()
        topk_index = torch.topk(logits,40).indices
        index = np.random.choice(len(topk_index), p = topk_values.numpy())
        text.append(dataset.index_to_word[topk_index[index].item()])
    return text

dataset = Dataset()
model = LSTMModel(dataset)

while True:
    for chunk in generate(dataset,model,input_text=input("\nEnter Text: "), n_words=100):
        print(chunk,end=" ")
    