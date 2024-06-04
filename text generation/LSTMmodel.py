import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
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
    def __init__(self, dataset, model_path="LSTMmodel.pth"):
        super(LSTMModel, self).__init__()
        self.n_layers = 16
        self.n_lstm = 512
        self.dropout = 0.2
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
    
def train_loop(dataset,initial_model,batch_size=1024,seq_length=32,max_epochs=50):
    model = torch.compile(initial_model)

    model.train()

    train_data = DataLoader(dataset,batch_size=batch_size)
    lr = 1e-3

    loss_fn = nn.CrossEntropyLoss()

    try:
        for epoch in range(max_epochs):
            optimizer = optim.Adam(model.parameters(), lr=lr)
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
                print(f"Epoch: {epoch}, batch: {batch}, loss: {loss.item()}")
            
            print(f"Epoch : {epoch}, avg_loss : {average_loss}")
            lr = lr / 10
    except KeyboardInterrupt:
        initial_model.eval()
        torch.save(initial_model.state_dict(),initial_model.model_path)
        print("Model Saved!")

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