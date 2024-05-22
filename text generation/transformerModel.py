import torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from collections import Counter
import math

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length=8):
        self.sequence_length=sequence_length
        self.words = self.load_words()
        self.vocab = self.load_vocab()

        self.index_to_word = {index: word for index, word in enumerate(self.vocab)}
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}

        self.data = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        data = pd.read_parquet("data/data.parquet")
        text = data["text"].str.cat(sep=" ")
        return text.split(" ")
        
    def load_vocab(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, i):
        return (torch.tensor(self.data[i:i+self.sequence_length]),torch.tensor(self.data[i+1:i+self.sequence_length+1]))

class EvalData(torch.utils.data.Dataset):
    def __init__(self, train_data):
        self.sequence_length = train_data.sequence_length
        self.index_to_word = train_data.index_to_word
        self.word_to_index = train_data.word_to_index
        self.data = self.load_words()
    
    def load_words(self):
        data = pd.read_parquet("data/eval.parquet")
        text = data["text"].str.cat(sep=" ")
        text = text.split(" ")
        output = [self.word_to_index[w] for w in text]
        return output

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, i):
        return (torch.tensor(self.data[i:i+self.sequence_length]),torch.tensor(self.data[i+1:i+self.sequence_length+1]))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self,dataset,d_model=512,n_head=8,n_layers=10,dropout=0.2,model_path="transformer.pth"):
        super(TransformerModel,self).__init__()
        self.n_vocab = len(dataset.vocab)
        self.src_mask = None
        self.path = model_path

        self.p_encoder = PositionalEncoding(d_model,dropout)
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head,dropout=dropout),num_layers=n_layers)
        self.fc = nn.Linear(d_model,self.n_vocab)
        
        try:
            self.load_state_dict(torch.load(model_path))
            print("Model Loaded!")
        except:
            print("Initializing new model!")
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            nn.init.zeros_(self.fc.bias)
            nn.init.uniform_(self.fc.weight, -0.1, 0.1)

    def generate_mask(self, size):
        return torch.log(torch.tril(torch.ones(size,size)))

    def forward(self, x, mask = True):
        if mask:
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = self.generate_mask(len(x))
                self.src_mask = mask
        else:
            self.src_mask = None
            
        x = self.embedding(x) * math.sqrt(self.n_vocab)
        x = self.p_encoder(x)
        output = self.transformer(x,mask=self.src_mask)
        logits = self.fc(output)
        return logits

def eval(model, dataset,batch_size):
    model.eval()
    eval_data = DataLoader(dataset=dataset,batch_size=batch_size)
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch, (x,y) in enumerate(eval_data):
            y_pred = model(x)
            y_pred = y_pred.transpose(1,2)
            total_loss += len(x) * loss_fn(y_pred,y).item()
    
    return total_loss/len(eval_data)

def train_loop(model, dataset,batch_size=1024,max_epochs=5):
    lr = 0.001

    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_data = DataLoader(dataset=dataset,batch_size=batch_size)
    eval_data = DataLoader(dataset=EvalData(dataset),batch_size=batch_size)

    for epoch in range(max_epochs):
        for batch, (x,y) in enumerate(training_data):
            model.zero_grad()

            optimizer.zero_grad()

            y_pred = model(x)
            y_pred = y_pred.transpose(1,2)

            loss = loss_fn(y_pred,y)

            loss.backward()
            optimizer.step()
            print(f"Epoch: {str(epoch)},Batch: {str(batch)}, Loss: {str(loss.item())}")
            torch.save(model.state_dict(),model.path)

        eval_loss = eval(model,eval_data,batch_size)
        print(f"Average loss for epoch {epoch} : {eval_loss}")

dataset = Dataset()
model = TransformerModel(dataset=dataset)

train_loop(model,dataset)