import torch
import math
import torch.nn as nn

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dim):
        super(mLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dim = dim

        self.w_q = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_k = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_v = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.w_o = nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        self.b_z = nn.Parameter(torch.Tensor(hidden_size))
        self.b_q = nn.Parameter(torch.Tensor(hidden_size))
        self.b_v = nn.Parameter(torch.Tensor(hidden_size))

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        nn.init.xavier_uniform_(self.w_i)
        nn.init.xavier_uniform_(self.w_f)
        nn.init.xavier_uniform_(self.w_o)

        nn.init.zeros_(self.b_i)
        nn.init.zeros_(self.b_f)
        nn.init.zeros_(self.b_o)
        nn.init.zeros_(self.b_z)
        nn.init.zeros_(self.b_q)
        nn.init.zeros_(self.b_v)
    
    def forward(self, x_t, state):
        c_prev, n_prev = state

        q_t = torch.matmul(self.w_q,x_t) + self.b_q
        k_t = torch.matmul(self.w_k,x_t) / math.sqrt(self.dim) + self.b_k
        v_t = torch.matmul(self.w_v,x_t) + self.b_v

        i_tilda = torch.matmul(self.w_i,x_t) + self.b_i
        f_tilda = torch.matmul(self.w_f,x_t) + self.b_f
        o_tilda = torch.matmul(self.w_o,x_t) + self.b_o

        i_t = torch.exp(i_tilda)
        f_t = nn.Sigmoid(f_tilda)
        o_t = nn.Sigmoid(o_tilda)

        c_t = torch.matmul(f_t,c_prev) + torch.matmul(i_t,torch.matmul(v_t,k_t))
        n_t = torch.matmul(f_t,n_prev) + torch.matmul(i_t,k_t)

        h_tilda = torch.matmul(c_t,q_t) / max(torch.abs(torch.matmul(n_t,q_t)),1)
        h_t = nn.Conv1d(o_t,h_tilda)

        return h_t, (c_t,n_t)

class mLSTM(nn.Module):
    def __init__(self, n_lstm, n_layers, dim, dropout):
        super(mLSTM, self).__init__()
        self.layers = nn.ModuleList([mLSTMCell(n_lstm,n_lstm,dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state):
        x = self.dropout(x)
        c,n = state
        for layer in self.layers:
            x, (c,n) = layer(x,(c,n))
        return x, (c,n)

class xLSTMModel(nn.Module):
    def __init__(self, dataset, model_path="xLSTMmodel.pth"):
        super(xLSTMModel, self).__init__()
        self.n_layers = 16
        self.n_lstm = 512
        self.dropout = 0.2
        self.n_vocab = len(dataset.vocab)
        self.embedding = nn.Embedding(num_embeddings=self.n_vocab,embedding_dim=self.n_lstm)
        self.lstm = mLSTM(n_lstm=self.n_lstm,n_layers=self.n_layers,dim=self.n_lstm,dropout=self.dropout)
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