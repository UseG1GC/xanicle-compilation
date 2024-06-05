import torch
import math
import torch.nn as nn

class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dim):
        super(mLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dim = dim

        self.w_q = nn.Linear(hidden_size, input_size)
        self.w_k = nn.Linear(hidden_size, input_size)
        self.w_v = nn.Linear(hidden_size, input_size)
        self.w_i = nn.Linear(hidden_size, input_size)
        self.w_f = nn.Linear(hidden_size, input_size)
        self.w_o = nn.Linear(hidden_size, input_size)

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

        q_t = self.w_q(x_t)
        k_t = self.w_k(x_t) / math.sqrt(self.dim)
        v_t = self.w_v(x_t)

        i_tilda = self.w_i(x_t)
        f_tilda = self.w_f(x_t)
        o_tilda = self.w_o(x_t)

        i_t = torch.exp(i_tilda)
        f_t = torch.sigmoid(f_tilda)
        o_t = torch.sigmoid(o_tilda)

        c_t = torch.multiply(f_t,c_prev) + torch.multiply(i_t,torch.multiply(v_t,k_t))
        n_t = torch.multiply(f_t,n_prev) + torch.multiply(i_t,k_t)

        h_tilda = torch.multiply(c_t,q_t) / max(torch.abs(torch.multiply(n_t,q_t)),1)
        h_t = torch.conv1d(o_t,h_tilda)

        return h_t, (c_t,n_t)

class mLSTM(nn.Module):
    def __init__(self, n_lstm, n_layers, dim, dropout):
        super(mLSTM, self).__init__()
        self.layers = nn.ModuleList([mLSTMCell(n_lstm,n_lstm,dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, init_state):
        bs, seq_len, _ = x.size()
        x = self.dropout(x)
        outputs = []
        current_states = init_state
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_states = []
            for layer, state in zip(self.layers, current_states):
                h_t, new_state = layer(x_t, state)
                new_states.append(new_state)
                x_t = h_t
            outputs.append(h_t.unsqueeze(1))
            current_states = new_states

        outputs = torch.cat(outputs, dim=1)
        return outputs, current_states


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