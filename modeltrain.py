from LSTMmodel import *
from trainfunc import *
from xLSTM import *

seq_length = 128

dataset = Dataset(sequence_length=seq_length)
model = LSTMModel(dataset=dataset)

train_loop(dataset,model,seq_length=seq_length)