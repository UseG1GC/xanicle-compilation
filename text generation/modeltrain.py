from LSTMmodel import *
from trainfunc import *
from xLSTM import *

seq_length = 32

dataset = Dataset(sequence_length=seq_length)
model = LSTMModel(dataset=dataset)

train_loop(dataset,model,seq_length=seq_length,batch_size=256,max_epochs=100)