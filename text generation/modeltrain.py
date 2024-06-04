from LSTMmodel import *

seq_length = 32

dataset = Dataset(sequence_length=seq_length)
model = LSTMModel(dataset=dataset)

train_loop(dataset,model,seq_length=seq_length,batch_size=1024,max_epochs=100)