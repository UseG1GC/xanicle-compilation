from LSTMmodel import *

dataset = Dataset()
model = LSTMModel(dataset=dataset)

train_loop(dataset,model,seq_length=4,batch_size=8192,max_epochs=100)