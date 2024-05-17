from LSTMmodel import *

dataset = Dataset()
model = LSTMStack(dataset=dataset)

train_loop(dataset,model,seq_length=9,batch_size=1024,max_epochs=10)