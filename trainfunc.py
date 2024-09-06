import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

def train_loop(dataset,initial_model,batch_size=32768,seq_length=32,max_epochs=500):
    model = torch.compile(initial_model)

    model.train()

    train_data = DataLoader(dataset,batch_size=batch_size)
    lr = 1e-4

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
    except KeyboardInterrupt:
        initial_model.eval()
        torch.save(initial_model.state_dict(),initial_model.model_path)
        print("Model Saved!")

def generate(dataset,model,input_text,n_words):
    text = dataset.vocab.tokenize(input_text)
    decoder = dataset.vocab.decoder()
    model.eval()

    a, b = model._init_state(len(text))

    for i in range(0,n_words):
        x = torch.tensor(text)
        y_prediction, (a, b) = model(x, (a,b))

        output = y_prediction.transpose(1,2)
        logits = torch.nn.functional.softmax(output, dim=0).detach()
        topk_values = torch.nn.functional.softmax(torch.topk(logits,40).values,dim=0).detach()
        topk_index = torch.topk(logits,40).indices
        index = np.random.choice(len(topk_index), p = topk_values.numpy())
        text.append(topk_index[index].item())
    return decoder.decode(text)