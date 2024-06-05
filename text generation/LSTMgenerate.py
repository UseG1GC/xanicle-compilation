from LSTMmodel import *
from trainfunc import *

dataset = Dataset()
model = LSTMModel(dataset=dataset)

n_words = int(input("Enter amount of words to autocomplete: "))


while True:
    print(generate(dataset,model,input_text=input("Enter Text: "),n_words=n_words))