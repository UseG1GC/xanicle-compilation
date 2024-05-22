from LSTMmodel import *

dataset = Dataset()
model = LSTMModel(dataset=dataset)

n_words = input("Enter amount of words to autocomplete: ")

while True:
    for chunk in generate(dataset,model,input_text=input("\nEnter Text: "),n_words=n_words):
        print(chunk, end=" ")