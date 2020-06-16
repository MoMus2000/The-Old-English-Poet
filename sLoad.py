from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import textwrap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.models.load_model('/Users/a./Desktop/poetry/Higher Val Acc')

data = open('/Users/a./Desktop/poetry/sonnets.txt').read()
corpus = data.lower().split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1
print(total_words)
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences,max_sequence_len,padding = 'pre'))

xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels,num_classes = total_words)


seed_text = input(" Thou Seed me some Text ... ")
next_words= input(" How many words ? ")
next_words = int(next_words)
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print("\n")
txt = textwrap.fill(seed_text, width=75)
print(txt)
f= open("Shairi.txt","w+")
f.write(txt+"\r\n")
f.close()
