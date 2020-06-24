import tensorflow as tf
from tensorflow import keras
import textwrap
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data = open('/Users/a./Desktop/poetry/sonnets.txt').read()
corpus = data.lower().split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1
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


def create_model():
  model = Sequential()
  model.add(Embedding(total_words,450,input_length = max_sequence_len-1))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Dropout(0.5))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Dropout(0.5))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Bidirectional(LSTM(500, return_sequences=True)))
  model.add(Bidirectional(LSTM(500)))
  model.add(Dropout(0.5))
  model.add(Dense(total_words/2, activation='relu'))
  model.add(Dense(total_words/2, activation='relu'))
  model.add(Dense(total_words/2, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(total_words/2, activation='relu'))
  model.add(Dense(total_words/2, activation='relu'))
  model.add(Dense(total_words/2, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(total_words/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
  model.add(Dense(total_words,activation = 'softmax'))

  return model

model = create_model()
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Adam()
  , metrics = ['accuracy'])


model.load_weights("/Users/a./Desktop/poetry/Final_Poetry.h5")


seed_text = input(" Seed Text :")
next_words= input(" Words? ")
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
	seed_text += " " + output_word +"\n"
print("\n")
txt = textwrap.fill(seed_text, width=40)
print(txt)
f= open("Shairi.txt","w+")
f.write(txt+"\r\n")
f.close()
