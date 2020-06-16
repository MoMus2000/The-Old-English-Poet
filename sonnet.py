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

model = Sequential()
model.add(Embedding(total_words,100,input_length = max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(120)))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(total_words,activation = 'softmax'))
model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Adam()
, metrics = ['accuracy'])
history = model.fit(xs, ys, epochs=135)


model.save('/Users/a./Desktop/poetry')
