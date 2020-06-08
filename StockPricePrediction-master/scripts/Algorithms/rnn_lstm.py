#! /usr/bin/python
'''
    Running LSTM Algorithm.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.core import *

max_features = 5883
maxlen = 80
batch_size = 32

in_out_neurons = 2
hidden_neurons = 300

import os
import sys
import pandas as pd


def _load_data(data, n_prev=100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev])
        docY.append(data.iloc[i+n_prev])

    all_X = np.array(docX)
    all_Y = np.array(docY)

    return all_X, all_Y


def train_test_split(dataframe, test_size=0.2):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(dataframe) * (1 - test_size)))

    X_train, y_train = _load_data(dataframe.iloc[0:ntrn])
    X_test, y_test = _load_data(dataframe.iloc[ntrn:])

    print(X_train, y_train)

    return (X_train, y_train), (X_test, y_test)


def rnn_lstm(file_dataframe, test_size=0.2, col="high"):
    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = train_test_split(
        file_dataframe[col], test_size=0.2)
    
    '''

    X_train = np.array([[ 360, 7, 19, 256, 82, 7], \
                        [ 6, 102, 37, 5, 1324, 7]])

    y_train = np.array([1, 0])

    X_test = X_train

    y_test = y_train

    print(X_train.shape, y_train.shape)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    
    '''

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    '''
    model = Sequential()
    model.add(Embedding(max_features, hidden_neurons, \
        input_length=maxlen, dropout=0.2))
    model.add(LSTM(hidden_neurons, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    #model.compile(loss="mean_squared_error", \
    #    optimizer="rmsprop", metrics=['accuracy'])
    '''

    input_dim = 32
    hidden = 32
    step = 10

    #The LSTM  model -  output_shape = (batch, step, hidden)
    model1 = Sequential()
    model1.add(LSTM(input_dim=input_dim, output_dim=hidden, input_length=step, return_sequences=True))

    #The weight model  - actual output shape  = (batch, step)
    # after reshape : output_shape = (batch, step,  hidden)
    model2 = Sequential()
    model2.add(Dense(input_dim=input_dim, output_dim=step))
    model2.add(Activation('softmax')) # Learn a probability distribution over each  step.
    #Reshape to match LSTM's output shape, so that we can do element-wise multiplication.
    model2.add(RepeatVector(hidden))
    model2.add(Permute((2, 1)))

    #The final model which gives the weighted sum:
    model = Sequential()
    model.add(Merge([model1, model2], 'sum', concat_axis=1))  # Multiply each element with corresponding weight a[i][j][k] * b[i][j]
    model.add((Merge([model1, model2], mode='sum', concat_axis=1)) # Sum the weighted elements.

    model.compile(loss='mse', optimizer='sgd')

    print('Train...')
    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)

    model.fit(X_train, y_train, batch_size=batch_size, \
        validation_data=(X_test, y_test), nb_epoch=5)
    score, accuracy = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', accuracy)

    return (score, accuracy)


def main(dir_path):
    '''
        Run Pipeline of processes on file one by one.
    '''
    files = os.listdir(dir_path)

    #for file_name in files:
    file_name="GOOGL.csv"
    print(file_name)

    file_dataframe = pd.read_csv(os.path.join(dir_path, file_name))

    print(rnn_lstm(file_dataframe, 0.1, 'high'))

    #break

if __name__ == '__main__':
    main(sys.argv[1])
