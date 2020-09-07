import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import LeakyReLU, Input, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed
from pre_processor import Preprocessor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Autoencoder():

    def __init__(self, epochs=10, optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,\
                 epsilon=0, w_decay=0, momentum=0, dropout=0):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.kernel_initializer = kernel_init
        self.gamma_initializer = gamma
        self.dropout = dropout
        self.epsilon = epsilon
        self.weight_decay = w_decay
        self.momentum = momentum

    def build_autoencoder(self, X):
        inputs = Input(shape= (X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)
        model = Model(inputs=inputs, outputs=output)

        return model



if __name__ == "__main__":
    # Preprocessing the dataset
    preprocessor = Preprocessor('logs_lhcb.json')
    preprocessor.preprocessing()
    print("debug 1: ", (preprocessor.x_all.shape), preprocessor.x_all.dtype)

    # Reshape inputs
    preprocessor.x_all = preprocessor.x_all.reshape(preprocessor.x_all.shape[0], 1, preprocessor.x_all.shape[1])
    print("preprocessor.x_all shape in AUTO-ENCODER: ", preprocessor.x_all.shape)

    # Build the model
    model_ = Autoencoder(optimizer='adam', loss='mae')
    model = model_.build_autoencoder(preprocessor.x_all)
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    # Train the model
    history = model.fit(preprocessor.x_all, preprocessor.x_all, epochs = model_.epochs, batch_size=model_.batch_size,
              validation_split=0.4).history

    # Plot the training loss
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.show()