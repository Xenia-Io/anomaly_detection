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
import seaborn as sns
import pandas as pd
import numpy as np



class Autoencoder():

    def __init__(self, epochs=100 , optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,\
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
        # output = TimeDistributed(Dense(X.shape[2]))(L5)
        output = Dense(X.shape[2], activation='softmax')(L5)
        model = Model(inputs=inputs, outputs=output)

        return model



if __name__ == "__main__":
    # Preprocessing the dataset
    preprocessor = Preprocessor('logs_for_supervised.json', True, False)
    preprocessor.preprocessing()
    # print("debug 3: ", preprocessor.x_train)

    # Build train and test dataframes
    data_tuples = list(zip(preprocessor.x_train, preprocessor.y_train))
    train_df = pd.DataFrame(data_tuples, columns=['messages', 'labels'])
    # print("xenia 1: ", train_df['messages'][0])
    # print("xenia 2: ", preprocessor.x_train[0])

    data_tuples2 = list(zip(preprocessor.x_test, preprocessor.y_test))
    test_df = pd.DataFrame(data_tuples2, columns=['messages', 'labels'])

    # Reshape inputs
    preprocessor.x_train = preprocessor.x_train.reshape(preprocessor.x_train.shape[0], 1, preprocessor.x_train.shape[1])
    preprocessor.x_test = preprocessor.x_test.reshape(preprocessor.x_test.shape[0], 1, preprocessor.x_test.shape[1])
    print("preprocessor.x_train shape in AUTO-ENCODER: ", preprocessor.x_train.shape)

    # Build the model
    model_ = Autoencoder(optimizer='adam', loss='mse')
    model = model_.build_autoencoder(preprocessor.x_train)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Train the model
    history = model.fit(preprocessor.x_train, preprocessor.x_train, epochs = model_.epochs, batch_size=model_.batch_size,
              validation_split=0.4).history

    # Plot the training loss
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.show()

    # Plot loss distribution of the training set
    X_predict = model.predict(preprocessor.x_train)
    X_predict = X_predict.reshape(X_predict.shape[0], X_predict.shape[2])
    X_predict = pd.DataFrame(X_predict, columns=train_df.columns)
    X_predict.index = train_df.index

    scored = pd.DataFrame(index=train_df.index)
    Xtrain = preprocessor.x_train.reshape(preprocessor.x_train.shape[0], preprocessor.x_train.shape[2])
    scored['Loss_mse'] = np.mean(np.power(X_predict - Xtrain, 2), axis=1)
    plt.figure(figsize=(10,10), dpi=80)
    sns.distplot(scored['Loss_mse'] , bins=20, kde=True, color='blue')
    plt.xlim([0.0,0.5])
    plt.show()

    # Calculate the loss on the test set
    X_predict = model.predict(preprocessor.x_test)
    X_predict = X_predict.reshape(X_predict.shape[0], X_predict.shape[2])
    X_predict = pd.DataFrame(X_predict, columns=test_df.columns)
    X_predict.index = test_df.index

    scored = pd.DataFrame(index = test_df.index)
    Xtest = preprocessor.x_test.reshape(preprocessor.x_test.shape[0], preprocessor.x_test.shape[2])
    scored['Loss_mse'] = np.mean(np.power(X_predict-Xtest, 2), axis=1)
    scored['Threshold'] = 0.6
    scored['Anomaly'] = scored['Loss_mse'] > scored['Threshold']
    scored.head()

    X_predict_train = model.predict(preprocessor.x_train)
    X_predict_train = X_predict_train.reshape(X_predict_train.shape[0], X_predict_train.shape[2])
    X_predict_train = pd.DataFrame(X_predict_train, columns=train_df.columns)
    X_predict_train.index = train_df.index

    scored_train = pd.DataFrame(index=train_df.index)
    Xtrain = preprocessor.x_train.reshape(preprocessor.x_train.shape[0], preprocessor.x_train.shape[2])
    scored_train['Loss_mse'] = np.mean(np.power(X_predict_train - Xtrain, 2), axis=1)
    scored_train['Threshold'] = 0.6
    scored_train['Anomaly'] = scored_train['Loss_mse'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])
    scored.plot(logy=True, figsize=(10,10), color=['blue', 'red'])
    plt.show()