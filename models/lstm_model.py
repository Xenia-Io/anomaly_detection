from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
# from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
import kerastuner as kt
import tensorflow as tf
import seaborn as sns
from utils import *
import pandas as pd
import numpy as np
import gensim

sns.set(style='whitegrid', context='notebook')


class LSTM_Model(HyperModel):

    def __init__(self, pretrained_weights, epochs=3, optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,
                 epsilon=0, w_decay=0, momentum=0, dropout=0, embed_size=300, max_features=10000, maxlen=200):
        """
        Credits to: https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b
        """
        super().__init__()
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
        self.embed_size = embed_size  # how big is each word vector
        self.max_features = max_features  # how many unique words to use (i.e num rows in embedding vector)
        self.maxlen = maxlen  # max number of words in a log message to use
        self.emdedding_size = 64
        self.vocab_size = 55
        self.pretrained_weights = pretrained_weights


    def build(self, hp):
        print("... Start building LSTM model ...")

        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.emdedding_size, weights=[self.pretrained_weights]))
        model.add(LSTM(units=self.emdedding_size))
        model.add(Dense(units=self.emdedding_size))
        model.add(Activation('relu'))
        # hp = HyperParameters()
        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta'])
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model


    def word2idx(self, word_model, word):
        try:
            return word_model.wv.vocab[word].index
            # If word is not in index return 0. I realize this means that this
            # is the same as the word of index 0 (i.e. most frequent word), but 0s
            # will be padded later anyway by the embedding layer (which also
            # seems dirty but I couldn't find a better solution right now)
        except KeyError:
            return 0

    def idx2word(self, word_model, idx):
        return word_model.wv.index2word[idx]


    def detect_mad_outliers(self, points, threshold=3.5):
        # calculate the median of the input array
        print("points_______________ ", points)
        median = np.median(points, axis=0)

        # calculate the absolute difference of each data point to the calculated median
        deviation = np.abs(points - median)

        # take the median of those absolute differences
        med_abs_deviation = np.median(deviation)

        # 0.6745 is the 0.75th quartile of the standard normal distribution,
        # to which the MAD converges.
        modified_z_score = 0.6745 * deviation / med_abs_deviation

        # return as extra information what the original mse value was at which the threshold is hit
        # need to find a way to compute this mathematically, but I'll just use the index of the nearest candidate for now
        idx = (np.abs(modified_z_score - threshold)).argmin()
        print("idx 1: ", idx)
        if idx >= len(points):
            idx = np.argmin(points)
        print("idx 2: ", idx)
        threshold_value = points[idx]

        return modified_z_score, threshold_value


if __name__ == "__main__":
    print("tf version: ", tf.__version__)
    print("tf.keras version: ", tf.keras.__version__)

    # Preprocessing the dataset
    preprocessor = Preprocessor('./data/big_dataset.json', True, visualize=False)
    df = preprocessor.load_data_supervised('./data/big_dataset.json')
    df_copy = df.copy()
    messages = df['messages'].values

    for i in range(messages.shape[0]):
        # print("i: ", i, " with message: ", messages[i])
        messages[i] = ''.join([i for i in messages[i] if not i.isdigit()])
        messages[i] = gensim.utils.simple_preprocess (messages[i])
        # print("i: ", i, " with message: ", messages[i])

    w2v_model, pretrained_weights, vocab_size, emdedding_size = build_NLP_model(messages)
    test_NLP_model(w2v_model)

    # Data preparation for feeding the network
    set_x, set_x_test, set_y, set_y_test, train_x, train_y, val_x, \
    val_y, test_x, test_y = prepare_data(preprocessor, df_copy, w2v_model)

    # Compile model
    lstm = LSTM_Model(pretrained_weights)
    # model_ = lstm.build_lstm(train_x, vocab_size, emdedding_size, pretrained_weights)

    tuner = kt.tuners.RandomSearch(
        lstm,
        objective='val_accuracy',
        max_trials=20,
        directory='my_dir')

    tuner.search(train_x, train_y,
                 validation_data=(val_x, val_y),
                 epochs=10)
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    # model_.compile(optimizer='adam', loss='mse')
    # model_.summary()
    print("best_hyperparameters : ", best_hyperparameters.get('optimizer'))
    print("best_hyperparameters : ", best_hyperparameters.get('learning_rate'))
    print("test_x.shape: ", test_x.shape)
    exit(0)
    # Visualization
    visualise_data(set_x, set_x_test, set_y, set_y_test, tsne=False, pca=False, umap=False)

    # dot_img_file = 'model_2_lstm.png'
    # tf.keras.utils.plot_model(model_, to_file=dot_img_file, show_shapes=True)

    # Train the model
    history = model_.fit(
        train_x, train_y,
        epochs=lstm.epochs,
        batch_size=lstm.batch_size,
        shuffle=True, validation_data=(val_x, val_y)
    ).history

    # Plot the training and validation loss
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.show()

    # Test the models

    reconstructions = model_.predict(test_x)
    print("HERE..............  ::: ", reconstructions.shape, test_x.shape)
    # for i in range((test_x.shape[0])):
    #     # print("test" , i , ": ",test_x[i])
    #     print("recon" , i , ": ",reconstructions[i])

    # calculating the mean squared error reconstruction loss per row in the numpy array
    mse = np.mean(np.power(test_x - reconstructions, 2), axis=1)
    print(mse.shape)
    # showing the reconstruction losses for a subsample of transactions
    print(f'Mean Squared Error reconstruction losses for {5} clean transactions:')
    print(mse[np.where(test_y == 0)][:5])
    print(f'\nMean Squared Error reconstruction losses for {5} fraudulent transactions:')
    print(mse[np.where(test_y == 1)][:5])

    # adjust this parameter to customise the recall/precision trade-off
    Z_SCORE_THRESHOLD = 3

    # find the outliers on our reconstructions' mean squared errors
    mad_z_scores, threshold_value = lstm.detect_mad_outliers(mse, threshold=Z_SCORE_THRESHOLD)
    mad_outliers = (mad_z_scores > Z_SCORE_THRESHOLD).astype(int)
    print("mad outliers of our reconstructions' MSE: ", mad_outliers)

    anomalies = len(mad_outliers[mad_outliers == True])
    total_trades = len(test_y)
    d = (anomalies / total_trades * 100)

    print("MAD Z-score > ", Z_SCORE_THRESHOLD, " is the selected threshold.")
    print("Any trade with a MSRE >= ", threshold_value, " is flagged.")
    print("This results in", anomalies, "detected anomalies, or ", d, "% out of ", total_trades, "trades reported")

    data = np.column_stack((range(len(mse)), mse))
    print("data =", type(data), data)
    # scatter's x & y
    clean_x, clean_y = data[test_y == 0][:, 0], data[test_y == 0][:, 1]
    fraud_x, fraud_y = data[test_y == 1][:, 0], data[test_y == 1][:, 1]
    print("clean x,y : ", clean_x, clean_y)
    print("fraud x,y : ", fraud_x, fraud_y)

    # instantiate new figure
    fig, ax = plt.subplots(figsize=(15, 8))

    # plot reconstruction errors
    ax.scatter(clean_x, clean_y, s=20, color='g', alpha=0.6, label='Clean')
    ax.scatter(fraud_x, fraud_y, s=30, color='r', alpha=1, label='Fraud')

    # MAD threshold line
    ax.plot([threshold_value for i in range(len(mse))], color='orange', linewidth=1.5,
            label='MAD threshold')

    # change scale to log & limit x-axis range
    ax.set_yscale('log')
    ax.set_xlim(0, (len(mse) + 100))

    # title & labels
    fig.suptitle('Mean Squared Reconstruction Errors & MAD Threshold', fontsize=14)
    ax.set_xlabel('Pseudo Message ID\n(Index in MSE List)')
    ax.set_ylabel('Mean Squared Error\n(Log Scale)')

    # orange legend for threshold value
    ax.legend(loc='lower left', prop={'size': 9})

    # display
    fig.show()
    plt.show()
