import string
import random
from time import time
import multiprocessing
from pre_processor import Preprocessor
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
import pandas as pd
import numpy as np
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.models import Model, Sequential


class LSTM_Model():

    def __init__(self, epochs= 5, optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,\
                 epsilon=0, w_decay=0, momentum=0, dropout=0, embed_size = 300, max_features = 10000,\
                 maxlen = 200):
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


    def build_autoencoder(self, X, vocab_size, emdedding_size):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
        model.add(LSTM(units=emdedding_size))
        model.add(Dense(units=vocab_size))
        model.add(Activation('relu'))

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


class SequenceIterator:
    def __init__(self, dataset, drop_threshold, seq_length, word_model):
        self.dataset = dataset
        self.word_model = word_model

        self.translator = str.maketrans('', '', string.punctuation + '–')
        self.categories, self.ccount = np.unique(dataset.labels, return_counts=True)

        self.seq_length = seq_length

        # Samples of categories with less than this number of samples will be ignored
        self.drop_categos = []
        for cat, count in zip(self.categories, self.ccount):
            if count < drop_threshold:
                self.drop_categos.append(cat)

        # Remaining categories
        self.categories = np.setdiff1d(self.categories, self.drop_categos)

    def word2idx(self, word_model, word):
        try:
            return word_model.wv.vocab[word].index
            # If word is not in index return 0. I realize this means that this
            # is the same as the word of index 0 (i.e. most frequent word), but 0s
            # will be padded later anyway by the embedding layer (which also
            # seems dirty but I couldn't find a better solution right now)
        except KeyError:
            return 0

    def __iter__(self):
        i=0
        for news, cat in zip(self.dataset.iloc[:, 0], self.dataset.iloc[:, 1]):
            i = i+1
            if cat in self.drop_categos:
                continue

            # Make all characters lower-case
            news = news.lower()

            # Clean string of all punctuation
            news = news.translate(self.translator)

            words = np.array([self.word2idx(self.word_model, w) for w in news.split(' ')[:self.seq_length] if w != ''])

            yield (words, cat)


if __name__ == "__main__":
    # print(tf.__version__)
    #
    # print("Is there a GPU available: "),
    # print(tf.test.is_gpu_available())
    #
    # print("Is the Tensor on GPU #0:  "),
    # print(x.device.endswith('GPU:0'))
    #
    # print("Device name: {}".format((x.device)))
    # print(tf.executing_eagerly())
    # print(tf.keras.__version__)

    # Preprocessing the dataset
    preprocessor = Preprocessor('big_dataset.json', True, visualize= False, auto_encoder = True)
    df_copy = preprocessor.df.copy()
    messages = preprocessor.df['messages'].values

    for i in range(messages.shape[0]):
        # print("i: ", i, " with message: ", messages[i])
        # messages[i] = re.sub("[\(\[].*?[\)\]]", "", messages[i])
        messages[i] = ''.join([i for i in messages[i] if not i.isdigit()])
        messages[i] = gensim.utils.simple_preprocess (messages[i])
        # print("i: ", i, " with message: ", messages[i])


    # for i in preprocessor.df.index:
    #     preprocessor.df.at[i, "messages"] = messages[i]
    print("After gensim cleaning df_copy[0] : ", df_copy['messages'][0])
    print("After gensim cleaning preprocessor.df[0]: ", preprocessor.df['messages'][0])


    w2v_model = gensim.models.Word2Vec(size=200, window=5, min_count=1, workers=10)
    t = time()

    w2v_model.build_vocab(messages, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 4)))

    t = time()

    w2v_model.train(messages, total_examples=w2v_model.corpus_count, epochs=10)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    cores = multiprocessing.cpu_count()
    print("Number of CPUs:", cores)

    # Make the model memory efficient
    w2v_model.init_sims(replace=True)

    # Test w2v_model
    w1 = ["servers"]
    # print(list(X_test['messages']))
    print(w2v_model.wv.most_similar(positive=w1, topn=6))
    print(w2v_model.wv.similarity(w1, "no"))
    print(w2v_model.wv.similarity(w1, "reachable"))
    print(w2v_model.wv.similarity(w1, "alarm"))

    pretrained_weights = w2v_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape
    print('Result embedding shape:', pretrained_weights.shape)
    print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, emdedding_size))
    print("Vocabulary: ", w2v_model.wv.vocab.keys())

    # splitting by class
    fraud = df_copy[df_copy.labels == 1]
    clean = df_copy[df_copy.labels == 0]
    print(f"""Shape of the clean and fraud sets:
               clean (rows, cols) = {clean.shape}
               fraud (rows, cols) = {fraud.shape}""")

    # Data preparation: build train and test datasets
    # training set: exlusively non-fraud transactions
    num_train = int(preprocessor.train_ratio * clean.shape[0])
    df_copy_train = clean.iloc[0:num_train]
    X_train = clean.iloc[0:num_train]
    y_train = X_train['labels'].values
    # X_train = X_train.drop('labels', axis=1)
    X_test = pd.concat([clean.iloc[num_train:], fraud])
    df_copy_test = pd.concat([clean.iloc[num_train:], fraud])
    y_test = X_test['labels'].values
    # X_test = X_test.drop('labels', axis=1)

    print(f"""Shape of the datasets:
                   X_train (rows, cols) = {X_train.shape}
                   X_test (rows, cols) = {X_test.shape}""")
    print("xxx 1 :", type(X_train), X_train.shape, len(X_train['messages'][0]))

    # Build auto-encoder
    lstm = LSTM_Model()
    DROP_THRESHOLD = 1

    print('\nPreparing the data for LSTM...')
    sequences_train = SequenceIterator(df_copy_train, DROP_THRESHOLD, lstm.maxlen, w2v_model)
    sequences_test = SequenceIterator(df_copy_test, DROP_THRESHOLD, lstm.maxlen, w2v_model)
    # Used for generating the labels in the set
    cat_dict = {k: v for k, v in zip(sequences_train.categories, range(len(sequences_train.categories)))}
    cat_dict_test = {k: v for k, v in zip(sequences_test.categories, range(len(sequences_test.categories)))}

    set_x = []
    set_y = []
    for w, c in sequences_train:
        set_x.append(w)
        set_y.append(cat_dict[c])

    set_x_test = []
    set_y_test = []
    for w, c in sequences_test:
        set_x_test.append(w)
        set_y_test.append(cat_dict_test[c])

    # Padding sequences with 0.
    set_x = pad_sequences(set_x, maxlen=lstm.maxlen, padding='pre', value=0)
    set_y = np.array(set_y)
    set_x_test = pad_sequences(set_x_test, maxlen=55, padding='pre', value=0)
    set_y_test = np.array(set_y_test)

    print("set_x.shape: ", set_x.shape, type(set_x))
    print("set_y.shape: ", set_y.shape, type(set_y))

    # print("set_x_test.shape: ", set_x_test.shape, type(set_x_test))
    # print("set_y_test.shape: ", set_y_test.shape, type(set_y_test))

    VALID_PER = 0.15  # Percentage of the whole set that will be separated for validation

    total_samples = df_copy_train.shape[0]
    total_samples_test = df_copy_test.shape[0]
    n_val = int(VALID_PER * total_samples)
    n_train = total_samples - n_val
    n_test = df_copy_test.shape[0]

    random_i = random.sample(range(total_samples), total_samples)
    random_j = random.sample(range(total_samples_test), total_samples_test)
    # clean.iloc[0:num_train]
    print("POPI: ", df_copy_train.shape)
    train_x = set_x[random_i[:n_train]]
    train_y = set_y[random_i[:n_train]]
    val_x = set_x[random_i[n_train:n_train + n_val]]
    val_y = set_y[random_i[n_train:n_train + n_val]]
    test_x = set_x_test[random_j[:n_test]]
    test_y = set_y_test[random_j[:n_test]]


    print("Train Shapes - X: {} - Y: {}".format(train_x.shape, train_y.shape))
    print("Val Shapes - X: {} - Y: {}".format(val_x.shape, val_y.shape))
    print("Test Shapes - X: {} - Y: {}".format(test_x.shape, test_y.shape))
    print(df_copy_test[df_copy_test.labels == 1])

    # Compile model
    model_ = lstm.build_autoencoder(train_x, vocab_size, emdedding_size)
    model_.compile(loss='binary_crossentropy', optimizer='adam')
    # # model_.compile(optimizer='adam', loss='mse')
    model_.summary()
    print("HEREEEE..............", test_x.shape)
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
    # calculating the mean squared error reconstruction loss per row in the numpy array
    mse = np.mean(np.power(test_x - reconstructions, 2), axis=1)

    # showing the reconstruction losses for a subsample of transactions
    print(f'Mean Squared Error reconstruction losses for {5} clean transactions:')
    print(mse[np.where(test_y == 0)][:5])
    print(f'\nMean Squared Error reconstruction losses for {5} fraudulent transactions:')
    print(mse[np.where(test_y == 1)][:5])
