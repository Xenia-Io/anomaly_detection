import string
import random
import sys
from time import time
import multiprocessing
from pre_processor import Preprocessor
import seaborn as sns
from umap import UMAP
sns.set(style='whitegrid', context='notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gensim
import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.losses as tf_losses
import tensorflow.keras.backend as bck
from tensorflow.keras import layers, regularizers
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Layer, Reshape
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Lambda, GRU, ELU
from tensorflow.keras.models import Model, Sequential
import tensorflow_probability as tfp

tfd = tfp.distributions


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction_mean = decoder(z)
            likelihood = tfd.Normal(loc=reconstruction_mean, scale=1.0)
            neg_log_likelihood = -1.0 * likelihood.log_prob(tf.cast(data, tf.float32))
            reconstruction_loss = tf.reduce_mean(neg_log_likelihood)

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = (reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        likelihood = tfd.Normal(loc=reconstruction, scale=1.0)
        neg_log_likelihood = -1.0 * likelihood.log_prob(tf.cast(data, tf.float32))
        reconstruction_loss = tf.reduce_mean(neg_log_likelihood)

        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = (reconstruction_loss + kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

class Autoencoder_Model():

    def __init__(self, epsilon_std=1, timesteps = 1, epochs= 1, latent_dim = 32, intermediate_dim = 96, \
                 optimizer='adam', loss='mae', batch_size=1, kernel_init=0, gamma=0, epsilon=0, \
                 w_decay=0, momentum=0, dropout=0, embed_size = 300, max_features = 10000, maxlen = 64):
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
        # self.input_dim = input_dim
        self.timesteps = timesteps # input timestep dimension
        self.intermediate_dim = intermediate_dim # output shape of LSTM
        self.latent_dim = latent_dim # latent z-layer shape
        self.epsilon_std = epsilon_std # z-layer sigma
        self.emdedding_size = 200 # z-layer sigma



    def detect_mad_outliers(self, points, threshold=3.5):
        # calculate the median of the input array
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


    def tsne_scatter(self, features, labels, dimensions=2, RANDOM_SEED = 42):
        if dimensions not in (2, 3):
            raise ValueError('tsne_scatter can only plot in 2d or 3d')

        # dimensionality reduction
        features_embedded = TSNE(n_components=dimensions, random_state=RANDOM_SEED).fit_transform(list(features))

        # initialising the plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # counting dimensions
        if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

        # plotting data
        ax.scatter(
            *zip(*features_embedded[np.where(labels == 1)]),
            marker='o',
            color='r',
            s=20,
            alpha=0.99,
            label='Fraud'
        )
        ax.scatter(
            *zip(*features_embedded[np.where(labels == 0)]),
            marker='o',
            color='g',
            s=20,
            alpha=0.99,
            label='Clean'
        )

        # storing it to be displayed later
        plt.legend(loc='best')
        plt.title('TSNE representation of clean and fraud instances')
        plt.show()


    def umap_scatter(self, features, labels, dimensions=2):
            if dimensions not in (2, 3):
                raise ValueError('umap_scatter can only plot in 2d or 3d')

            # dimensionality reduction
            features_embedded = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(list(features))

            # initialising the plot
            fig, ax = plt.subplots(figsize=(8, 8))

            # counting dimensions
            if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

            # plotting data
            ax.scatter(
                *zip(*features_embedded[np.where(labels == 1)]),
                marker='o',
                color='r',
                s=10,
                alpha=0.99,
                label='Fraud'
            )
            ax.scatter(
                *zip(*features_embedded[np.where(labels == 0)]),
                marker='o',
                color='g',
                s=10,
                alpha=0.99,
                label='Clean'
            )

            # storing it to be displayed later
            plt.legend(loc='best')
            plt.title('UMAP representation for clean and fraud instances')
            plt.show()


    def pca_scatter(self, features, labels, dimensions=2):
            if dimensions not in (2, 3):
                raise ValueError('pca_scatter can only plot in 2d or 3d')

            # dimensionality reduction
            features_embedded = PCA(n_components=dimensions).fit_transform(list(features))

            # initialising the plot
            fig, ax = plt.subplots(figsize=(8, 8))

            # counting dimensions
            if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

            # plotting data
            ax.scatter(
                *zip(*features_embedded[np.where(labels == 1)]),
                marker='o',
                color='r',
                s=10,
                alpha=0.99,
                label='Fraud'
            )
            ax.scatter(
                *zip(*features_embedded[np.where(labels == 0)]),
                marker='o',
                color='g',
                s=10,
                alpha=0.99,
                label='Clean'
            )

            # storing it to be displayed later
            plt.legend(loc='best')
            plt.title('PCA representation for clean and fraud instances')
            plt.show()



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
            # print("WORD = ", word)
            # print("WORD2 IDX:", word_model.wv.vocab[word].index)
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
    tf.config.experimental_run_functions_eagerly(True)
    print(tf.__version__)
    # tf.compat.v1.InteractiveSession()

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
    preprocessor = Preprocessor('big_dataset.json', True, visualize= False, dnn = True)
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


    # w2v_model = gensim.models.Word2Vec(size=200, window=5, min_count=1, workers=10)
    w2v_model = gensim.models.Word2Vec(size=64, window=5, min_count=1, workers=10)
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
                   training set (rows, cols) = {df_copy_train.shape}
                   test set (rows, cols) = {df_copy_test.shape}""")
    # print("xxx 1 :", type(X_train), X_train.shape, len(X_train['messages'][0]))

    # Build auto-encoder
    autoencoder = Autoencoder_Model()
    DROP_THRESHOLD = 1

    print('\nPreparing the data for LSTM...')
    sequences_train = SequenceIterator(df_copy_train, DROP_THRESHOLD, autoencoder.maxlen, w2v_model)
    print("seq train: ", sequences_train)
    sequences_test = SequenceIterator(df_copy_test, DROP_THRESHOLD, autoencoder.maxlen, w2v_model)
    # Used for generating the labels in the set
    cat_dict = {k: v for k, v in zip(sequences_train.categories, range(len(sequences_train.categories)))}
    cat_dict_test = {k: v for k, v in zip(sequences_test.categories, range(len(sequences_test.categories)))}

    set_x = []
    set_y = []
    for w, c in sequences_train:
        # print("w = ", w)
        # print("c = ", c)
        set_x.append(w)
        set_y.append(cat_dict[c])

    set_x_test = []
    set_y_test = []
    for w, c in sequences_test:
        set_x_test.append(w)
        set_y_test.append(cat_dict_test[c])

    # Padding sequences with 0.
    set_x = pad_sequences(set_x, maxlen=autoencoder.maxlen, padding='pre', value=0)
    set_y = np.array(set_y)
    # set_x_test = pad_sequences(set_x_test, maxlen=55, padding='pre', value=0)
    set_x_test = pad_sequences(set_x_test, maxlen=autoencoder.maxlen, padding='pre', value=0)
    set_y_test = np.array(set_y_test)

    print("set_x.shape: ", set_x.shape, type(set_x))
    print("set_y.shape: ", set_y.shape, type(set_y))

    # print("set_x_test.shape: ", set_x_test.shape, type(set_x_test))
    # print("set_y_test.shape: ", set_y_test.shape, type(set_y_test))

    VALID_PER = 0.15  # Percentage of the whole set that will be separated for validation

    total_samples = df_copy_train.shape[0]
    total_samples_test = df_copy_test.shape[0]
    n_val = int(VALID_PER * total_samples)
    # n_train = total_samples - n_val
    n_train = 200
    n_test = df_copy_test.shape[0]

    random_i = random.sample(range(total_samples), total_samples)
    random_j = random.sample(range(total_samples_test), total_samples_test)
    print("df_copy_train.shape: ", df_copy_train.shape)
    train_x = set_x[random_i[:n_train]]
    train_y = set_y[random_i[:n_train]]
    val_x = set_x[random_i[n_train:n_train + 4]]
    val_y = set_y[random_i[n_train:n_train + 4]]
    test_x = set_x_test[random_j[:n_test]]
    test_y = set_y_test[random_j[:n_test]]


    print("Train Shapes - X: {} - Y: {}".format(train_x.shape, train_y.shape))
    print("Val Shapes - X: {} - Y: {}".format(val_x.shape, val_y.shape))
    print("Test Shapes - X: {} - Y: {}".format(test_x.shape, test_y.shape))
    # print(df_copy_test[df_copy_test.labels == 1])

    print("train_x shape: ", train_x.shape)
    # train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    # test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
    print("train_x shape: ", train_x.shape)
    print("sample 0 ", len(train_x[0]))

    # Compile model
    # encoder, decoder = autoencoder.build_autoencoder(train_x, vocab_size, emdedding_size, pretrained_weights)
    x = Input(batch_shape=(None, emdedding_size))
    # x = Input(batch_shape=(self.timesteps, emdedding_size,))
    # print("X shape now : ", x.shape) --> (None, 64)
    # print("pretrained_weights shape now : ", pretrained_weights.shape) --> (55, 64)
    x_embed = Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights],
                        trainable=False)(x)
    print("x_embed: ", x_embed)
    # print("X embed shape now : ", x_embed.shape) -->  (None, 64, 64)

    h = LSTM(autoencoder.intermediate_dim, return_sequences=False, recurrent_dropout=0.2)(x_embed)

    z_mean = Dense(autoencoder.latent_dim, name="z_mean")(h)
    z_log_var = Dense(autoencoder.latent_dim, name="z_log_var")(h)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Sampling()([z_mean, z_log_var])
    # z = Lambda(sampling, name="sampling")([z_mean, z_log_var])

    encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # build a generator that can sample sentences from the learned distribution
    # we instantiate these layers separately so as to reuse them later
    repeated_context = RepeatVector(emdedding_size)
    decoder_h = LSTM(autoencoder.intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
    decoder_mean = Dense(emdedding_size, activation='linear', name='decoder_mean')
    h_decoded = decoder_h(repeated_context(z))
    x_decoded_mean = decoder_mean(h_decoded)

    decoder_input = Input(shape=(autoencoder.latent_dim,))
    _h_decoded = decoder_h(repeated_context(decoder_input))
    _x_decoded_mean = decoder_mean(_h_decoded)
    _x_decoded_mean = Activation('relu', name="relu")(_x_decoded_mean)
    _x_decoded_out = Reshape((4096,))(_x_decoded_mean)
    _x_decoded_out = Dense(emdedding_size, activation='linear', name='decoder_out')(_x_decoded_out)
    decoder = Model(decoder_input, _x_decoded_out, name="decoder")
    # latent_inputs = Input(shape=(self.latent_dim,))
    # decoder_outputs = GRU(self.intermediate_dim, return_sequences=False, recurrent_dropout=0.2)(latent_inputs)
    # decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer='sgd')
    # vae.summary()
    # print("test_x.shape: ", test_x.shape) ---> test_x.shape:  (3006, 55)

    # Visualization
    visualisation_initial = np.concatenate([set_x, set_x_test])
    visual_initial_y = np.concatenate([set_y, set_y_test])
    visual_dict = {'messages': list(visualisation_initial), 'labels': list(visual_initial_y)}

    visual_df = pd.DataFrame(visual_dict, columns=['messages', 'labels'])
    print("Debug : ", visualisation_initial.shape, visual_initial_y.shape)
    # print(visual_df) ----> [10006 rows x 2 columns (messages and lables) ]

    # isolate features from labels
    features, labels = visual_df.drop('labels', axis=1).values, visual_df.labels.values

    data_subset = visual_df.messages.values
    # autoencoder.tsne_scatter(data_subset, labels, dimensions=2)
    # autoencoder.pca_scatter(data_subset, labels)
    # autoencoder.umap_scatter(data_subset, labels)



    # Train the model
    history = vae.fit(
        train_x, train_x,
        epochs=7,
        batch_size=10,
        validation_data=(val_x, val_x),
        shuffle=True
    ).history

    # Plot the training and validation loss
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='train_loss', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='val_loss', linewidth=2)
    # ax.plot(history['kl_loss'], 'r', label='kl_loss', linewidth=2)
    # ax.plot(history['reconstruction_loss'], 'r', color='green',  label=' reconstruction loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.show()



    def plot_label_clusters(encoder, decoder, data, labels):
        # display a 2D plot of the classes in the latent space
        z_mean, _, _ = encoder.predict(data)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.title("A 2D plot of the classes in the latent space")
        plt.show()


    plot_label_clusters(encoder, decoder, train_x, train_y)
    # # Test the mode
    # reconstructions = vae.predict(train_x)
    # print("reconstructions and train_x shapes : ", reconstructions.shape, train_x.shape)
    # # calculating the mean squared error reconstruction loss per row in the numpy array
    # mse = np.mean(np.power(train_x - reconstructions, 2), axis=1)
    #
    # # showing the reconstruction losses for a subsample of transactions
    # print(f'Mean Squared Error reconstruction losses for {5} clean transactions:')
    # print(mse[np.where(train_y == 0)][:5])
    # print(f'\nMean Squared Error reconstruction losses for {5} fraudulent transactions:')
    # print(mse[np.where(train_y == 1)][:5])

    # # adjust this parameter to customise the recall/precision trade-off
    # Z_SCORE_THRESHOLD = 3
    #
    # # find the outliers on our reconstructions' mean squared errors
    # mad_z_scores, threshold_value = vae.detect_mad_outliers(mse, threshold=Z_SCORE_THRESHOLD)
    # mad_outliers = (mad_z_scores > Z_SCORE_THRESHOLD).astype(int)
    # print("mad outliers of our reconstructions' MSE: ", mad_outliers)
    #
    # anomalies = len(mad_outliers[mad_outliers == True])
    # total_trades = len(test_y)
    # d = (anomalies / total_trades * 100)
    #
    # print("MAD Z-score > ", Z_SCORE_THRESHOLD, " is the selected threshold.")
    # print("Any trade with a MSRE >= ", threshold_value, " is flagged.")
    # print("This results in", anomalies, "detected anomalies, or ", d, "% out of ", total_trades, "trades reported")
    #
    # data = np.column_stack((range(len(mse)), mse))
    # print("data =", type(data), data)
    # # scatter's x & y
    # clean_x, clean_y = data[test_y == 0][:, 0], data[test_y == 0][:, 1]
    # fraud_x, fraud_y = data[test_y == 1][:, 0], data[test_y == 1][:, 1]
    # print("clean x,y : ", clean_x, clean_y)
    # print("fraud x,y : ", fraud_x, fraud_y)
    #
    # # instantiate new figure
    # fig, ax = plt.subplots(figsize=(15, 8))
    #
    # # plot reconstruction errors
    # ax.scatter(clean_x, clean_y, s=20, color='g', alpha=0.6, label='Clean')
    # ax.scatter(fraud_x, fraud_y, s=30, color='r', alpha=1, label='Fraud')
    #
    # # MAD threshold line
    # ax.plot([threshold_value for i in range(len(mse))], color='orange', linewidth=1.5,
    #         label='MAD threshold')
    #
    # # change scale to log & limit x-axis range
    # ax.set_yscale('log')
    # ax.set_xlim(0, (len(mse) + 100))
    #
    # # title & labels
    # fig.suptitle('Mean Squared Reconstruction Errors & MAD Threshold', fontsize=14)
    # ax.set_xlabel('Pseudo Message ID\n(Index in MSE List)')
    # ax.set_ylabel('Mean Squared Error\n(Log Scale)')
    #
    # # orange legend for threshold value
    # ax.legend(loc='lower left', prop={'size': 9})
    #
    # # display
    # fig.show()
    # plt.show()
    #
