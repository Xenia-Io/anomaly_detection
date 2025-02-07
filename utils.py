from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import multiprocessing
from time import time
from umap import UMAP
import seaborn as sns
import pandas as pd
import numpy as np
import string
import random
import gensim

sns.set(style='whitegrid', context='notebook')


def visualise_data(set_x, set_x_test, set_y, set_y_test, tsne=False, pca=False, umap=False):
    visualisation_initial = np.concatenate([set_x, set_x_test])
    visual_initial_y = np.concatenate([set_y, set_y_test])
    visual_dict = {'messages': list(visualisation_initial), 'labels': list(visual_initial_y)}

    visual_df = pd.DataFrame(visual_dict, columns=['messages', 'labels'])

    # isolate features from labels
    features, labels = visual_df.drop('labels', axis=1).values, visual_df.labels.values

    data_subset = visual_df.messages.values
    if tsne:
        tsne_scatter(data_subset, labels, dimensions=2)
    if pca:
        pca_scatter(data_subset, labels)
    if umap:
        umap_scatter(data_subset, labels)


def tsne_scatter(features, labels, dimensions=2, RANDOM_SEED=42):
    if dimensions not in (2, 3):
        raise ValueError('tsne_scatter can only plot in 2d or 3d')

    # dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=RANDOM_SEED).fit_transform(list(features))

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8, 10))

    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker='o',
        color='r',
        s=50,
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

    # Show results
    plt.title('TSNE representation of clean and fraud instances')
    plt.legend(loc='best', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def umap_scatter(features, labels, dimensions=2):
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
        s=37,
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

    # Show results
    plt.title('UMAP representation for clean and fraud instances')
    plt.legend(loc='best', fontsize=29)
    plt.show()


def pca_scatter(features, labels, dimensions=2):
    if dimensions not in (2, 3):
        raise ValueError('pca_scatter can only plot in 2d or 3d')

    # dimensionality reduction
    features_embedded = PCA(n_components=dimensions).fit_transform(list(features))

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8, 10))

    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # plotting data
    ax.scatter(
        *zip(*features_embedded[np.where(labels == 1)]),
        marker='o',
        color='r',
        s=55,
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

    # Show results
    plt.title('PCA representation for clean and fraud instances')
    plt.legend(loc='best', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def build_NLP_model(messages):
    w2v_model = gensim.models.Word2Vec(size=64, window=5, min_count=1, workers=10)
    t = time()

    w2v_model.build_vocab(messages, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 4)))

    t = time()

    w2v_model.train(messages, total_examples=w2v_model.corpus_count, epochs=10, compute_loss=True)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    # getting the training loss value
    training_loss = w2v_model.get_latest_training_loss()
    print("training_loss of word2vec: ", training_loss)

    cores = multiprocessing.cpu_count()
    print("Number of CPUs:", cores)

    # Make the model memory efficient
    w2v_model.init_sims(replace=True)

    pretrained_weights = w2v_model.wv.syn0
    vocab_size, emdedding_size = pretrained_weights.shape

    print('Result embedding shape:', pretrained_weights.shape)
    print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, emdedding_size))
    print("Vocabulary: ", w2v_model.wv.vocab.keys())

    return w2v_model, pretrained_weights, vocab_size, emdedding_size


def test_NLP_model(model):
    # Test w2v_model
    w1 = ["alarm"]
    print(model.wv.most_similar(positive=w1, topn=6))
    print(model.wv.similarity(w1, "set"))
    print(model.wv.similarity(w1, "cleared"))
    print(model.wv.similarity(w1, "unreachable"))
    print(model.wv.similarity(w1, "class"))

    w1 = ["server"]
    print(model.wv.most_similar(positive=w1, topn=6))
    print(model.wv.similarity(w1, "set"))
    print(model.wv.similarity(w1, "cleared"))
    print(model.wv.similarity(w1, "unreachable"))
    print(model.wv.similarity(w1, "class"))

    w1 = ["license"]
    print(model.wv.most_similar(positive=w1, topn=6))
    print(model.wv.similarity(w1, "set"))
    print(model.wv.similarity(w1, "cleared"))
    print(model.wv.similarity(w1, "unreachable"))
    print(model.wv.similarity(w1, "class"))

    w1 = ["chassis"]
    print(model.wv.most_similar(positive=w1, topn=6))
    print(model.wv.similarity(w1, "set"))
    print(model.wv.similarity(w1, "cleared"))
    print(model.wv.similarity(w1, "unreachable"))
    print(model.wv.similarity(w1, "class"))

    w1 = ["class"]
    print(model.wv.most_similar(positive=w1, topn=6))
    print(model.wv.similarity(w1, "set"))
    print(model.wv.similarity(w1, "cleared"))
    print(model.wv.similarity(w1, "unreachable"))
    print(model.wv.similarity(w1, "class"))


def prepare_data(preprocessor, df_copy, w2v_model, maxlen = 64, is_LSTM=False):
    print('\nPreparing the data for DL models...')
    DROP_THRESHOLD = 1

    # splitting by class
    fraud = df_copy[df_copy.labels == 1]
    clean = df_copy[df_copy.labels == 0]
    print(f"""Shape of the clean and fraud sets:
                   clean (rows, cols) = {clean.shape}
                   fraud (rows, cols) = {fraud.shape}""")

    # Build train and test datasets - training set
    num_train = int(preprocessor.train_ratio * clean.shape[0])
    if is_LSTM:
        partition_idx = len(fraud.index) / 2
        print(len(fraud.index))
        df_fraud_train = fraud.iloc[0:int(partition_idx), :]
        print("debug : ", len(df_fraud_train.index))
        df_copy_train = pd.concat([clean.iloc[0:num_train], df_fraud_train])

        print(df_copy_train)
        df_copy_test = pd.concat([clean.iloc[num_train:], fraud])
    else:
        df_copy_train = clean.iloc[0:num_train]
        df_copy_test = pd.concat([clean.iloc[num_train:], fraud])

    print(f"""Shape of the datasets:
                       training set (rows, cols) = {df_copy_train.shape}
                       test set (rows, cols) = {df_copy_test.shape}""")

    sequences_train = SequenceIterator(df_copy_train, DROP_THRESHOLD, maxlen, w2v_model)
    print("seq train: ", sequences_train)
    sequences_test = SequenceIterator(df_copy_test, DROP_THRESHOLD, maxlen, w2v_model)
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
    set_x = pad_sequences(set_x, maxlen = maxlen, padding='pre', value=0)
    set_y = np.array(set_y)
    set_x_test = pad_sequences(set_x_test, maxlen = maxlen, padding='pre', value=0)
    set_y_test = np.array(set_y_test)

    print("set_x.shape: ", set_x.shape, type(set_x))
    print("set_y.shape: ", set_y.shape, type(set_y))

    VALID_PER = 0.15  # Percentage of the whole set that will be separated for validation

    total_samples = df_copy_train.shape[0]
    total_samples_test = df_copy_test.shape[0]
    n_val = int(VALID_PER * total_samples)
    n_train = total_samples - n_val
    n_test = df_copy_test.shape[0]

    random_i = random.sample(range(total_samples), total_samples)
    print("df_copy_train.shape: ", df_copy_train.shape) # (7006, 4)

    train_x = set_x[random_i[:n_train]]
    train_y = set_y[random_i[:n_train]]
    val_x = set_x[random_i[n_train:]]
    val_y = set_y[random_i[n_train:]]
    test_x = set_x_test[:n_test]
    test_y = set_y_test[:n_test]

    print("Train Shapes - X: {} - Y: {}".format(train_x.shape, train_y.shape))
    print("Val Shapes - X: {} - Y: {}".format(val_x.shape, val_y.shape))
    print("Test Shapes - X: {} - Y: {}".format(test_x.shape, test_y.shape))

    return set_x, set_x_test, set_y, set_y_test, train_x, train_y, val_x, val_y, test_x, test_y


def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z_mean[0]")
    plt.ylabel("z_mean[1]")
    plt.title("A 2D plot of the classes in the latent space")
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

    def idx2word(self, word_model, idx):
        return word_model.wv.index2word[idx]

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
