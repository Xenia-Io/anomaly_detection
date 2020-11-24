from sklearn.tree import plot_tree
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Lambda
from tensorflow.python.keras.layers import RepeatVector, TimeDistributed, Layer, Reshape
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Model, Sequential
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import SGD
from kerastuner.tuners import RandomSearch
import tensorflow.keras.losses as tf_losses
import tensorflow.keras.backend as bck
import tensorflow_probability as tfp
from kerastuner import HyperModel
from sklearn import metrics
import kerastuner as kt
import tensorflow as tf
from utils import *
import time

from matplotlib.colors import ListedColormap
# Import statements required for Plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

tfd = tfp.distributions


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, epsilon_std=1, timesteps = 1, epochs= 1, latent_dim = 32, intermediate_dim = 96, \
                 optimizer='adam', loss='mae', batch_size=1, kernel_init=0, gamma=0, epsilon=0, \
                 w_decay=0, momentum=0, dropout=0, embed_size = 300, max_features = 10000, maxlen = 64, **kwargs):
        super(VAE, self).__init__(**kwargs)
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
        self.timesteps = timesteps  # input timestep dimension
        self.intermediate_dim = intermediate_dim  # output shape of LSTM
        self.latent_dim = latent_dim  # latent z-layer shape
        self.epsilon_std = epsilon_std  # z-layer sigma
        self.emdedding_size = 200  # z-layer sigma

        self.encoder, self.decoder = self.build_encoder_decoder()


    def build_encoder_decoder(self):
        x = Input(batch_shape=(None, emdedding_size))
        x_embed = Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights],
                            trainable=False)(x)
        print("x_embed: ", x_embed)

        h = LSTM(self.intermediate_dim, return_sequences=False, recurrent_dropout=0.2)(x_embed)

        z_mean = Dense(self.latent_dim, name="z_mean")(h)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(h)

        z = Sampling()([z_mean, z_log_var])

        encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # build a generator that can sample sentences from the learned distribution
        # we instantiate these layers separately so as to reuse them later
        repeated_context = RepeatVector(emdedding_size)
        decoder_h = LSTM(self.intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
        decoder_mean = Dense(emdedding_size, activation='linear', name='decoder_mean')
        h_decoded = decoder_h(repeated_context(z))
        x_decoded_mean = decoder_mean(h_decoded)

        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = decoder_h(repeated_context(decoder_input))
        _x_decoded_mean = decoder_mean(_h_decoded)
        _x_decoded_mean = Activation('relu', name="relu")(_x_decoded_mean)
        _x_decoded_out = Reshape((4096,))(_x_decoded_mean)
        _x_decoded_out = Dense(emdedding_size, activation='linear', name='decoder_out')(_x_decoded_out)
        decoder = Model(decoder_input, _x_decoded_out, name="decoder")

        decoder.summary()

        return encoder, decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction_mean = self.decoder(z)
            likelihood = tfd.Normal(loc=reconstruction_mean, scale=1.0)
            neg_log_likelihood = -1.0 * likelihood.log_prob(tf.cast(data, tf.float32))
            reconstruction_loss = tf.reduce_mean(neg_log_likelihood)

            kl_loss = tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
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

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        likelihood = tfd.Normal(loc=reconstruction, scale=1.0)
        neg_log_likelihood = -1.0 * likelihood.log_prob(tf.cast(data, tf.float32))
        reconstruction_loss = tf.reduce_mean(neg_log_likelihood)

        kl_loss = tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = (reconstruction_loss + kl_loss)
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


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


def plot_decision_boundary(mse_test_data, test_y):
    # Parameters
    n_classes = 3
    n_estimators = 30
    cmap = plt.cm.RdYlBu
    plot_step = 0.02  # fine step width for decision surface contours
    plot_step_coarser = 0.5  # step widths for coarse classifier guesses
    RANDOM_SEED = 13  # fix the seed on each iteration

    plot_idx = 1

    models = [DecisionTreeClassifier(max_depth=None),
              RandomForestClassifier(n_estimators=n_estimators),
              ExtraTreesClassifier(n_estimators=n_estimators),
              AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                 n_estimators=n_estimators)]

    for model in models:

        # Shuffle
        idx = np.arange(mse_test_data.shape[0])
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(idx)
        mse_test_data = mse_test_data[idx]
        y = test_y[idx]

        # Standardize
        mean = mse_test_data.mean(axis=0)
        std = mse_test_data.std(axis=0)
        mse_test_data = (mse_test_data - mean) / std

        # Train
        model.fit(mse_test_data, y)

        scores = model.score(mse_test_data, y)
        # Create a title for each column and the console by using str() and
        # slicing away useless parts of the string
        model_title = str(type(model)).split(
            ".")[-1][:-2][:-len("Classifier")]

        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(
                len(model.estimators_))
        print(model_details, " has a score of ", scores)

        plt.subplot(2, 2, plot_idx)

        if plot_idx <= len(models):
            # Add a title at the top of each column
            plt.title(model_title, fontsize=9)

        # Now plot the decision boundary using a fine mesh as input to a
        # filled contour plot
        x_min, x_max = mse_test_data[:, 0].min() - 1, mse_test_data[:, 0].max() + 1
        y_min, y_max = mse_test_data[:, 1].min() - 1, mse_test_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        # Plot either a single DecisionTreeClassifier or alpha blend the
        # decision surfaces of the ensemble of classifiers
        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap)
        else:
            # Choose alpha blend level with respect to the number
            # of estimators
            # that are in use (noting that AdaBoost can use fewer estimators
            # than its maximum if it achieves a good enough fit early on)
            estimator_alpha = 1.0 / len(model.estimators_)
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, alpha=estimator_alpha, cmap=cmap)

        # Build a coarser grid to plot a set of ensemble classifications
        # to show how these are different to what we see in the decision
        # surfaces. These points are regularly space and do not have a
        # black outline
        xx_coarser, yy_coarser = np.meshgrid(
            np.arange(x_min, x_max, plot_step_coarser),
            np.arange(y_min, y_max, plot_step_coarser))
        Z_points_coarser = model.predict(np.c_[xx_coarser.ravel(),
                                               yy_coarser.ravel()]
                                         ).reshape(xx_coarser.shape)
        cs_points = plt.scatter(xx_coarser, yy_coarser, s=15,
                                c=Z_points_coarser, cmap=cmap,
                                edgecolors="none")

        # Plot the training points, these are clustered together and have a
        # black outline
        plt.scatter(mse_test_data[:, 0], mse_test_data[:, 1], c=y,
                    cmap=ListedColormap(['r', 'y', 'b']),
                    edgecolor='k', s=20)
        plot_idx += 1  # move on to the next plot in sequence

    plt.suptitle("Classifiers", fontsize=12)
    plt.axis("tight")
    plt.tight_layout(h_pad=0.2, w_pad=0.2, pad=2.5)
    plt.show()


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    print(tf.__version__)
    print(tf.executing_eagerly())
    print(tf.keras.__version__)

    # Preprocessing the dataset
    preprocessor = Preprocessor('./data/big_dataset.json', True, visualize= False)
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
                    val_y, test_x, test_y= prepare_data(preprocessor, df_copy, w2v_model)


    vae = VAE()
    vae.compile(optimizer= 'sgd')
    # vae.summary()

    # Visualization
    visualise_data(set_x, set_x_test, set_y, set_y_test, tsne=False, pca=False, umap=False)


    # Train the model
    history = vae.fit(
        train_x, train_x,
        epochs=1,
        batch_size=10,
        validation_data=(val_x, val_x),
        shuffle=True
    ).history

    # Plot the training and validation loss
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='train_loss', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='val_loss', linewidth=2)
    ax.plot(history['kl_loss'], 'g', label='kl_loss', linewidth=2)
    ax.plot(history['reconstruction_loss'], 'o', color='green',  label=' reconstruction loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.show()

    # plot_label_clusters(vae.encoder, train_x, train_y)


    # Test the mode
    encoded_inputs = vae.encoder.predict(test_x)
    reconstructions = vae.decoder.predict(encoded_inputs)
    # print("reconstructions and train_x shapes : ", reconstructions.shape, test_x.shape) # (3006, 64) both
    # print("reconstruction sample[0]: ", reconstructions[0])
    # print("test_x sample[0]: ", test_x[0])
    # print("test_y sample[0]: ", test_y[0])
    # print("test_x sample[102]: ", test_x[102])
    # print("test_y sample[102]: ", test_y[102])
    # print(type(reconstructions), type(test_x))
    # print(type(reconstructions[0]), type(test_x[0]))
    encoded_inputs = vae.encoder.predict(train_x)
    reconstructions_train = vae.decoder.predict(encoded_inputs)
    mse_train = np.mean(np.power(train_x - reconstructions_train, 2), axis=1)
    mse_test = np.mean(np.power(test_x - reconstructions, 2), axis=1)
    mse_test_data = np.column_stack((range(len(mse_test)), mse_test))

    # scatter's x & y
    clean_x, clean_y = mse_test_data[test_y == 0][:, 0], mse_test_data[test_y == 0][:, 1]
    fraud_x, fraud_y = mse_test_data[test_y == 1][:, 0], mse_test_data[test_y == 1][:, 1]
    print("clean x,y : ", clean_x, clean_y)
    print("fraud x,y : ", fraud_x, fraud_y)

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(max_depth=4, n_estimators=20)

    print(train_x.shape, mse_train.shape, train_y.shape)
    # Train the model using the training set
    clf.fit(mse_train.reshape(-1, 1), train_y)

    # prediction on test set
    y_pred = clf.predict(mse_test.reshape(-1, 1))

    # Model Accuracy, how often is the classifier correct?
    print("RandomForest classifier's Accuracy: ", metrics.accuracy_score(test_y, y_pred))

    # Plot decision boundary
    plot_decision_boundary(mse_test_data, test_y)

    exit(0)




    # calculating the mean squared error reconstruction loss per row in the numpy array
    mse = np.mean(np.power(test_x - reconstructions, 2), axis=1)

    # showing the reconstruction losses for a subsample of transactions
    # print(f'Mean Squared Error reconstruction losses')
    # print(mse)

    sorted_index_array = np.argsort(mse)

    # sorted array
    sorted_array = mse[sorted_index_array]

    # we want 3 largest value
    rslt = sorted_array[-10:]

    # show the output
    print("{} largest value:".format(10),
          rslt)

    # adjust this parameter to customise the recall/precision trade-off
    Z_SCORE_THRESHOLD = 33

    # find the outliers on our reconstructions' mean squared errors
    mad_z_scores, threshold_value = vae.detect_mad_outliers(mse, threshold=Z_SCORE_THRESHOLD)
    mad_outliers = (mad_z_scores > Z_SCORE_THRESHOLD).astype(int)
    print("mad outliers of our reconstructions' MSE: ", mad_outliers)

    anomalies = len(mad_outliers[mad_outliers == True])
    total_trades = len(test_y)
    d = (anomalies / total_trades * 100)

    print("MAD Z-score > ", Z_SCORE_THRESHOLD, " is the selected threshold.")
    print("Any trade with a MSRE >= ", threshold_value, " is flagged.")
    print("This results in", anomalies, "detected anomalies, or ", d, "% out of ", total_trades, "trades reported")

    data = np.column_stack((range(len(mse)), mse))
    print("data =", data)
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

