from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Lambda
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from tensorflow.python.keras.layers import RepeatVector, Reshape
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from tensorflow.keras.models import Model
from kerastuner.tuners import RandomSearch
import tensorflow.keras.losses as tf_losses
from sklearn.metrics import accuracy_score
from pre_processor import Preprocessor
import tensorflow.keras.backend as bck
import tensorflow_probability as tfp
from sklearn.tree import plot_tree
from sklearn.utils import shuffle
from kerastuner import HyperModel
from sklearn import metrics, svm
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
from sklearn import tree
import tensorflow as tf
from utils import *
import warnings
import graphviz
import time
import os

tfd = tfp.distributions


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(tf.keras.Model):
    def __init__(self, pretrained_weights, emdedding_size, vocab_size, epsilon_std=1, timesteps = 1, epochs= 1, \
                 optimizer='adam', loss='mae', batch_size=1, kernel_init=0, gamma=0, epsilon=0, latent_dim = 32, \
                 w_decay=0, momentum=0, dropout=0, embed_size = 300, max_features = 10000, intermediate_dim = 96,\
                 maxlen = 64, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
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
        self.emdedding_size = emdedding_size  # z-layer sigma
        self.pretrained_weights = pretrained_weights
        self.vocab_size = vocab_size

        self.encoder, self.decoder = self.build_encoder_decoder()


    def build_encoder_decoder(self):
        x = Input(batch_shape=(None, self.emdedding_size))
        x_embed = Embedding(input_dim=self.vocab_size, output_dim=self.emdedding_size, weights=[self.pretrained_weights],
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
        repeated_context = RepeatVector(self.emdedding_size)
        decoder_h = LSTM(self.intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
        decoder_mean = Dense(self.emdedding_size, activation='linear', name='decoder_mean')
        h_decoded = decoder_h(repeated_context(z))
        x_decoded_mean = decoder_mean(h_decoded)

        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = decoder_h(repeated_context(decoder_input))
        _x_decoded_mean = decoder_mean(_h_decoded)
        _x_decoded_mean = Activation('relu', name="relu")(_x_decoded_mean)
        _x_decoded_out = Reshape((4096,))(_x_decoded_mean)
        _x_decoded_out = Dense(self.emdedding_size, activation='linear', name='decoder_out')(_x_decoded_out)
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


def splitting_sets(mse_train_val, mse_test, train_y, val_y, test_y):
    mse_all = np.concatenate((mse_train_val, mse_test), axis=0)
    mse_all_data = np.column_stack((range(len(mse_all)), mse_all))
    y_train_val = np.concatenate((train_y, val_y), axis=0)
    y_all = np.concatenate((y_train_val, test_y), axis=0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(mse_all_data[:, 1], mse_all_data[:, 0], c=y_all, cmap=cm_bright, alpha=0.6)
    ax.set_xlabel("MSE values")
    ax.set_ylabel("Samples")
    plt.title("2D plot MSE values grouped by class for each sample")
    plt.show()

    # scatter's x & y
    clean_x, clean_y = mse_all_data[y_all == 0][:, 0], mse_all_data[y_all == 0][:, 1]
    fraud_x, fraud_y = mse_all_data[y_all == 1][:, 0], mse_all_data[y_all == 1][:, 1]
    plt.scatter(clean_x, clean_y)
    plt.scatter(fraud_x, fraud_y)
    plt.show()

    _df = pd.DataFrame({'mse': mse_all, 'labels': y_all})
    fraud = _df[_df.labels == 1]
    clean = _df[_df.labels == 0]

    num_train = int(0.6 * clean.shape[0])

    partition_idx = len(fraud.index) / 2
    df_fraud_train = fraud.iloc[0:int(partition_idx), :]
    df_copy_train = pd.concat([clean.iloc[0:num_train], df_fraud_train])
    df_copy_test = pd.concat([clean.iloc[num_train:], fraud])

    mse_train = df_copy_train.mse.values
    mse_train_data = np.column_stack((range(len(mse_train)), mse_train))
    y_train_ = df_copy_train.labels.values

    mse_test = df_copy_test.mse.values
    mse_test_data = np.column_stack((range(len(mse_test)), mse_test))
    y_test_ = df_copy_test.labels.values

    return mse_all_data, y_all, mse_train_data, y_train_, mse_test_data, y_test_


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, title = None, resolution=10.0):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.title(title)


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    print(tf.__version__)
    print(tf.executing_eagerly())
    print(tf.keras.__version__)

    # Preprocessing the dataset
    preprocessor = Preprocessor('../data/dataset_100k.json', True, visualize= False)
    df = preprocessor.load_data_supervised('../data/dataset_100k.json')
    df_copy = df.copy()
    messages = df['messages'].values

    for i in range(messages.shape[0]):
        messages[i] = ''.join([i for i in messages[i] if not i.isdigit()])
        messages[i] = gensim.utils.simple_preprocess(messages[i])

    w2v_model, pretrained_weights, vocab_size, emdedding_size = build_NLP_model(messages)
    # test_NLP_model(w2v_model)

    # Data preparation for feeding the network
    set_x, set_x_test, set_y, set_y_test, train_x, train_y, val_x, \
                    val_y, test_x, test_y= prepare_data(preprocessor, df_copy, w2v_model)


    vae = VAE(pretrained_weights, emdedding_size, vocab_size)

    vae.compile(optimizer = Adam(learning_rate=0.01))
    print(bck.eval(vae.optimizer))
    print(bck.eval(vae.optimizer.lr))
    # vae.summary()

    # Visualization
    visualise_data(set_x, set_x_test, set_y, set_y_test, tsne=False, pca=False, umap=False)

    checkpoint_path = "training_weights_100k/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 save_weights_only=True,
                                 verbose=1)

    vae.save_weights(checkpoint_path.format(epoch=0))

    # # Train the model
    # history = vae.fit(
    #     train_x, train_x,
    #     epochs=50,
    #     batch_size=400,
    #     validation_data=(val_x, val_x),
    #     shuffle=True, callbacks=[checkpoint]
    # ).history

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    vae.load_weights(latest)


    # # Plot the training and validation loss
    # fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    # ax.plot(history['loss'], 'b', label='train_loss', linewidth=2)
    # ax.plot(history['val_loss'], 'r', label='val_loss', linewidth=2)
    # # ax.plot(history['kl_loss'], 'g', label='kl_loss', linewidth=2)
    # # ax.plot(history['reconstruction_loss'], 'o', color='green',  label=' reconstruction loss', linewidth=2)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel("Loss")
    # ax.legend(loc='upper right')
    # plt.show()

    # plot_label_clusters(vae.encoder, train_x, train_y)


    # Test the model
    encoded_inputs = vae.encoder.predict(test_x)
    reconstructions = vae.decoder.predict(encoded_inputs)

    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)
    encoded_inputs = vae.encoder.predict(train_val_x)
    reconstructions_train_val = vae.decoder.predict(encoded_inputs)

    mse_train_val = np.mean(np.power(train_val_x - reconstructions_train_val, 2), axis=1)
    mse_test = np.mean(np.power(test_x - reconstructions, 2), axis=1)

    print("MSE training:", mse_train_val)
    print("MSE test:", mse_test)

    # Find threshold
    mse_all = np.concatenate((mse_train_val, mse_test), axis=0)
    y_all = np.concatenate((train_val_y, test_y), axis=0)
    clf = tree.DecisionTreeClassifier(max_depth=2)  # This is to ensure that we only get a threshold
    clf = clf.fit(mse_all.reshape(-1, 1), y_all)

    dot_data = tree.export_graphviz(clf, class_names=['0', '1'], out_file=None)
    graph = graphviz.Source(dot_data)
    print(graph)

    text_representation = tree.export_text(clf)
    print(text_representation)


    # fig = plt.figure(figsize=(10, 10))
    # _ = tree.plot_tree(clf, class_names=['0', '1'], filled=True)
    # plt.savefig('../plots/01022021/tree_threshold_100k.png')
    # plt.show()

    threshold = 915
    mse_test = np.asarray(mse_test)

    # Get the indeces of the data with high MSE
    reclassifying_indeces = np.argwhere(mse_test > threshold)
    print(len(reclassifying_indeces), " instances must be classified again out of the ",
          mse_test.shape[0] , " testing data." )
    count_debug = 0
    for i in range(len(mse_test)):
        if mse_test[i] > threshold:
            count_debug += 1
    print("data that are > threshold: ", count_debug)

    count_debug = 0
    for i in range(len(mse_test)):
        if mse_test[i] <= threshold:
            count_debug += 1
    print("data that are <= threshold: ", count_debug)


    print("\n...Starting supervised learning for missclassified data...")
    misclassified_X = test_x[reclassifying_indeces[:]]
    misclassified_y = test_y[reclassifying_indeces[:]]
    debug_var = mse_test[reclassifying_indeces[:]]
    print("\n MSE values above threshold that must be classified again : ", len(debug_var))

    misclassified_X = arr = np.squeeze(misclassified_X, -2)
    misclassified_y = misclassified_y.reshape(-1,)
    misclassified_X = PCA(n_components=2).fit_transform(misclassified_X)
    # Random shuffle
    indexes = shuffle(np.arange(misclassified_X.shape[0]))
    misclassified_X = misclassified_X[indexes]
    misclassified_y = misclassified_y[indexes]
    (_x_train, _y_train), (_x_test, _y_test) = preprocessor._split_data(misclassified_X, y_data = misclassified_y)

    # Plot missclassified data
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(misclassified_X[:, 0], misclassified_X[:, 1], c=(misclassified_y), cmap=cm_bright, alpha=0.6)
    plt.title("Instances with MSE values > threshold grouped by class")
    plt.show()

    # Fine tuning SVM with GridSearch
    print("misclassified_X.shape = ", misclassified_X.shape)
    print("misclassified_X.shape = ", misclassified_y.shape)

    # # GridSearch on SVM's parameters
    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, cv=cv)
    # grid.fit(misclassified_X, misclassified_y)

    # C_2d_range = [1e-2, 1, 1e2]
    # gamma_2d_range = [1e-1, 1, 1e1]
    # classifiers = []
    # for C in C_2d_range:
    #     for gamma in gamma_2d_range:
    #         clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    #         clf.fit(misclassified_X, misclassified_y)
    #         classifiers.append((C, gamma, clf))
    # print("The best parameters are %s with a score of %0.2f"
    #       % (grid.best_params_, grid.best_score_))
    #
    # scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
    #                                                      len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    # plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    # plt.xlabel('gamma')
    # plt.ylabel('C')
    # plt.colorbar()
    # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    # plt.yticks(np.arange(len(C_range)), C_range)
    # plt.title('Validation accuracy')
    # plt.show()

    # Visualization of RBF decision surfaces
    svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma=0.1, decision_function_shape='ovo')
    svm_rbf.fit(_x_train, _y_train)
    # title = "SVM with RBF kernel decision surfaces"
    # plot_decision_regions(misclassified_X, misclassified_y, svm_rbf, title=title)
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.show()

    # polynomial_degree_range = [1, 3, 5, 6]
    # # gamma = [ 0.01, 0.1, 1.0, 100.0 ]
    # accuracy_table = pd.DataFrame(columns=['degree', 'Accuracy'])
    # # accuracy_table = pd.DataFrame(columns=['gamma', 'Accuracy'])
    # accuracy_table['Degree'] = polynomial_degree_range
    #
    # plt.figure(figsize=(10, 10))
    #
    # j = 0
    #
    # for i in polynomial_degree_range:
    #     # Apply SVM model to training data
    #     svm_poly = svm.SVC(kernel='poly', degree=i, C=1.0, random_state=0)
    #     svm_poly.fit(_x_train, _y_train)
    #
    #     # Predict using model
    #     y_pred_ = svm_poly.predict(_x_test)
    #
    #     # Saving accuracy score in table
    #     accuracy_table.iloc[j, 1] = accuracy_score(_y_test, y_pred_)
    #     j += 1
    #
    #     title = 'SVM with polynomial Kernel using degree =' + str(i)
    #     # Printing decision regions
    #     plt.subplot(3, 2, j)
    #     plt.subplots_adjust(hspace=0.4)
    #     plot_decision_regions(misclassified_X
    #                           , misclassified_y
    #                           , svm_poly, title
    #                           )
    #
    #     plt.title('SVM with polynomial Kernel using degree = %s' % i)
    #
    # print(accuracy_table)

    degree_range = [0.01, 0.5, 0.1, 1.0, 10.0, 100.0]

    acc_table = pd.DataFrame(columns=['degree', 'Accuracy'])
    acc_table['degree'] = degree_range

    plt.figure(figsize=(10, 10))

    j = 0

    for i in degree_range:
        # Apply SVM model to training data
        svm_rbf_clf = svm.SVC(kernel='rbf', C=i, random_state=0)
        svm_rbf_clf.fit(_x_train, _y_train)

        # Predict using model
        y_pred_ = svm_rbf_clf.predict(_x_test)

        # Saving accuracy score in table
        acc_table.iloc[j, 1] = accuracy_score(_y_test, y_pred_)
        j += 1

        # Printing decision regions
        plt.subplot(3, 2, j)
        plt.subplots_adjust(hspace=0.4)
        plot_decision_regions(X=misclassified_X
                              , y=misclassified_y
                              , classifier=svm_rbf_clf
                              )

        plt.title('RBF Kernel using C = %s' % i)

    print(acc_table)

    # Apply SVM for missclassified data
    _y_pred = svm_rbf.predict(_x_test)

    print("\n ************ Print test metrics ************ \n")
    print('Accuracy of SVM on training set: {:.2f}'
          .format(svm_rbf.score(_x_train, _y_train)))
    print('Accuracy of SVM on testing set: {:.2f}'
          .format(svm_rbf.score(_x_test, _y_test)))

    accuracy = metrics.accuracy_score(_y_test, _y_pred)
    f1_score = metrics.f1_score(_y_test, _y_pred)
    precision = metrics.precision_score(_y_test, _y_pred)
    recall = metrics.recall_score(_y_test, _y_pred)

    print("F1-score: ", round(f1_score * 100, 2), "%")
    print("Precision: ", round(precision * 100, 2), "%")
    print("Recall: ", round(recall * 100, 2), "%")
    print("Accuracy: ", round(accuracy * 100, 2), "%")

    # Classification report
    labelNames = ["clean", "anomalies"]
    print(classification_report(_y_test, _y_pred, target_names=labelNames))
    fpr_roc, tpr_roc, thresholds_roc = metrics.roc_curve(_y_test, _y_pred)

    roc_auc = metrics.auc(fpr_roc, tpr_roc)
    print("AUC = ", roc_auc)

    # Plot ROC curve for SVM
    plt.figure()
    plt.plot(fpr_roc, tpr_roc, color='darkorange',
             lw=2, label='ROC curve ' )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()