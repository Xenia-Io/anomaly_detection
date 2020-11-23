from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import roc_curve, auc
import tensorflow.keras.backend as K
from kerastuner import HyperModel
from sklearn import metrics
import kerastuner as kt
import tensorflow as tf
import seaborn as sns
from utils import *
import pandas as pd
import numpy as np
import gensim

from sklearn.metrics import classification_report, confusion_matrix

sns.set(style='whitegrid', context='notebook')


class LSTM_Model(HyperModel):

    def __init__(self, pretrained_weights, epochs=6, optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,
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

        def get_f1(y_true, y_pred):  # taken from old keras source code
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            recall = true_positives / (possible_positives + K.epsilon())
            f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
            return f1_val

        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        print("... Start building LSTM model ...")
        fc_units = self.emdedding_size/2

        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.emdedding_size, weights=[self.pretrained_weights]))
        model.add(LSTM(units=self.emdedding_size))
        model.add(Dropout(0.3))
        model.add(Dense(units=fc_units))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=2))
        model.add(Activation('softmax'))

        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta'])
        model.compile(optimizer, loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

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
        median = np.median(points, axis=0)

        # calculate the absolute difference of each data point to the calculated median
        deviation = np.abs(points - median)
        print("median = ", median)
        print("deviation: ", deviation)
        print("np.median(deviation): ", np.median(deviation))

        # take the median of those absolute differences
        med_abs_deviation = np.median(deviation)

        # 0.6745 is the 0.75th quartile of the standard normal distribution,
        # to which the MAD converges.
        modified_z_score = 0.6745 * deviation / med_abs_deviation

        # return as extra information what the original mse value was at which the threshold is hit
        # need to find a way to compute this mathematically, but I'll just use the index of the nearest candidate for now
        idx = (np.abs(modified_z_score - threshold)).argmin()

        if idx >= len(points):
            idx = np.argmin(points)

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
    val_y, test_x, test_y = prepare_data(preprocessor, df_copy, w2v_model, is_LSTM=True)

    # Compile model
    lstm = LSTM_Model(pretrained_weights)
    # model_ = lstm.build_lstm(train_x, vocab_size, emdedding_size, pretrained_weights)

    # tuner = kt.tuners.RandomSearch(
    #     lstm,
    #     objective='val_loss',
    #     max_trials=20,
    #     directory='my_dir')

    tuner = kt.tuners.Hyperband(
        lstm,
        objective='val_loss',
        max_epochs=5,
        directory='my_dir')

    # tuner = kt.tuners.BayesianOptimization(
    #         lstm,
    #         objective='val_loss',
    #         max_trials=10,
    #         seed=42,
    #         executions_per_trial=5,
    #         directory='my_dir'
    #     )

    # use one-hot labels
    train_y_onehot = (np.eye(2)[train_y]) # shape = (5956, 2)
    val_y_onehot = (np.eye(2)[val_y])     # shape = (1050, 2)
    test_y_onehot = (np.eye(2)[test_y])   # shape = (3006, 2)

    tuner.search(train_x, train_y_onehot,
                 validation_data=(val_x, val_y_onehot),
                 epochs=5)

    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print("Tuner summary: ", tuner.results_summary())
    print("Best trial: ", tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values)

    best_model.summary()
    print("best optimizer : ", best_hyperparameters.get('optimizer'))
    print("best learning_rate: ", best_hyperparameters.get('learning_rate'))
    print("test_x.shape: ", test_x.shape)
    # exit(0)

    # Visualization
    visualise_data(set_x, set_x_test, set_y, set_y_test, tsne=False, pca=False, umap=False)

    # dot_img_file = 'model_2_lstm.png'
    # tf.keras.utils.plot_model(model_, to_file=dot_img_file, show_shapes=True)

    # Train the model
    history = best_model.fit(
        train_x, train_y_onehot,
        epochs=lstm.epochs,
        batch_size=lstm.batch_size,
        shuffle=True, validation_data=(val_x, val_y_onehot)
    ).history

    # Plot the training and validation loss
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')

    # fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    # ax.plot(history['get_f1'], 'b', label='Train', linewidth=2)
    # ax.plot(history['val_get_f1'], 'r', label='Validation', linewidth=2)
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel("Accuracy")
    # ax.legend(loc='upper right')

    plt.show()

    # exit(0)

    # Test the model
    # scores = best_model.evaluate(test_x, test_y_onehot, verbose=0)
    # loss_train, accuracy_train = best_model.evaluate(train_x, train_y_onehot, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy_train))
    # loss_test, accuracy_test = best_model.evaluate(test_x, test_y_onehot, verbose=False)
    # print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    # print("Percentage of Test Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = best_model.predict(test_x)

    print("Shape of predictions and test_y: ", predictions.shape, test_y_onehot.shape)
    predictions = tf.one_hot(tf.argmax(predictions, axis=1), depth=2)

    corrects_ = 0
    for i in range(len(test_y_onehot)):
        if((test_y_onehot[i] == tf.keras.backend.get_value(predictions[i])).all()):
            # print(test_y_onehot[i], "  - target // ", tf.keras.backend.get_value(predictions[i]), " - prediction")

            corrects_ = corrects_ + 1
        else:
            print(type(test_y_onehot[i]))
            print("prediction is wrong in position ", i, " with prediction: ", \
            tf.keras.backend.get_value(predictions[i]), " and target " ,test_y_onehot[i])

    print("correct predictions = ", corrects_ , " totall = ", len(test_y_onehot))
    accuracy_test_ = round((corrects_ / len(test_y_onehot))*100, 4)
    print("Test accuracy: ", accuracy_test_, "%")

    # evaluate the model
    loss, accuracy, f1_score, precision, recall = best_model.evaluate(test_x, test_y_onehot, verbose=0)

    print("\n ************ Print test metrics ************ \n")
    print("F1-score: ", round(f1_score*100, 2), "%")
    print("Precision: ", round(precision*100, 2), "%")
    print("Recall: ", round(recall*100, 2), "%")
    print("Loss: ", round(loss*100, 2), "%")
    print("Accuracy: ", round(accuracy*100, 2), "%")

    test_y = np.argmax(test_y_onehot, axis=-1)  # getting the labels
    y_prediction = np.argmax(predictions, axis=-1)  # getting the confidence of postive class
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(test_y, y_prediction)

    roc_auc = metrics.auc(fpr_roc, tpr_roc)
    print("auc = ", roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr_roc, tpr_roc, color='darkorange',
             lw=lw, label='ROC curve ' )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
