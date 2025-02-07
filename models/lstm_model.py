from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from pre_processor import Preprocessor
import tensorflow.keras.backend as K
from kerastuner import HyperModel
from sklearn import metrics
import kerastuner as kt
import tensorflow as tf
from utils import *

sns.set(style='whitegrid', context='notebook')


class LSTM_Model(HyperModel):

    def __init__(self, pretrained_weights, vocab_size, epochs=30, optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,
                 epsilon=0, w_decay=0, momentum=0, dropout=0, embed_size=300, max_features=10000, maxlen=200):
        """
        Credits to: https://gist.github.com/maxim5/c35ef2238ae708ccb0e55624e9e0252b
        """
        super().__init__()
        # self.optimizer = optimizer
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
        self.vocab_size = vocab_size
        self.pretrained_weights = pretrained_weights


    def build(self, hp):

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

        hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-5, 1e-4])
        optimizer = hp.Choice('optimizer', ['adam', 'adagrad', 'sgd', 'rmsprop', 'adadelta'])

        model.compile(optimizer, loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m])

        return model


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("tf version: ", tf.__version__)
    print("tf.keras version: ", tf.keras.__version__)

    # Preprocessing the dataset
    # preprocessor = Preprocessor('../data/dataset2.json', True, visualize=False)
    # df = preprocessor.load_data_supervised('../data/dataset2.json')
    preprocessor = Preprocessor('../data/dataset_100k.json', True, visualize=False)
    df = preprocessor.load_data_supervised('../data/dataset_100k.json')
    df_copy = df.copy()
    messages = df['messages'].values

    for i in range(messages.shape[0]):
        messages[i] = ''.join([i for i in messages[i] if not i.isdigit()])
        messages[i] = gensim.utils.simple_preprocess (messages[i])

    w2v_model, pretrained_weights, vocab_size, emdedding_size = build_NLP_model(messages)
    # test_NLP_model(w2v_model)

    # Data preparation for feeding the network
    set_x, set_x_test, set_y, set_y_test, train_x, train_y, val_x, \
    val_y, test_x, test_y = prepare_data(preprocessor, df_copy, w2v_model, is_LSTM=True)

    # Compile model
    lstm = LSTM_Model(pretrained_weights, vocab_size)

    tuner = kt.tuners.Hyperband(
        lstm,
        objective='val_loss',
        max_epochs=2,
        directory='my_dir_100k')

    # use one-hot labels
    train_y_onehot = (np.eye(2)[train_y])
    val_y_onehot = (np.eye(2)[val_y])
    test_y_onehot = (np.eye(2)[test_y])

    tuner.search(train_x, train_y_onehot, validation_data=(val_x, val_y_onehot), epochs=2)

    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print("Tuner summary: ", tuner.results_summary())
    print("Best trial: ", tuner.oracle.get_best_trials(num_trials=2)[0].hyperparameters.values)
    print("best optimizer : ", best_hyperparameters.get('optimizer'))
    print("best learning_rate: ", best_hyperparameters.get('learning_rate'))
    print("Best model summary:")
    best_model.summary()

    # Visualization
    visualise_data(set_x, set_x_test, set_y, set_y_test, tsne=False, pca=False, umap=False)

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

    # Test the model
    scores = best_model.evaluate(test_x, test_y_onehot, verbose=0)
    loss_train, accuracy_train, _,_,_ = best_model.evaluate(train_x, train_y_onehot, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy_train))
    loss_test, accuracy_test, _, _, _= best_model.evaluate(test_x, test_y_onehot, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy_test))
    print("Percentage of Test Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = best_model.predict(test_x)
    predictions = tf.one_hot(tf.argmax(predictions, axis=1), depth=2)

    # Evaluate the model
    loss, accuracy, f1_score, precision, recall = best_model.evaluate(test_x, test_y_onehot, verbose=0)

    print("\n ************ Print test metrics ************ \n")
    print("F1-score: ", round(f1_score*100, 6), "%")
    print("Precision: ", round(precision*100, 6), "%")
    print("Recall: ", round(recall*100, 6), "%")
    print("Loss: ", round(loss*100, 6), "%")
    print("Accuracy: ", round(accuracy*100, 6), "%")

    # Initialize the label names
    labelNames = ["clean", "anomalies"]

    # Classification report
    print(classification_report(test_y_onehot, predictions,
                                target_names=labelNames))

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
