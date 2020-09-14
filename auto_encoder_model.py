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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
sns.set(style='whitegrid', context='notebook')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
import random as rn
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

        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = Dense(X.shape[2], activation='softmax')(L5)
        model = Model(inputs=inputs, outputs=output)

        return model



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
        threshold_value = points[idx]

        return modified_z_score, threshold_value



    def tsne_scatter(self, features, labels, dimensions=2):
        if dimensions not in (2, 3):
            raise ValueError('tsne_scatter can only plot in 2d or 3d')

        # t-SNE dimensionality reduction
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
        plt.show()


if __name__ == "__main__":

    # Preprocessing the dataset
    preprocessor = Preprocessor('logs_for_supervised.json', True, False)
    preprocessor.preprocessing()

    RANDOM_SEED = 42
    TRAINING_SAMPLE = len(preprocessor.x_train)
    VALIDATE_SIZE = 0.2
    # setting random seeds for libraries to ensure reproducibility
    np.random.seed(RANDOM_SEED)
    rn.seed(RANDOM_SEED)

    # Update dataframe with the PCA representation in column 'messages'
    x_all_list = np.concatenate((preprocessor.x_train, preprocessor.x_test), axis=0)
    y_all_list = np.concatenate((preprocessor.y_train, preprocessor.y_test), axis=0)

    for i in preprocessor.df.index:
        preprocessor.df.at[i, "messages"] = x_all_list[i]
    preprocessor.df.labels = y_all_list

    print("EDWWW ---- item 0 ", preprocessor.df['messages'][0], preprocessor.x_train[0], x_all_list[0])
    print("EDWWW ---- item 61", preprocessor.df['messages'][61], preprocessor.x_test[18], x_all_list[61])
    print("EDWWW ---- item 42", preprocessor.df['messages'][42], preprocessor.x_train[42], x_all_list[42])

    # Mapping labels into integers
    mapping = {'INFO': 0, 'WARNING': 0, 'SEVERE': 1}
    preprocessor.df = preprocessor.df.replace({'labels': mapping})

    # Visualize inputs
    # manual parameter
    RATIO_TO_FRAUD = 5

    # splitting by class
    fraud = preprocessor.df[preprocessor.df.labels == 1]
    clean = preprocessor.df[preprocessor.df.labels == 0]
    print(f"""Shape of the datasets:
        clean (rows, cols) = {clean.shape}
        fraud (rows, cols) = {fraud.shape}""")

    # undersample clean transactions
    clean_undersampled = clean.sample(
        int(len(fraud) * RATIO_TO_FRAUD)
    )

    # concatenate with fraud transactions into a single dataframe
    visualisation_initial = pd.concat([fraud, clean_undersampled])
    column_names = list(visualisation_initial.drop('labels', axis=1).columns)

    # isolate features from labels
    features, labels = visualisation_initial.drop('labels', axis=1).values, visualisation_initial.labels.values

    print(f"""The non-fraud dataset has been undersampled from {len(clean):,} to {len(clean_undersampled):,}.
    This represents a ratio of {RATIO_TO_FRAUD}:1 to fraud.""")

    # shuffle our training set
    # clean = clean.sample(frac=1).reset_index(drop=True)

    # training set: exlusively non-fraud transactions
    X_train = clean.iloc[:TRAINING_SAMPLE]
    # print("___HERE : ", X_train.shape, X_train)
    # print("EVI ---- item 0 ", preprocessor.df['messages'][0], preprocessor.x_train[0], X_train['messages'][0])
    # print("EVI ---- item 42 ", preprocessor.df['messages'][42], preprocessor.x_train[42], X_train['messages'][42])
    X_train = clean.iloc[:TRAINING_SAMPLE].drop('labels', axis=1)

    # testing  set: the remaining non-fraud + all the fraud
    X_test = preprocessor.df.loc[TRAINING_SAMPLE: len(x_all_list), :]
    # X_test = clean.iloc[TRAINING_SAMPLE:]
    # print("___HERE : ", X_test.shape, X_test)
    #
    # X_test = clean.iloc[TRAINING_SAMPLE:].append(fraud).sample(frac=1)
    print("___HERE X_test type: ", type(X_test), X_test)
    # print("EVI ---- item 0 ", preprocessor.df['messages'][43], preprocessor.x_test[0], X_test['messages'][43])
    # print("EVI ---- item 1 ", preprocessor.df['messages'][44], preprocessor.x_test[1], X_test['messages'][44])
    # print("EVI ---- item 2 ", preprocessor.df['messages'][45], preprocessor.x_test[2], X_test['messages'][45])


    # print(f"""Our testing set is composed as follows: {X_test.labels.value_counts()}""")

    # X_train, X_validate = train_test_split(X_train,
    #                                        test_size=len(preprocessor.x_test),
    #                                        random_state=RANDOM_SEED)
    #
    # manually splitting the labels from the test df
    X_test, y_test = X_test.drop('labels', axis=1).values, X_test.labels.values
    print("--------X_test labels: ", y_test)
    print("--------X-test : ", X_test)

    print(f"""Shape of the datasets:
        training (rows, cols) = {(preprocessor.x_train.shape)}
        testing  (rows, cols) = {preprocessor.x_test.shape}""")

    print(f"""Shape of the datasets:
         training (rows, cols) = {(X_train.shape)}
         testing  (rows, cols) = {X_test.shape}""")

    # Reshape inputs
    # print("REEEE : ", preprocessor.x_train[0], X_train['messages'][0]) --- typwnei TA IDIA
    preprocessor.x_train = preprocessor.x_train.reshape(preprocessor.x_train.shape[0], 1, preprocessor.x_train.shape[1])
    preprocessor.x_test = preprocessor.x_test.reshape(preprocessor.x_test.shape[0], 1, preprocessor.x_test.shape[1])
    print("preprocessor.x_train shape in AUTO-ENCODER: ", preprocessor.x_train.shape)


    print(f"""Shape of the datasets:
            training (rows, cols) = {(preprocessor.x_train.shape)}
            testing  (rows, cols) = {preprocessor.x_test.shape}""")


    # Build the model
    model_ = Autoencoder(optimizer='adam', loss='mse')
    model = model_.build_autoencoder(preprocessor.x_train)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    data_subset = visualisation_initial.messages.values
    model_.tsne_scatter(data_subset, labels, dimensions=2)


    # Train the model
    history = model.fit(
        preprocessor.x_train, preprocessor.x_train,
        epochs=model_.epochs,
        batch_size=model_.batch_size,
        shuffle=True, validation_split=0.3
    )
    print("XENIA DD : " , preprocessor.x_test.shape)
    # pass the transformed test set through the autoencoder to get the reconstructed result
    reconstructions = model.predict(preprocessor.x_test)
    # calculating the mean squared error reconstruction loss per row in the numpy array
    mse = np.mean(np.power(preprocessor.x_test - reconstructions, 2), axis=1)
    # print("x_test[0] shape: ", preprocessor.x_test[0].shape, preprocessor.x_test[0])
    # print("reconstr[0] shape: ", reconstructions[0].shape, reconstructions[0])
    # print("mse[0]: ", type(mse[0]), mse[0])
    print("prepro y_test: ", preprocessor.y_test)
    print(mse.shape, " -- mse: ", mse)

    # set to whatever you like
    sample_size = 5

    # Mapping labels into integers
    mapping = {'INFO': 0, 'WARNING': 0, 'SEVERE': 1}
    print("DEBUGG 1 : ", preprocessor.y_test)
    preprocessor.y_test = list(map(mapping.get, preprocessor.y_test))
    print("DEBUGG 2 : ", preprocessor.y_test)
    print("DEBUGG 1 : ", type(preprocessor.y_test))

    # showing the reconstruction losses for a subsample of transactions
    # showing the reconstruction losses for a subsample of transactions
    print(f'Mean Squared Error reconstruction losses for {sample_size} clean transactions:')
    print(mse[np.where(y_test == 0)][:sample_size])
    print(f'\nMean Squared Error reconstruction losses for {sample_size} fraudulent transactions:')
    print(mse[np.where(y_test == 1)][:sample_size])

    # adjust this parameter to customise the recall/precision trade-off
    Z_SCORE_THRESHOLD = 3

    # find the outliers on our reconstructions' mean squared errors
    mad_z_scores, threshold_value = model_.detect_mad_outliers(mse, threshold=Z_SCORE_THRESHOLD)
    mad_outliers = (mad_z_scores > Z_SCORE_THRESHOLD).astype(int)
    print("mad outliers of our reconstructions' MSE: ", mad_outliers)

    anomalies = len(mad_outliers[mad_outliers == True])
    total_trades = len(preprocessor.y_test)
    d = (anomalies / total_trades * 100)

    print("MAD Z-score > ", Z_SCORE_THRESHOLD, " is the selected threshold.")
    print("Any trade with a MSRE >= ", threshold_value, " is flagged.")
    print("This results in", anomalies, "detected anomalies, or ", d,"% out of ", total_trades , "trades reported")


    data = np.column_stack((range(len(mse)), mse))
    print("data: ", data)
    # scatter's x & y
    clean_x, clean_y = data[y_test == 0][:, 0], data[y_test == 0][:, 1]
    fraud_x, fraud_y = data[y_test == 1][:, 0], data[y_test == 1][:, 1]
    print("XENIA 1 : ", clean_x, clean_y)
    print("XENIA 2 : ", fraud_x, fraud_y)

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
    ax.set_xlim(0, len(mse))

    # title & labels
    fig.suptitle('Mean Squared Reconstruction Errors & MAD Threshold', fontsize=14)
    ax.set_xlabel('Pseudo Transaction ID\n(Index in List)')
    ax.set_ylabel('Mean Squared Error\n(Log Scale)')

    # orange legend for threshold value
    ax.legend(loc='upper right', prop={'size': 9})

    # display
    fig.show()
    plt.show()