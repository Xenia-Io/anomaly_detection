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
from umap import UMAP
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

    def __init__(self, epochs= 10, optimizer='adam', loss='mae', batch_size=0, kernel_init=0, gamma=0,\
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


    def tsne_scatter(self, features, labels, dimensions=2):
        if dimensions not in (2, 3):
            raise ValueError('tsne_scatter can only plot in 2d or 3d')

        # dimensionality reduction
        features_embedded = UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(list(features))
        # features_embedded = TSNE(n_components=dimensions, random_state=RANDOM_SEED).fit_transform(list(features))

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
    preprocessor = Preprocessor('big_dataset.json', True, True)
    preprocessor.preprocessing()
    print("DEBUG_ AUTOENCODER AFTER PREPROCESSING: ", preprocessor.x_train.shape, preprocessor.x_test.shape)

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


    # Mapping labels into integers
    mapping = {'info': 0, 'warning': 0, 'notice': 0, 'severe': 1}
    preprocessor.df = preprocessor.df.replace({'labels': mapping})

    # Visualize inputs
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
    X_train = clean.iloc[:TRAINING_SAMPLE].drop('labels', axis=1)

    # testing  set: the remaining non-fraud + all the fraud
    X_test = preprocessor.df.loc[TRAINING_SAMPLE: len(x_all_list), :]
    # X_test = clean.iloc[TRAINING_SAMPLE:].append(fraud).sample(frac=1)

    print(f"""Our testing set is composed as follows: {X_test.labels.value_counts()}""")
    print(list(X_test['labels'].values))
    # manually splitting the labels from the test df
    X_test, y_test = X_test.drop('labels', axis=1).values, X_test.labels.values

    print(f"""Shape of the datasets:
        training (rows, cols) = {(preprocessor.x_train.shape)}
        testing  (rows, cols) = {preprocessor.x_test.shape}""")

    print(f"""Shape of the datasets:
         training (rows, cols) = {(X_train.shape)}
         testing  (rows, cols) = {X_test.shape}""")

    # configure our pipeline
    pipeline = Pipeline([('normalizer', Normalizer()),
                         ('scaler', MinMaxScaler())])
    pipeline.fit(preprocessor.x_train)
    preprocessor.x_train = pipeline.transform(preprocessor.x_train)

    # Reshape inputs
    # print("x_train vs X_train : ", preprocessor.x_train[0], X_train['messages'][0]) --- typwnei TA IDIA
    preprocessor.x_train = preprocessor.x_train.reshape(preprocessor.x_train.shape[0], 1, preprocessor.x_train.shape[1])
    preprocessor.x_test = preprocessor.x_test.reshape(preprocessor.x_test.shape[0], 1, preprocessor.x_test.shape[1])


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
    ).history

    # Plot the training and validation loss
    fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    plt.show()

    # pass the transformed test set through the autoencoder to get the reconstructed result
    reconstructions = model.predict(preprocessor.x_test)
    # calculating the mean squared error reconstruction loss per row in the numpy array
    mse = np.mean(np.power(preprocessor.x_test - reconstructions, 2), axis=1)

    # set to whatever you like
    sample_size = 5

    # Mapping labels into integers
    mapping = {'info': 0, 'warning': 0, 'notice': 0, 'severe': 1}
    preprocessor.y_test = list(map(mapping.get, preprocessor.y_test))

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
    # scatter's x & y
    clean_x, clean_y = data[y_test == 0][:, 0], data[y_test == 0][:, 1]
    fraud_x, fraud_y = data[y_test == 1][:, 0], data[y_test == 1][:, 1]
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
    ax.set_xlim(0, (len(mse)+100))

    # title & labels
    fig.suptitle('Mean Squared Reconstruction Errors & MAD Threshold', fontsize=14)
    ax.set_xlabel('Pseudo Message ID\n(Index in MSE List)')
    ax.set_ylabel('Mean Squared Error\n(Log Scale)')

    # orange legend for threshold value
    ax.legend(loc='lower left', prop={'size': 9})

    # display
    fig.show()
    plt.show()