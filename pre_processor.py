import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler
import time
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import re


class Preprocessor():

    def __init__(self, filename):
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.df = self.load_data(filename)
        self.x_all = np.concatenate((self.x_train, self.x_test), axis=0)
        self.x_all_trans_no_pca = []
        self.x_all_median = []


    def preprocessing(self, printing=False, visualize=False):

        feature_extractor = FeatureExtractor()
        self.x_all = feature_extractor.fit_transform(self.x_all, term_weighting='tf-idf') # x_all shape: (500, 67)
        self.x_train = feature_extractor.fit_transform(self.x_train, term_weighting='tf-idf')  #x_train shape: (350,67)
        self.x_test = feature_extractor.transform(np.array(self.x_test))  # Test data shape: (150, 67)
        self.x_all_trans_no_pca = np.copy(self.x_all)

        if printing:
            print("-------- AFTER FEATURE EXTRACTION ------------")
            print("Shape of x_all : ", self.x_all.shape)
            print("Shape of x_test : ", self.x_test.shape)
            print("Shape of copied and un-transformed x_all : ", self.x_all_trans_no_pca.shape)
            print("Sample of copied and un-transformed x_all: ", self.x_all_trans_no_pca[0])

        # Apply dimensionality reduction
        self.x_all = self.apply_PCA(self.x_all)

        # Visualization of the data
        if visualize:
            self.visualize_simple_inputs()
            self.visualize_pca_inputs()
            self.visualize_tsne_inputs(self.x_all.shape[0])


    def load_data(self, log_file, label_file=None, window='session', train_ratio=0.7, \
                  split_type='sequential', save_csv=False, window_size=0, printing=False):
        """ Load the logs into train and test data
        Arguments
        ---------
            log_file: str, the file path of structured log.
            label_file: str, the file path of anomaly labels, None for unlabeled data
            window: str, the window options including `session` (default).
            train_ratio: float, the ratio of training data for train/test split.
            split_type: `sequential` means to split the data sequentially without label_file.

        Returns
        -------
            (x_train, y_train): the training data
            (x_test, y_test): the testing data
        """

        print('====== Start loading the data ======')

        if log_file.endswith('.json'):
            assert window == 'session', "Only window=session is supported for our dataset."
            print("Loading", log_file)

            with open(log_file) as json_file:
                data = json.load(json_file)

                # construct list of '_sources'
                sources_list = []
                for item in range(len(data['responses'][0]['hits']['hits'])):
                    sources_list.append(data['responses'][0]['hits']['hits'][item]['_source'])

            # Build my dataframe
            data_df = pd.DataFrame(sources_list)
            x_data = data_df['message'].values
            print("x_data shape: ", x_data.shape)

            for i in range(x_data.shape[0]):
                # print(i, x_data[i])
                x_data[i] = re.sub("[\(\[].*?[\)\]]", "", x_data[i])
                x_data[i] = ''.join([i for i in x_data[i] if not i.isdigit()])
                # print(i, x_data[i])

            data_df['message'] = x_data
            # print("here: ", data_df['message'])

            if printing:
                print("type of x_data: ", type(x_data))
                print("length of x_data[0]: ", len(x_data[0]))
                print("length of x_data[499]: ", len(x_data[499]))
                print("shape of x_data: ", x_data.shape)
                print("The keys of dataframe: ", data_df.keys())
                print("data_df: ", (data_df))

            # Split train and test data
            (x_train, _), (x_test, _) = self._split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                x_data.shape[0], x_train.shape[0], x_test.shape[0]))

            if save_csv:
                data_df.to_csv('data_instances.csv', index=False)

            # print("Sum for train and test instances: ", x_train.sum(), x_test.sum())

            return (x_train, None), (x_test, None), data_df

        else:
            raise NotImplementedError('load_data() only support json files!')


    def _split_data(self, x_data, y_data=None, train_ratio=0.7, split_type='uniform'):

        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]

        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]

        # Random shuffle
        indexes = shuffle(np.arange(x_train.shape[0]))
        x_train = x_train[indexes]
        if y_train is not None:
            y_train = y_train[indexes]

        return (x_train, y_train), (x_test, y_test)


    def apply_PCA(self, x_all):
        scaler = StandardScaler()
        scaler.fit(x_all)
        x_all = scaler.transform(x_all)

        # Make an instance of the Model
        pca = PCA(n_components=2)
        pca.fit(x_all)
        x_all = pca.transform(x_all)
        print("Number of components PCA choose after fitting: ", pca.n_components_)

        return x_all


    def visualize_simple_inputs(self):
        """ Visualize the input data after feature extraction and before PCA """

        # For each message get the median value of all its events
        for i in range(len(self.x_all_trans_no_pca)):
            self.x_all_median.append(np.median(self.x_all_trans_no_pca[i]))

        all_data_df = pd.DataFrame(np.array(self.x_all_median), columns=['events'])

        plt.plot(np.array(all_data_df))
        plt.xlabel('messages')
        plt.ylabel('events')
        plt.title("Input data after feature extraction")
        plt.show()


    def visualize_pca_inputs(self):
        # For re-producability of the results
        features = ['pixel' + str(i) for i in range(self.x_all_trans_no_pca.shape[1])]
        df = pd.DataFrame(self.x_all_trans_no_pca,columns=features)
        print('Size of the dataframe: {}'.format(df.shape))
        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[features].values)
        df['pca-one'] = pca_result[:, 0]
        df['pca-two'] = pca_result[:, 1]
        df['pca-three'] = pca_result[:, 2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        plt.figure(figsize=(10, 10))
        sns.scatterplot(x="pca-one", y="pca-two", palette=sns.color_palette("hls", 10),
                        data=df.loc[rndperm, :], legend="full", alpha=0.3)
        plt.title("Input data after PCA in 2d plot")
        plt.show()

        ax = plt.figure(figsize=(10, 10)).gca(projection='3d')
        ax.scatter(
            xs=df.loc[rndperm, :]["pca-one"],
            ys=df.loc[rndperm, :]["pca-two"],
            zs=df.loc[rndperm, :]["pca-three"],
            cmap='tab10'
        )
        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.title("Input data after PCA in 3d plot")
        plt.show()


    def visualize_tsne_inputs(self, N):
        # For re-producability of the results
        features = ['pixel' + str(i) for i in range(self.x_all_trans_no_pca.shape[1])]
        df = pd.DataFrame(self.x_all_trans_no_pca, columns=features)
        print('Size of the dataframe: {}'.format(df.shape))
        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        df_subset = df.loc[rndperm[:N], :].copy()
        data_subset = df_subset[features].values
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_subset)
        df_subset['pca-one'] = pca_result[:, 0]
        df_subset['pca-two'] = pca_result[:, 1]
        df_subset['pca-three'] = pca_result[:, 2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        df_subset['tsne-2d-one'] = tsne_results[:, 0]
        df_subset['tsne-2d-two'] = tsne_results[:, 1]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",  palette=sns.color_palette("hls", 10),
                        data=df_subset, legend="full", alpha=0.3)
        plt.title("Input data after t-SNE in 2d plot")
        plt.show()

        print(df_subset.keys())
        plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(1, 2, 1)
        sns.scatterplot(x="pca-one", y="pca-two", palette=sns.color_palette("hls", 10),
                        data=df_subset, legend="full", alpha=0.3, ax=ax1)
        ax2 = plt.subplot(1, 2, 2)
        sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", palette=sns.color_palette("hls", 10),
                        data=df_subset, legend="full", alpha=0.3, ax=ax2)
        plt.suptitle("PCA vs t-SNE")
        plt.show()
