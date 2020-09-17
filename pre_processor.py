import pandas as pd
import numpy as np
import json
from umap import UMAP
from sklearn.utils import shuffle
from feature_extractor import FeatureExtractor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import re


class Preprocessor():

    def __init__(self, filename, supervised, visualize= True, auto_encoder = True):
        self.supervised = supervised
        self.visualize = visualize

        if auto_encoder:
            self.df = self.load_data_supervised(filename)
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.df = self.load_data(filename)
            self.x_all = np.concatenate((self.x_train, self.x_test), axis=0)
            self.x_all_trans_no_pca = []
            self.x_all_median = []


    def preprocessing(self, printing=False):

        feature_extractor = FeatureExtractor()
        self.x_all = feature_extractor.fit_transform(self.x_all, term_weighting='tf-idf')
        self.x_train = feature_extractor.fit_transform(self.x_train, term_weighting='tf-idf')
        self.x_test = feature_extractor.transform(np.array(self.x_test))
        self.x_all_trans_no_pca = np.copy(self.x_all)

        if printing:
            print("-------- AFTER FEATURE EXTRACTION ------------")
            print("Shape of x_all : ", self.x_all.shape)
            print("Shape of x_test : ", self.x_test.shape)
            print("Shape of copied and un-transformed x_all : ", self.x_all_trans_no_pca.shape)
            print("Sample of copied and un-transformed x_all: ", self.x_all_trans_no_pca[0])
            print("Sample of x_all (no PCA applied in x_all till now though): ", self.x_all[0])

        # Apply dimensionality reduction
        if self.supervised:
            scaler = MinMaxScaler()
            scaler.fit(self.x_train)
            x_train = scaler.transform(self.x_train)
            x_test = scaler.transform(self.x_test)
            self.x_train = self.apply_Dim_Reduction(x_train, apply_pca=False, apply_tSNE=True, apply_umap=False)
            self.x_test = self.apply_Dim_Reduction(x_test, apply_pca=False, apply_tSNE=True, apply_umap=False)
        else:
            scaler = StandardScaler()
            self.x_all = scaler.fit_transform(self.x_all)
            print("is supervised ? : " , self.supervised)
            self.x_all = self.apply_Dim_Reduction(self.x_all, apply_pca=True, apply_tSNE=False, apply_umap=False)

        # Visualization of the data
        if self.visualize:
            if self.supervised:
                self.visualize_inputs_(self.x_all_trans_no_pca)
                self.visualize_simple_inputs()
                self.visualize_pca_inputs_sup()
                self.visualize_tsne_inputs_sup(self.x_all.shape[0])
            else:
                self.visualize_inputs_(self.x_all_trans_no_pca)
                self.visualize_simple_inputs()
                self.visualize_pca_inputs()
                self.visualize_tsne_inputs(self.x_all.shape[0])

    # Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
    def plot_wordcloud(self, text, mask=None, max_words=200, max_font_size=100, figure_size=(12.0, 10.0),
                       title=None, title_size=20, image_color=False):
        stopwords = set(STOPWORDS)
        more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
        stopwords = stopwords.union(more_stopwords)

        wordcloud = WordCloud(background_color='black',
                              stopwords=stopwords,
                              max_words=max_words,
                              max_font_size=max_font_size,
                              random_state=42,
                              width=800,
                              height=400,
                              mask=mask)
        wordcloud.generate(str(text))

        plt.figure(figsize=figure_size)
        if image_color:
            image_colors = ImageColorGenerator(mask);
            plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
            plt.title(title, fontdict={'size': title_size,
                                       'verticalalignment': 'bottom'})
        else:
            plt.imshow(wordcloud);
            plt.title(title, fontdict={'size': title_size, 'color': 'black',
                                       'verticalalignment': 'bottom'})
        plt.axis('off');
        plt.tight_layout()
        plt.show()


    def load_data_supervised(self, log_file, train_ratio=0.7, printing=True):
        print('====== Start loading the data ======')

        if log_file.endswith('.json'):
            with open(log_file) as json_file:
                data = json.load(json_file)

                # construct list of '_sources'
                dict_of_lists = {}
                messages = []
                labels = []
                for item in range(len(data['responses'][0]['hits']['hits'])):
                    labels.append(data['responses'][0]['hits']['hits'][item]['_source']['severity'])
                    messages.append(data['responses'][0]['hits']['hits'][item]['_source']['message'])
                dict_of_lists['messages'] = messages
                dict_of_lists['labels'] = labels

            # Build the dataframe
            df = pd.DataFrame(dict_of_lists, columns=['messages', 'labels'])

            # Clean the messages from numbers and unwanted characters
            x_data = df['messages'].values
            for i in range(x_data.shape[0]):
                x_data[i] = re.sub("[\(\[].*?[\)\]]", "", x_data[i])
                x_data[i] = ''.join([i for i in x_data[i] if not i.isdigit()])
            df['messages'] = x_data

            # Mapping labels into integers
            mapping = {'info': 0, 'warning': 0, 'notice': 0, 'severe': 1}
            df = df.replace({'labels': mapping})

            if printing:
                print("Number of log messages in the dataset: ", df.shape[0])
                print("Value counts for clean and fraud samples:\n", df.labels.value_counts())
                print(df[df.labels==1].head(6))
                print("Percentage of fraudulent logs:{}%".format(round((sum(df.labels == 1) * 100/len(df.labels)), 4)))

            if self.visualize:
                self.plot_wordcloud(df[df.labels == 0]["messages"], title="Word Cloud of normal messages")
                self.plot_wordcloud(df[df.labels == 1]["messages"], title="Word Cloud of fraudulent messages")

        else:
            raise NotImplementedError('Function only supports json files!')

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
            if self.supervised:
                print("Loading", log_file)

                with open(log_file) as json_file:
                    data = json.load(json_file)

                    # construct list of '_sources'
                    dict_of_lists = {}
                    messages = []
                    labels = []
                    for item in range(len(data['responses'][0]['hits']['hits'])):
                        labels.append(data['responses'][0]['hits']['hits'][item]['_source']['severity'])
                        messages.append(data['responses'][0]['hits']['hits'][item]['_source']['message'])
                    dict_of_lists['messages'] = messages
                    dict_of_lists['labels'] = labels

                # Build the dataframe
                df = pd.DataFrame(dict_of_lists, columns=['messages', 'labels'])
                x_data = df['messages'].values

                # Clean the messages from numbers and unwanted characters
                for i in range(x_data.shape[0]):
                    # print(i, x_data[i])
                    x_data[i] = re.sub("[\(\[].*?[\)\]]", "", x_data[i])
                    x_data[i] = ''.join([i for i in x_data[i] if not i.isdigit()])
                    # print(i, x_data[i])
                df['messages'] = x_data

                # Split train and test data
                (x_train, y_train), (x_test, y_test) = self._split_data(x_data, y_data = df['labels'].values, \
                                                                        train_ratio=train_ratio, split_type=split_type)
                print('Total: {} instances, train: {} instances, test: {} instances'.format(
                    x_data.shape[0], x_train.shape[0], x_test.shape[0]))

                if save_csv:
                    df.to_csv('data_instances.csv', index=False)

                # print("Sum for train and test instances: ", x_train.sum(), x_test.sum())

                # Update dataframe after shuffling
                x_all_list = np.concatenate((x_train, x_test), axis=0)
                y_all_list = np.concatenate((y_train, y_test), axis=0)

                df['messages'] = x_all_list
                df['labels'] = y_all_list

                # Mapping labels into integers
                mapping = {'info': 0, 'warning': 0, 'notice': 0, 'severe': 1}
                df = df.replace({'labels': mapping})
                return (x_train, y_train), (x_test, y_test), df

            else:
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

                # Clean the messages from numbers and unwanted characters
                for i in range(x_data.shape[0]):
                    # print(i, x_data[i])
                    x_data[i] = re.sub("[\(\[].*?[\)\]]", "", x_data[i])
                    x_data[i] = ''.join([i for i in x_data[i] if not i.isdigit()])
                    # print(i, x_data[i])

                data_df['message'] = x_data

                if printing:
                    print("type of x_data: ", type(x_data))
                    print("length of x_data[0]: ", len(x_data[0]))
                    print("length of x_data[499]: ", len(x_data[499]))
                    print("shape of x_data: ", x_data.shape)
                    print("The keys of dataframe: ", data_df.keys())
                    print("data_df: ", (data_df))

                # Split train and test data - Shuffle train data
                (x_train, _), (x_test, _) = self._split_data(x_data, train_ratio=train_ratio, split_type=split_type)
                print('Total: {} instances, train: {} instances, test: {} instances'.format(
                    x_data.shape[0], x_train.shape[0], x_test.shape[0]))

                if save_csv:
                    data_df.to_csv('data_instances.csv', index=False)

                # print("Sum for train and test instances: ", x_train.sum(), x_test.sum())

                # Update dataframe after shuffling
                x_all_list = np.concatenate((x_train, x_test), axis=0)
                data_df['message'] = x_all_list

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


    def apply_Dim_Reduction(self, X, apply_pca=True, apply_tSNE=False, apply_umap=False):

        if apply_pca:
            print("Starting PCA Analysis...")
            X = PCA(n_components=2).fit_transform(X)
        elif apply_tSNE:
            print("Starting t-SNE Analysis...")
            X = TSNE(n_components=2).fit_transform(X)
        else:
            print("Starting UMAP Analysis...")
            X = UMAP(n_neighbors=15,
                      min_dist=0.1,
                      metric='correlation').fit_transform(X)
        return X


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

    def visualize_inputs_(self, X):
        X_PCA = PCA(n_components=5).fit_transform(X)
        X_TSNE = TSNE().fit_transform(X)
        X_UMAP = UMAP(n_neighbors=15,
                      min_dist=0.1,
                      metric='correlation').fit_transform(X)
        fig = plt.figure(figsize=(40, 30))
        plt.subplot(2, 2, 1)
        plt.scatter(X_PCA[:, 0], X_PCA[:, 1], cmap='Set1')
        plt.title("Principal Component Analysis", fontsize=10)
        plt.subplot(2, 2, 2)
        plt.scatter(X_UMAP[:, 0], X_UMAP[:, 1], cmap='Set1')
        plt.title("Uniform Manifold Approximation and Projections", fontsize=10)
        plt.subplot(2, 2, 3)
        plt.scatter(X_TSNE[:, 0], X_TSNE[:, 1], cmap='Set1')
        plt.title("t-Distributed Stochastic Neighbor Embedding", fontsize=10)
        plt.show()


    def visualize_pca_inputs_sup(self):
        features = ['feature' + str(i) for i in range(self.x_all_trans_no_pca.shape[1])]

        df = pd.DataFrame(self.x_all_trans_no_pca, columns=features)
        print("debuggg 1 ------- ", self.df['labels'].values)
        df['y'] = self.df['labels']
        df['label'] = df['y'].apply(lambda i: str(i))
        print('Size of the dataframe: {}'.format(self.df.shape))

        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[features].values)
        df['pca-one'] = pca_result[:, 0]
        df['pca-two'] = pca_result[:, 1]
        df['pca-three'] = pca_result[:, 2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        colors = ["#ff0b04", "#4374b3"]  # Set your custom color palette
        customPalette = sns.set_palette(sns.color_palette(colors))
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x="pca-one", y="pca-two", hue="y", palette=customPalette,
                        data=df.loc[rndperm, :], legend="full", alpha=0.3)
        plt.title("Input data after PCA in 2d plot for labeled data")
        plt.show()

        colors = {'0': 'red', '1': 'blue'}
        ax = plt.figure(figsize=(10, 10)).gca(projection='3d')
        ax.scatter(
            xs = df.loc[rndperm, :]["pca-one"],
            ys = df.loc[rndperm, :]["pca-two"],
            zs = df.loc[rndperm, :]["pca-three"],c=df['label'].apply(lambda x: colors[x])
        )

        ax.set_xlabel('pca-one')
        ax.set_ylabel('pca-two')
        ax.set_zlabel('pca-three')
        plt.title("Input data after PCA in 3d plot for labeled data")
        plt.show()

    def visualize_pca_inputs(self):

        features = ['feature' + str(i) for i in range(self.x_all_trans_no_pca.shape[1])]
        df = pd.DataFrame(self.x_all_trans_no_pca,columns=features)
        print('Size of the dataframe: {}'.format(df.shape))

        # For re-producability of the results
        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        # Apply PCA
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

    def visualize_tsne_inputs_sup(self, N):
        # For re-producability of the results
        features = ['feature' + str(i) for i in range(self.x_all_trans_no_pca.shape[1])]
        df = pd.DataFrame(self.x_all_trans_no_pca, columns=features)
        df['y'] = self.df['labels']
        df['label'] = df['y'].apply(lambda i: str(i))
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
        plt.figure(figsize=(10, 10))
        colors = ["#ff0b04", "#4374b3"]  # Set your custom color palette
        customPalette = sns.set_palette(sns.color_palette(colors))
        sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y",palette=customPalette,
                        data=df_subset, legend="full", alpha=0.9)
        plt.title("Input data after t-SNE in 2d plot for labeled data")
        plt.show()

        plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(1, 2, 1)
        sns.scatterplot(x="pca-one", y="pca-two", hue="y",palette=customPalette,
                        data=df_subset, legend="full", alpha=0.9, ax=ax1)
        ax2 = plt.subplot(1, 2, 2)
        sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="y",palette=customPalette,
                        data=df_subset, legend="full", alpha=0.9, ax=ax2)
        plt.suptitle("PCA vs t-SNE for labeled data")
        plt.show()


    def visualize_tsne_inputs(self, N):
        # For re-producability of the results
        features = ['feature' + str(i) for i in range(self.x_all_trans_no_pca.shape[1])]
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
        plt.figure(figsize=(10, 10))
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
