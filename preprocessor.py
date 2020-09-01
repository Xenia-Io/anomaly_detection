import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
import re

class Preprocessor():

    def preprocessing(self, filename, printing=False):

        # Load the dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.df = self.load_data(filename)
        print(self.x_train[320])

        if printing:
            print("Shape of x_train: ", type(self.x_train[0]), self.x_train.shape)
            print("Here is the x_train 349: ", (self.x_train[349]))
            print("Len of x_train: ", len(self.x_train))
            print("Type of x_train: ", type(self.x_train))
            print("type of x_train[0]: ", type(self.x_train[0]))
            print("len of x_train[0]: ", len(self.x_train[0]))

        ###########################################################################
        ###########################################################################

        # Finding outliers
        self.x_all = np.concatenate((self.x_train, self.x_test), axis=0)
        # print("x_all: ", self.x_all)

        self.find_outliers(self.x_all)

        feature_extractor = FeatureExtractor()
        self.x_train = feature_extractor.fit_transform(self.x_train, term_weighting='tf-idf')  # Train data shape: 350-by-80
        self.x_test = feature_extractor.transform(np.array(self.x_test))  # Test data shape: 150-by-80
        self.x_outliers = feature_extractor.fit_transform(self.outliers.astype(np.str), term_weighting='tf-idf')

        # reshape x_outliers if needed
        if (self.x_outliers[0].shape[0]) != (self.x_train[0].shape[0]):
            n = np.abs(len(self.x_outliers[0]) - len(self.x_train[0]))
            z = np.zeros((len(self.x_outliers), n), dtype=self.x_outliers.dtype)
            self.x_outliers = np.concatenate((self.x_outliers, z), axis=1)


    def find_outliers(self, x_all, printing=False):

        if printing:
            print("x_all shape: ", x_all.shape)
            print("x_all[0]: ", x_all[0])
            print("x_all[200]: ", x_all[200])

        # Create dictionary to save message index and the corresponding message
        self.x_all_dict = {}

        for i in range(len(x_all)):
            self.x_all_dict[i] = x_all[i]

        # Copy X array
        self.x_all_copied = np.empty_like(x_all)
        self.x_all_copied[:] = x_all

        if printing:
            print("Length of x_all_dict: ", len(self.x_all_dict))
            print("x_all_copied[0]: ", self.x_all_copied[0])
            print("x_all_copied[200]: ", self.x_all_copied[200])

        feature_extractor = FeatureExtractor()
        self.x_all_trans = feature_extractor.fit_transform(x_all, term_weighting='tf-idf')

        # For each message get the median value for all of its events
        self.x_all_median = []
        for i in range(len(self.x_all_trans)):
            self.x_all_median.append(np.median(self.x_all_trans[i]))

        # print(x_all_dict)

        if printing:
            print("x_all_median type: ", type(self.x_all_median))
            print("x_all_median length: ", len(self.x_all_median))

        if printing:
            print("x_all shape after transformation: ", self.x_all_trans.shape)  # x_all shape after transformation:

        # Plot
        self.all_data = pd.DataFrame(np.array(self.x_all_median), columns=['events'])

        if printing:
            print("Keys of all_data dataframe: ", self.all_data.keys())
            print("The all_data dataframe: ", self.all_data)
            print(self.all_data.columns)

        plt.plot(np.array(self.x_all_median))
        plt.xlabel('messages')
        plt.ylabel('events')

        if printing:
            print("all_data shape: ", self.all_data.shape)  # all_data shape:  (500, 1)

        maxInColumns = np.amax(self.x_all_median, axis=0)
        max_idx = np.argmax(self.x_all_median)

        if printing:
            print(maxInColumns, max_idx)

        self.x_all_median = np.array(self.x_all_median)

        self.outliers = self.x_all_median[self.x_all_median > 0.07]
        self.outliers_idx = np.nonzero(self.x_all_median > 0.07)

        if printing:
            print("Values bigger than 0.07 :", self.outliers)
            print("Their indices are :", self.outliers_idx)
            print("The actual messages that are considered as outliers: ")

        for j in range(len(list(self.outliers_idx[0]))):
            index = list(self.outliers_idx[0])[j]
            if printing:
                print("\n outlier index: ", index, " with message: ", self.x_all_dict[index])
                print(self.x_all_copied[index])

        # plt.show()

        # return self.x_all_dict, self.x_all_copied, self.x_all_trans, x_all_median, all_data, outliers, outliers_idx


    def _split_data(self, x_data, y_data=None, train_ratio=0.7, split_type='uniform'):

        num_train = int(train_ratio * x_data.shape[0])
        # print("x_data.shape : ", x_data.shape)
        # print("num_train: ", num_train)
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