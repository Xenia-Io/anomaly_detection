import pandas as pd
import os
import numpy as np
import re
import json
from sklearn.utils import shuffle
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import umap

def _split_data(x_data, y_data=None, train_ratio=0.7, split_type='uniform'):

    num_train = int(train_ratio * x_data.shape[0])
    print("x_data.shape : ", x_data.shape)
    print("num_train: ", num_train)
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


def load_data(log_file, label_file=None, window='session', train_ratio=0.7, \
              split_type='sequential', save_csv=False, window_size=0, printing=True):
    """ Load HDFS structured log into train and test data
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
        print("Loading", type(log_file))

        with open(log_file) as json_file:
            data = json.load(json_file)

            # construct list of '_sources'
            sources_list = []
            for item in range(len(data['responses'][0]['hits']['hits'])):
                sources_list.append(data['responses'][0]['hits']['hits'][item]['_source'])

        # Build my dataframe
        data_df = pd.DataFrame(sources_list)
        x_data = data_df['message'].values

        if printing:
            print("type of x_data: ", type(x_data))
            print("length of x_data[0]: ", len(x_data[0]))
            print("length of x_data[499]: ", len(x_data[499]))
            print("shape of x_data: ", x_data.shape)
            print("The keys of dataframe: ", data_df.keys())
            print("data_df: ", (data_df))

        # Split train and test data
        (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
        print('Total: {} instances, train: {} instances, test: {} instances'.format(
            x_data.shape[0], x_train.shape[0], x_test.shape[0]))

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        # print("Sum for train and test instances: ", x_train.sum(), x_test.sum())

        return (x_train, None), (x_test, None)

    else:
        raise NotImplementedError('load_data() only support json files!')



def load_data2(log_file, label_file=None, window='session', train_ratio=0.7, \
              split_type='sequential', save_csv=False, window_size=0):
    """ Load HDFS structured log into train and test data
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

    sources_list = []
    dictionary = {}  # dictionary to store the id and the source

    if log_file.endswith('.json'):
        assert window == 'session', "Only window=session is supported for our dataset."
        print("Loading", type(log_file))
        data_dict = OrderedDict()

        with open(log_file) as json_file:
            data = json.load(json_file)

            # construct list of '_sources'

            for item in range(len(data['responses'][0]['hits']['hits'])):
                sources_list.append(data['responses'][0]['hits']['hits'][item]['_source'])
                id_key = data['responses'][0]['hits']['hits'][item]['_id']
                source_value = data['responses'][0]['hits']['hits'][item]['_source']
                dictionary.update({id_key: source_value})

        print("Dictionary has", len(dictionary.keys()), "keys", type(dictionary.keys()))

        data_df = pd.DataFrame(sources_list)
        id_list = list(dictionary.keys())
        id_set = set(id_list)

        # for idx, row in data_df.iterrows():
            # print(idx, " row: ", row['message'])
        for id in id_set:
            if not id in data_dict:
                data_dict[id] = []
            # data_dict[id].append(dictionary[id]['message'])
            data_dict[id].append(dictionary[id]['message'])
            # data_dict[id].append(row['message'])

        data_df2 = pd.DataFrame(list(data_dict.items()), columns=['Id', 'EventSequence'])

        # Split train and test data
        x_data = data_df2['EventSequence'].values
        print("data_df2: ", (data_df2))
        print(len(x_data[0]))
        print(" shape of x_data[0][0]: ", len(x_data[0][0]))
        print(" shape of x_data[499][0]: ", len(x_data[499][0]))
        print("The keys of dataframe: ", data_df.keys())

        (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
        print('Total: {} instances, train: {} instances, test: {} instances'.format(
            x_data.shape[0], x_train.shape[0], x_test.shape[0]))

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        # print("Sum for train and test instances: ", x_train.sum(), x_test.sum())

        return (x_train, None), (x_test, None)

    else:
        raise NotImplementedError('load_data() only support json files!')