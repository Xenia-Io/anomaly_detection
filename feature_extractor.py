import pandas as pd
import os
import numpy as np
import re
from collections import Counter
from scipy.special import expit
from itertools import compress
from torch.utils.data import DataLoader, Dataset


class FeatureExtractor():

    def __init__(self):
        self.idf_vec = None
        self.mean_vec = None
        self.events = None
        self.term_weighting = None
        self.normalization = None


    def fit_transform(self, X_seq, term_weighting=None, normalization=None):
        """ Fit and transform the data matrix
        Arguments
        ---------
            X_seq: ndarray, log sequences matrix
            term_weighting: None or `tf-idf`
            normalization: None or `zero-mean`

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('\n====== Transformed train data summary ======')
        self.term_weighting = term_weighting
        self.normalization = normalization

        X_counts = []
        print("Shape of X_seq: ", X_seq.shape)
        print("len(X_seq): ", len(X_seq))

        for i in range(len(X_seq)):
            event_counts = Counter(X_seq[i])
            # print("X_seq[",i,"]: ", X_seq[i])
            # print("length of event_counts: ", len(event_counts))
            # print("event_counts: ", event_counts)
            X_counts.append(event_counts)
            # break

        print("Length of X_counts: ", len(X_counts))

        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        X = X_df.values

        # print("X_df.columns: ", X_df.columns)
        # print("X_df values: ", X_df.values)
        num_instance, num_event = X.shape
        print("num of instances: ", num_instance, " and num of events: ", num_event)
        print("Shape of X: ", X.shape)
        # print("X: ", X)

        if self.term_weighting == 'tf-idf':
            print("Shape of X: ", X.shape)
            df_vec = np.sum(X>0, axis=0) # axis = 0 are for columns
            # print("df_vec: ", df_vec)
            # print("shape of df_vec: ", df_vec.shape)
            self.idf_vec = np.log(num_instance / (df_vec + 1e-8))
            # print("Shape of idf_vec: ", self.idf_vec.shape)
            # print("np.tile shape: ", np.tile(self.idf_vec, (num_instance, 1)).shape)
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new


    def transform(self, X_seq):
        """ Transform the data matrix with trained parameters
        Arguments
        ---------
            X: log sequences matrix
            term_weighting: None or `tf-idf`
        Returns
        -------
            X_new: The transformed data matrix
        """
        print('====== Transformed test data summary ======')
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values

        num_instance, num_event = X.shape
        if self.term_weighting == 'tf-idf':
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
        if self.normalization == 'zero-mean':
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        elif self.normalization == 'sigmoid':
            X[X != 0] = expit(X[X != 0])
        X_new = X

        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))

        return X_new