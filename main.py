import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as pp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import json
from testing import *
from pandas.plotting import scatter_matrix
from dataloader import *
from feature_extractor import *
from svm_model import SVM
import sklearn
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from preprocessor import Preprocessor
from utils import find_outliers


# def preprocessing():

def main():

    preprocessor = Preprocessor()
    preprocessor.preprocessing('logs_lhcb.json')



    print('Starting fitting Isolation Forests')
    model = IsolationForest(n_estimators=100, warm_start=True)
    model.fit(preprocessor.x_train)  # fit 10 trees
    model.set_params(n_estimators=20)  # add 10 more trees

    print("Shape of x_test: ", preprocessor.x_test.shape)
    print("Shape of x_train: ", preprocessor.x_train.shape)
    print("Shape of x_outliers: ", preprocessor.x_outliers.shape)


    # Make predictions
    y_predictions = model.predict(preprocessor.x_test)  # fit the added trees
    y_pred_outliers = model.predict(preprocessor.x_outliers)
    print("Y: ", y_predictions)
    print("Y outliers: ", y_pred_outliers)


    # Accuracy for predictions
    print("Accuracy for predictions:", list(y_predictions).count(1) / y_predictions.shape[0])

    # Accuracy for outliers
    print("Accuracy for outliers:", list(y_pred_outliers).count(-1) / y_pred_outliers.shape[0])




    # # Load the dataset
    # (x_train, y_train), (x_test, y_test) = load_data('logs_lhcb.json')
    #
    # ############################################################################
    # ######################  Printing values  ###################################
    # print("Shape of x_train: ", type(x_train[0]), x_train.shape)
    # # print("Here is the x_train 349: ", (x_train[349]))
    # # print("Len of x_train: ", len(x_train))
    # # print("Type of x_train: ", type(x_train))
    # # print("type of x_train[0]: ", type(x_train[0]))
    # # print("len of x_train[0]: ", len(x_train[0]))
    #
    # ###########################################################################
    # ###########################################################################
    #
    # # Finding outliers
    # x_all = np.concatenate((x_train, x_test), axis=0)
    #
    # x_all_dict,x_all_copied,x_all_trans,x_all_median,all_data_df,outliers, outliers_idx = find_outliers(x_all, False)
    #
    #
    # feature_extractor = FeatureExtractor()
    # x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')  # Train data shape: 350-by-80
    # x_test = feature_extractor.transform(x_test)  # Test data shape: 150-by-80
    # x_outliers = feature_extractor.fit_transform(outliers.astype(np.str), term_weighting='tf-idf')




if __name__ == "__main__":
    main()
    # testing()



