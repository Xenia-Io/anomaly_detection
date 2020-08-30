import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor


def find_outliers(x_all, printing=False):

    if printing:
        print("x_all shape: ", x_all.shape)
        print("x_all[0]: ", x_all[0])
        print("x_all[200]: ", x_all[200])


    # Create dictionary to save message index and the corresponding message
    x_all_dict = {}

    for i in range(len(x_all)):
        x_all_dict[i] = x_all[i]

    # Copy X array
    x_all_copied = np.empty_like(x_all)
    x_all_copied[:] = x_all

    if printing:
        print("Length of x_all_dict: ", len(x_all_dict))
        print("x_all_copied[0]: ", x_all_copied[0])
        print("x_all_copied[200]: ", x_all_copied[200])

    feature_extractor = FeatureExtractor()
    x_all_trans = feature_extractor.fit_transform(x_all, term_weighting='tf-idf')

    # For each message get the median value for all of its events
    x_all_median = []
    for i in range(len(x_all_trans)):
        x_all_median.append(np.median(x_all_trans[i]))

    # print(x_all_dict)

    if printing:
        print("x_all_median type: ", type(x_all_median))
        print("x_all_median length: ", len(x_all_median))



    if printing:
        print("x_all shape after transformation: ", x_all_trans.shape)  # x_all shape after transformation:

    # Plot
    all_data = pd.DataFrame(np.array(x_all_median), columns=['events'])

    if printing:
        print("Keys of all_data dataframe: ", all_data.keys())
        print("The all_data dataframe: ", all_data)
        print(all_data.columns)

    plt.plot(np.array(x_all_median))
    plt.xlabel('messages')
    plt.ylabel('events')

    if printing:
        print("all_data shape: ", all_data.shape)  # all_data shape:  (500, 1)

    maxInColumns = np.amax(x_all_median, axis=0)
    max_idx = np.argmax(x_all_median)

    if printing:
        print(maxInColumns, max_idx)

    x_all_median = np.array(x_all_median)

    outliers = x_all_median[x_all_median > 0.07]
    outliers_idx = np.nonzero(x_all_median > 0.07)

    if printing:
        print("Values bigger than 0.07 :", outliers)
        print("Their indices are :", outliers_idx)
        print("The actual messages that are considered as outliers: ")

    for j in range(len(list(outliers_idx[0]))):
        index = list(outliers_idx[0])[j]
        if printing:
            print("\n outlier index: ", index, " with message: ", x_all_dict[index])
            print(x_all_copied[index])

    plt.show()

    return x_all_dict, x_all_copied, x_all_trans, x_all_median, all_data, outliers, outliers_idx