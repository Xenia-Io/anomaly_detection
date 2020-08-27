import pandas as pd
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import json
from dataloader import *
from feature_extractor import *
from svm_model import SVM
import sklearn
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

def main():
    # reading the JSON data
    filename = 'logs_lhcb.json'

    with open(filename) as json_file:
        data = json.load(json_file)
        print(type(data))
        print("total tooks", data['took'], " and response tooks: ", data['responses'][0]['took'])
        print("total of hits: ", data['responses'][0]['hits']['total'])
        print("max_score of hits: ", data['responses'][0]['hits']['max_score'])
        print("size of hits of hits: ", len(data['responses'][0]['hits']['hits']))
        print("hits of hits: ", data['responses'][0]['hits']['hits'][0])
        print("hits of hits: ", data['responses'][0]['hits']['hits'][499])



        # for p in data['hits']:
        #     print(type(p['_source']['message']))
        #     print('from hit: ' + p['_source']['message'])
            # print('id from hit: ' + p['_id'] + " || " + p['_source']['message'])

    # create a data_df
    # df = pd.read_json (filename,orient='index')
    print(type(data['responses'][0]['hits']['hits']))
    # construct list of '_sources'
    sources_list = []
    print(type(sources_list))
    for item in range(len(data['responses'][0]['hits']['hits'])):
        sources_list.append(data['responses'][0]['hits']['hits'][item]['_source'])
        # print(item, type(data['responses'][0]['hits']['hits'][item]['_source']), data['responses'][0]['hits']['hits'][item]['_source'])

    data_df = pd.DataFrame.from_dict(sources_list)

    print("List of sources: ", sources_list)
    print("Keys of dataframe: " , data_df.keys())
    # print(data_df)
    # print(type(data_df.physical_host))
    # print("data_df.kubernetes: " , data_df.kubernetes.tolist())
    # vectorizer = TfidfVectorizer()
    # x = vectorizer.fit_transform(data_df)
    # print(x.shape)
    # print(vectorizer.get_feature_names())
    # print(x.toarray())


    pipe = Pipeline([('count', CountVectorizer()), ('tfid', TfidfTransformer())]).fit(data_df)
    # print(pipe['count'].transform(data_df).toarray())
    print(pipe['tfid'].idf_)

    print(pipe.transform(data_df).shape)



if __name__ == "__main__":
    # main()

    (x_train, y_train), (x_test, y_test) = load_data('logs_lhcb.json')

    ############################################################################
    ######################  Printing values  ###################################
    print("Here is the x_train 0: ", (x_train[0]))
    print("Here is the x_train 349: ", (x_train[349]))
    print("Len of x_train: ", len(x_train))
    print("Type of x_train: ", type(x_train))
    print("type of x_train[0]: ", type(x_train[0]))
    print("len of x_train[0]: ", len(x_train[0]))
    # for i in range(len(x_train)):
    #     print(i, len(x_train[i]))
        # for j in range(len(x_train[i])):
        #     print("x_train[",i,"][",j,"]")
        # break
    ###########################################################################
    ###########################################################################

    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    ############################################################################
    ######################  Printing values  ###################################
    print("X_train after transformation : ", x_train)
    print("Shape of X_train after transformation : ", x_train.shape)
    print("len of x_train[0]: ", len(x_train[0]))
    print("\n values of x_train[0]: ", x_train[0])
    print("\n values of x_train[344]: ", x_train[344])
    ############################################################################

    # Plot before return
    print(plt.rcParams.get('figure.figsize'))

    fig_size = plt.rcParams["figure.figsize"]

    fig_size[0] = 10

    fig_size[1] = 8

    plt.rcParams["figure.figsize"] = fig_size
    new_data = pd.DataFrame(np.array(x_train))
    new_data.plot(style='o')
    plt.show()

    # print('Starting fitting Isolation Forests')
    model = IsolationForest(n_estimators=10, warm_start=True)
    model.fit(x_train)  # fit 10 trees
    model.set_params(n_estimators=20)  # add 10 more trees
    y_predictions = model.predict(x_test) # fit the added trees
    print("Y: ", y_predictions )

    print("Shape of x_test: ", x_test)
    print("Shape of x_train: ", x_train.shape)
    print("Shape of y: ", y_predictions.shape)




    # new, 'normal' observations ----
    print("Accuracy:", list(y_predictions).count(1) / y_predictions.shape[0])

    # outliers ----
    # print("Accuracy:", list(y_outliers).count(-1) / y_outliers.shape[0])


    # model = SVM()
    # model.fit(x_train, x_train[0] )

    # print('Train validation:')
    # precision, recall, f1 = model.evaluate(x_train, y_train)
    #
    # print('Test validation:')
    # precision, recall, f1 = model.evaluate(x_test, y_test)

