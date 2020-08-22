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
    # load_data('logs_lhcb.json')
    (x_train, y_train), (x_test, y_test) = load_data('logs_lhcb.json')
    # print("Here is the x_train: ", x_train)
    print("Shape of x_train: ", x_train.shape)

    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    print("X_train after transformation : ", x_train)