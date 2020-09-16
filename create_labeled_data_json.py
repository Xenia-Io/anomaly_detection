import numpy as np
import pandas as pd
import json

def main():
    with open('big_dataset.json') as json_file:
        data = json.load(json_file)
        # construct list of '_sources'
        dict_of_lists = {}
        messages = []
        labels = []
        sourc = []
        for item in range(len(data['responses'][0]['hits']['hits'])):
            sourc.append(data['responses'][0]['hits']['hits'][item]['_source'])
            labels.append(data['responses'][0]['hits']['hits'][item]['_source']['severity'])
            messages.append(data['responses'][0]['hits']['hits'][item]['_source']['message'])
        dict_of_lists['messages'] = messages
        dict_of_lists['labels'] = labels
        dict_of_lists['sourc'] = sourc

        print(len(messages), len(labels))
        # Build the dataframe
        df = pd.DataFrame(dict_of_lists, columns=['messages', 'labels', 'sourc'])
        print("df keys: ", df.columns)
        x_data = df['labels'].values
        print(len(x_data))
        print(df.labels.unique())
        grouped_df = df.agg({"labels": "nunique"})
        print(grouped_df)
        print(df['labels'].value_counts())
        print(df['sourc'])
        print(len(sourc))



        df['messages'] = x_data




if __name__ == "__main__":
    main()


