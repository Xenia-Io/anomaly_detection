from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class FeatureExtractor():

    def __init__(self, analyzer = 'word'):
        self.analyzer = analyzer


    def fit_transform(self, X_seq):
        """ Fit and transform the dataset matrix - create feature vector representations
        Arguments
        ---------
            X_seq: ndarray, log sequences matrix

        Returns
        -------
            X_new: The transformed data matrix
        """
        print('\n====== Transformed data summary ======')

        # Instantiate the vectorizer object
        vectorizer = TfidfVectorizer(analyzer= self.analyzer)

        # Create tokens from all dataset matrix
        count_wm = vectorizer.fit_transform(X_seq)
        count_tokens = vectorizer.get_feature_names()

        # DF: [100000 rows x 801 columns] (for dataset1 = 100K)
        # [each row = 1 log message] , [each column = 1 word]
        df_countvect = pd.DataFrame(data=count_wm.toarray(), columns=count_tokens)

        print(".Count Vectorizer results.\n")
        print(df_countvect)

        # Print the vector representation for a log message (print 1 row from df)
        print("DEBUG_0 : " ,df_countvect.loc[[20500]])

        # Get the first position of the maximum value for each word
        m = df_countvect.ne(0).idxmax()
        df = pd.DataFrame(dict(pos=m, val=df_countvect.lookup(m, m.index)))
        print(df)

        print('All data shape: {}-by-{}\n'.format(df_countvect.shape[0], df_countvect.shape[1]))

        X_new = df_countvect
        return X_new
