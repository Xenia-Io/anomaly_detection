# importing libaries ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest# Generating data ----


def testing():

    rng = np.random.RandomState(42)

    # Generating training data
    X_train = 0.2 * rng.randn(1000, 2)
    # print(X_train.shape, type(X_train))
    # print(X_train)
    X_train = np.r_[X_train + 3, X_train]
    print(X_train.shape, type(X_train))
    # print(X_train)
    maxInColumns = np.amax(X_train, axis=0)
    print('Max value of every column: ', maxInColumns)
    X_train = pd.DataFrame(X_train, columns = ['x1', 'x2'])

    # print(X_train.shape, type(X_train))
    # print(X_train)

    # X_train = X_train.cumsum()
    # X_train.plot()
    # plt.show()

    # Generating new, 'normal' observation
    X_test = 0.2 * rng.randn(200, 2)
    X_test = np.r_[X_test + 3, X_test]
    X_test = pd.DataFrame(X_test, columns = ['x1', 'x2'])

    # Generating outliers
    X_outliers = rng.uniform(low=-1, high=5, size=(50, 2))
    X_outliers = pd.DataFrame(X_outliers, columns = ['x1', 'x2'])