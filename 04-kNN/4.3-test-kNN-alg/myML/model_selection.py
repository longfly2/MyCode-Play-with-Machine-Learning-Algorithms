import numpy as np


def train_test_split(X, y, test_size=0.2):
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    X_train = X[shuffled_indexes[test_size:]]
    X_test = X[shuffled_indexes[:test_size]]
    y_train = y[shuffled_indexes[test_size:]]
    y_test = y[shuffled_indexes[:test_size]]
    return X_train, X_test, y_train, y_test
