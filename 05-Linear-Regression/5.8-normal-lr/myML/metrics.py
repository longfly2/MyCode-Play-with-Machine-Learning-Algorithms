import numpy as np
import math


def accuracy_score(y_true, y_predict):
    return np.sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'len must be the same.'
    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    return math.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'len must be the same.'
    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    assert len(y_true) == len(y_predict), 'len must be the same.'
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
