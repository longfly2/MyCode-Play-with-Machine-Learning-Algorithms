import numpy as np
import math
from collections import Counter
from .metrics import accuracy_score


class KNNClassifier:

    def __init__(self, k):
        self._X_train = None
        self._y_train = None
        self.k = k

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x) -> int:
        distances = [math.sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = [self._y_train[neighbor] for neighbor in np.argsort(distances)[:self.k]]
        return Counter(nearest).most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self) -> str:
        return "KNN(k=%d)" % self.k
