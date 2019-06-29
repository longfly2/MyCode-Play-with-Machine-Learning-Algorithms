import numpy as np
from .metrics import r2_score


class LinearRegression:
    def __init__(self) -> None:
        """初始化模型"""
        self.interception_ = None
        self.coef_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        X = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)
        self.interception_ = self._theta[:1]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        X = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression1() interception_=%f, coef_=%f" % (self.interception_, self.coef_)
