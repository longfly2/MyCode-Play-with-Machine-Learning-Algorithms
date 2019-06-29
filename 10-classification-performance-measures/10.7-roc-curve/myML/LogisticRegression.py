import numpy as np
from .metrics import accuracy_score


class LogisticRegression:
    def __init__(self) -> None:
        """初始化模型"""
        self.intercept_ = None
        self.coef_ = None
        self._theta = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4, epsilon=1e-8):
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        def J(X_b, y, theta):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(X_b, y, theta):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient(X_b, y, initial_theta, eta=0.01, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            iter_cnt = 0
            while iter_cnt < n_iters:
                last_theta = theta
                gradient = dJ(X_b, y, theta)
                theta = theta - eta * gradient
                if abs(J(X_b, y, theta) - J(X_b, y, last_theta)) < epsilon:
                    break
                iter_cnt += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        y = y_train
        theta = np.zeros((X_b.shape[1],))
        self._theta = gradient(X_b, y, theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict_proba(self, X_predict):
        X = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X.dot(self._theta))

    def predict(self, X_predict):
        y_proba = self.predict_proba(X_predict)
        return np.array(y_proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression1() interception_=%f, coef_=%f" % (self.intercept_, self.coef_)
