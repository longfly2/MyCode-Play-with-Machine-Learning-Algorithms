import numpy as np
from .metrics import r2_score


class LinearRegression:
    def __init__(self) -> None:
        """初始化模型"""
        self.intercept_ = None
        self.coef_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        X = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y_train)
        self.intercept_ = self._theta[:1]
        self.coef_ = self._theta[1:]
        return self

    def fit_bgd(self, X_train, y_train, eta=0.01, n_iters=1e4, epsilon=1e-8):
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        def J(X_b, y, theta):
            try:
                return np.sum((X_b.dot(theta) - y) ** 2)
            except:
                return float('inf')

        def dJ(X_b, y, theta):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)

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

    def fit_sgd(self, X_train, y_train, n_iter=5, t0=5, t1=50):
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        assert n_iter >= 1, "n_iter >= 1"

        def dJ_sgd(X_b_i, y_i, theta):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2

        def sgd(X_b, y, initial_theta, n_iter=5, t0=5, t1=50):
            theta = initial_theta

            def learning_rate(t):
                return t0 / (t + t1)

            m = len(X_b)
            for iter_cnt in range(n_iter):
                shuffled_indexes = np.random.permutation(m)
                X_b_new = X_b[shuffled_indexes, :]
                y_new = y[shuffled_indexes]
                for i in range(m):
                    gradient = dJ_sgd(X_b_new[i], y_new[i], theta)
                    theta = theta - learning_rate(iter_cnt * m + i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        y = y_train
        initial_theta = np.zeros((X_b.shape[1],))
        self._theta = sgd(X_b, y, initial_theta, n_iter, t0=5, t1=50)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        X = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X.dot(self._theta)

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression1() interception_=%f, coef_=%f" % (self.intercept_, self.coef_)
