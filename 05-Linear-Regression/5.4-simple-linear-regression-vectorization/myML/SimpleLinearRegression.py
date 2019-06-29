import numpy as np


class SimpleLinearRegression1:
    def __init__(self) -> None:
        """初始化模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        num = 0.0
        d = 0.0
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回y_predict的结果向量"""
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1() a_=%f, b=%f" % (self.a_, self.b_)


import numpy as np


class SimpleLinearRegression2:
    def __init__(self) -> None:
        """初始化模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot(x_train - x_mean)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回y_predict的结果向量"""
        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.a_ * x + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2() a_=%f, b=%f" % (self.a_, self.b_)
